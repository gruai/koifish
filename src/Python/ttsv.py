"""
TTSV (Test-Time Steering Vectors, "Model Whisper", AAAI 2026) for single-token
yes/no classification on BoolQ with a Qwen3 model.

Paper: arXiv 2512.04748. Method in one paragraph:
  * Learn V in R^{L x d} (L=20 in the paper, d = hidden size) and prepend it to the
    input *embeddings*: E' = [V ; E(x)]. The LLM stays frozen.
  * Optimise V by minimising the entropy of the model's output distribution on
    unlabeled data (no labels used). AdamW, lr 1e-3 -> 1e-5 linear decay,
    weight decay 1e-8, eps 1e-5, batch 16, 20 epochs.
  * Inference: the one fixed V is prepended to every input.

Adaptation made here (the paper evaluates long generations on math, not 1-token
classification): the "generated token" is just the first answer token, so the
loss is the entropy of the next-token distribution at the last prompt position.
No sampling/generation is needed during training.

Usage examples:
  python ttsv_boolq.py --model Qwen/Qwen3-1.7B
  python ttsv_boolq.py --model Qwen/Qwen3-1.7B --fit_split validation   # transductive, as in the paper
  python ttsv_boolq.py --model Qwen/Qwen3-1.7B --load_V ttsv_boolq.pt   # reuse a saved V
"""
import argparse
import math
import random
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

SYSTEM = (
    "You will be given a passage from Wikipedia and a yes/no question about it. "
    "Decide whether the correct answer to the question is yes or no, based on the passage. "
    "Reply 'true' if the answer is yes and 'false' if the answer is no. "
    "Reply with that single word only."
)


# --------------------------------------------------------------------------- data
def build_prompt(tok, passage, question):
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Passage: {passage}\n\nQuestion: Based on the passage, {question}?"},
    ]
    # enable_thinking=False appends the empty <think></think> block for Qwen3
    return tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def encode_split(tok, ds):
    prompts = [build_prompt(tok, ex["passage"], ex["question"]) for ex in ds]
    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
    labels = torch.tensor([bool(ex["answer"]) for ex in ds])
    return ids, labels


def collate(id_lists, pad_id, device):
    """Left-pad so that the last position of every row is the answer position."""
    m = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), m), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), m), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, m - len(x):] = torch.tensor(x)
        mask[i, m - len(x):] = 1
    return ids.to(device), mask.to(device)


def single_token_ids(tok, words):
    out = []
    for w in words:
        t = tok.encode(w, add_special_tokens=False)
        if len(t) == 1:
            out.append(t[0])
    return sorted(set(out))


# --------------------------------------------------------------------------- model
def last_logits(model, ids, mask, V=None):
    """Logits at the last position, with the steering vectors V prepended (if given).

    This is the whole method as far as the forward pass is concerned:
      1. look up token embeddings,
      2. concatenate V in front,
      3. extend the attention mask and recompute position ids so that padding
         does not change the distance between V and the real tokens.
    """
    emb = model.get_input_embeddings()(ids)  # (B, T, d), model dtype
    B = ids.size(0)
    if V is not None:
        pre = V.to(emb.dtype).unsqueeze(0).expand(B, -1, -1)
        emb = torch.cat([pre, emb], dim=1)
        ones = torch.ones(B, V.size(0), dtype=mask.dtype, device=mask.device)
        mask = torch.cat([ones, mask], dim=1)
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    out = model.model(inputs_embeds=emb, attention_mask=mask, position_ids=pos, use_cache=False)
    h = out.last_hidden_state[:, -1]  # only the last position -> avoids B x T x vocab logits
    return model.lm_head(h).float()


def label_scores(logits, true_ids, false_ids):
    lp = logits.log_softmax(-1)
    return torch.logsumexp(lp[:, true_ids], -1) - torch.logsumexp(lp[:, false_ids], -1)


def entropy(logits, label_ids=None):
    """Entropy (nats) of the next-token distribution. Full vocab by default (as in the paper)."""
    if label_ids is not None:
        lp = logits[:, label_ids].log_softmax(-1)
    else:
        lp = logits.log_softmax(-1)
    return -(lp.exp() * lp).sum(-1)


def init_V(model, L, mode, device):
    d = model.config.hidden_size
    if mode == "normal":  # paper's setting for Qwen models: N(0, 1)
        return torch.randn(L, d, device=device)
    W = model.get_input_embeddings().weight.detach().float()  # paper's setting for Llama: data statistics
    return W.mean(0) + W.std(0) * torch.randn(L, d, device=device)


# --------------------------------------------------------------------------- fit
def fit_ttsv(model, fit_ids, pad_id, true_ids, false_ids, args, device):
    V = nn.Parameter(init_V(model, args.L, args.init, device))
    opt = torch.optim.AdamW([V], lr=args.lr, weight_decay=1e-8, eps=1e-5)
    n = len(fit_ids)
    steps_per_epoch = math.ceil(n / args.bs)
    total = args.epochs * steps_per_epoch
    end_ratio = args.lr_end / args.lr
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: 1.0 - (1.0 - end_ratio) * min(s / max(total, 1), 1.0)
    )
    label_ids = (true_ids + false_ids) if args.entropy_over == "labels" else None

    model.eval()  # dropout off; parameters are frozen anyway
    step = 0
    for ep in range(args.epochs):
        order = list(range(n))
        random.shuffle(order)
        ent_sum = ptrue_sum = cnt = 0.0
        for s in range(0, n, args.bs):
            idx = order[s:s + args.bs]
            ids, mask = collate([fit_ids[i] for i in idx], pad_id, device)
            logits = last_logits(model, ids, mask, V)
            H = entropy(logits, label_ids)
            loss = H.mean()
            if args.balance_weight > 0:  # optional, NOT in the paper: avoid collapsing to one label
                sc = label_scores(logits, true_ids, false_ids)
                pbar = torch.sigmoid(sc).mean().clamp(1e-6, 1 - 1e-6)
                marg = -(pbar * pbar.log() + (1 - pbar) * (1 - pbar).log())
                loss = loss - args.balance_weight * marg
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            with torch.no_grad():
                sc = label_scores(logits, true_ids, false_ids)
                ent_sum += H.sum().item()
                ptrue_sum += (sc > 0).float().sum().item()
                cnt += len(idx)
        print(f"[fit] epoch {ep + 1:>2}/{args.epochs}  entropy {ent_sum / cnt:.4f}  "
              f"frac_pred_true {ptrue_sum / cnt:.3f}  lr {sched.get_last_lr()[0]:.2e}", flush=True)
    return V.detach()


# --------------------------------------------------------------------------- eval
@torch.no_grad()
def scores_for(model, id_lists, pad_id, true_ids, false_ids, V, bs, device):
    n = len(id_lists)
    order = sorted(range(n), key=lambda i: len(id_lists[i]))
    out = torch.empty(n)
    model.eval()
    for s in range(0, n, bs):
        idx = order[s:s + bs]
        ids, mask = collate([id_lists[i] for i in idx], pad_id, device)
        sc = label_scores(last_logits(model, ids, mask, V), true_ids, false_ids)
        out[torch.tensor(idx)] = sc.cpu()
    return out


def auc(scores, y):
    """Rank-based AUC (ties ignored)."""
    s = scores.double()
    ranks = torch.empty_like(s)
    ranks[s.argsort()] = torch.arange(1, len(s) + 1, dtype=torch.double)
    npos = int(y.sum())
    nneg = len(y) - npos
    return ((ranks[y].sum() - npos * (npos + 1) / 2) / (npos * nneg)).item()


def best_threshold(scores, y):
    cand = scores.sort().values
    accs = torch.stack([((scores > t) == y).float().mean() for t in cand])
    return cand[accs.argmax()].item()


def report(name, eval_scores, eval_y, calib_scores, calib_y):
    thr = best_threshold(calib_scores, calib_y)
    acc0 = ((eval_scores > 0) == eval_y).float().mean().item()
    acct = ((eval_scores > thr) == eval_y).float().mean().item()
    print(f"[{name:>8}] acc@0 {acc0:.4f} | acc@calib_thr({thr:+.2f}) {acct:.4f} | "
          f"AUC {auc(eval_scores, eval_y):.4f} | frac_pred_true@0 {(eval_scores > 0).float().mean():.3f}")


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--L", type=int, default=20, help="number of steering vectors")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--eval_bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_end", type=float, default=1e-5)
    ap.add_argument("--init", choices=["normal", "embed"], default="normal")
    ap.add_argument("--entropy_over", choices=["vocab", "labels"], default="vocab")
    ap.add_argument("--balance_weight", type=float, default=0.0)
    ap.add_argument("--fit_split", choices=["train", "validation"], default="train",
                    help="'train': fit on unlabeled train prompts, evaluate on validation (clean). "
                         "'validation': fit on the unlabeled eval prompts themselves (paper's setting).")
    ap.add_argument("--fit_n", type=int, default=2000)
    ap.add_argument("--calib_n", type=int, default=500, help="labeled train examples used only to pick a threshold")
    ap.add_argument("--eval_n", type=int, default=0, help="0 = full validation split (3270)")
    ap.add_argument("--load_V", default="")
    ap.add_argument("--out", default="ttsv_boolq.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if device == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dt)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dt)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)  # the LLM is frozen; only V gets gradients
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    true_ids = single_token_ids(tok, ["true", "True"])
    false_ids = single_token_ids(tok, ["false", "False"])
    assert true_ids and false_ids, "label words are not single tokens for this tokenizer"
    print("true ids:", true_ids, "false ids:", false_ids)

    train = load_dataset("google/boolq", split="train").shuffle(seed=args.seed)
    val = load_dataset("google/boolq", split="validation")
    if args.eval_n:
        val = val.select(range(args.eval_n))
    calib_ids, calib_y = encode_split(tok, train.select(range(args.calib_n)))
    eval_ids, eval_y = encode_split(tok, val)
    if args.fit_split == "train":
        fit_ds = train.select(range(args.calib_n, args.calib_n + args.fit_n))  # disjoint from calib
        fit_ids, _ = encode_split(tok, fit_ds)  # labels are ignored
    else:
        fit_ids = eval_ids  # transductive, unlabeled
    print(f"fit {len(fit_ids)} | calib {len(calib_ids)} | eval {len(eval_ids)}")

    # baseline: no steering vectors
    kw = dict(pad_id=pad_id, true_ids=true_ids, false_ids=false_ids, bs=args.eval_bs, device=device)
    base_c = scores_for(model, calib_ids, V=None, **kw)
    base_e = scores_for(model, eval_ids, V=None, **kw)
    report("baseline", base_e, eval_y, base_c, calib_y)

    if args.load_V:
        V = torch.load(args.load_V)["V"].to(device)
    else:
        t0 = time.time()
        V = fit_ttsv(model, fit_ids, pad_id, true_ids, false_ids, args, device)
        print(f"fit took {time.time() - t0:.0f}s")
        torch.save({"V": V.cpu(), "args": vars(args)}, args.out)
        print("saved", args.out)

    ttsv_c = scores_for(model, calib_ids, V=V, **kw)
    ttsv_e = scores_for(model, eval_ids, V=V, **kw)
    report("TTSV", ttsv_e, eval_y, ttsv_c, calib_y)


if __name__ == "__main__":
    main()