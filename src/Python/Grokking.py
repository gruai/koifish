"""
Grokking tutorial: a minimal reproduction of the experiments in
"Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets"
(Power et al., 2022) -- the paper behind https://github.com/openai/grok

Idea
----
Train a small transformer on a modular-arithmetic table (e.g. x / y mod 97),
using only a fraction of all (x, y) pairs for training. The model first
memorizes the training set (train acc -> 100%, val acc ~ chance), then, MUCH
later, validation accuracy suddenly jumps to ~100%. That delayed jump is
"grokking". Strong weight decay is what makes it happen reliably.

Run:
    pip install torch matplotlib
    python grokking_tutorial.py                       # x / y mod 97, 50% train
    python grokking_tutorial.py --op add --steps 20000
    python grokking_tutorial.py --train_frac 0.3      # smaller data => later grokking

Paper setup (approx.): 2-layer decoder-only transformer, width 128, 4 heads,
AdamW (lr 1e-3, weight decay 1, betas 0.9/0.98), linear warmup, batch 512.
"""

import argparse
import itertools
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# 1. Data: the full operation table for a binary op mod p
# ----------------------------------------------------------------------------
def build_dataset(op: str, p: int):
    """Return all equations as tensors of tokens and labels.

    Vocabulary: 0..p-1 are the numbers, p is the op token, p+1 is '='.
    Each example is the 4-token sequence [x, op, y, =]; the label is x∘y mod p.
    """
    OP, EQ = p, p + 1
    pairs = []
    for x, y in itertools.product(range(p), range(p)):
        if op == "div" and y == 0:  # division by zero is undefined
            continue
        if op == "add":
            r = (x + y) % p
        elif op == "sub":
            r = (x - y) % p
        elif op == "mul":
            r = (x * y) % p
        elif op == "div":
            r = (x * pow(y, p - 2, p)) % p  # y^-1 via Fermat's little theorem
        else:
            raise ValueError(op)
        pairs.append((x, OP, y, EQ, r))
    data = torch.tensor(pairs, dtype=torch.long)
    print(data)
    return data[:, :4], data[:, 4]  # inputs, labels


def split(inputs, labels, train_frac, seed):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(inputs), generator=g)
    n_train = int(train_frac * len(inputs))
    tr, va = perm[:n_train], perm[n_train:]
    print(tr)
    print(va)
    return (inputs[tr], labels[tr]), (inputs[va], labels[va])


# ----------------------------------------------------------------------------
# 2. Model: small causal (decoder-only) transformer
# ----------------------------------------------------------------------------
class TinyTransformer(nn.Module):
    def __init__(self, vocab, n_out, d_model=128, n_heads=4, n_layers=2, seq_len=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # pre-LayerNorm, like GPT
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_out, bias=False)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)  # causal mask

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))
        h = self.blocks(h, mask=self.mask)
        h = self.ln(h[:, -1])  # the '=' position predicts the answer
        return self.head(h)


# ----------------------------------------------------------------------------
# 3. Training loop
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, x, y):
    model.eval()
    logits = model(x)
    loss = F.cross_entropy(logits, y).item()
    acc = (logits.argmax(-1) == y).float().mean().item()
    model.train()
    return loss, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", default="div", choices=["add", "sub", "mul", "div"])
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--train_frac", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1.0)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}")

    inputs, labels = build_dataset(args.op, args.p)
    (xtr, ytr), (xva, yva) = split(inputs, labels, args.train_frac, args.seed)
    xtr, ytr, xva, yva = (t.to(device) for t in (xtr, ytr, xva, yva))
    print(f"train={len(xtr)}  val={len(xva)}  op={args.op} mod {args.p}")

    model = TinyTransformer(vocab=args.p + 2, n_out=args.p).to(device)
    print(model)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.98),
        weight_decay=args.weight_decay,
    )
    print(opt)
    # linear warmup, then constant LR
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / args.warmup))
    print(sched)

    hist = {"step": [], "train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    n = len(xtr)
    perm, ptr = torch.randperm(n, device=device), 0

    for step in range(1, args.steps + 1):
        # draw a minibatch (reshuffle each epoch)
        if ptr + args.batch_size > n:
            perm, ptr = torch.randperm(n, device=device), 0
        idx = perm[ptr:ptr + args.batch_size]
        ptr += args.batch_size

        loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % args.eval_every == 0 or step == 1:
            tl, ta = evaluate(model, xtr, ytr)
            vl, va = evaluate(model, xva, yva)
            for k, v in zip(hist, (step, ta, va, tl, vl)):
                hist[k].append(v)
            if step % (args.eval_every) == 0 or step == 1:
                print(f"step {step:6d} | train loss {tl:.4f} acc {ta:.3f} "
                      f"| val loss {vl:.4f} acc {va:.3f}")
            if va > 0.99 and ta > 0.99:
                print(f"Grokked at step {step}: val acc {va:.3f}")
                break

    if not args.no_plot:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(hist["step"], hist["train_acc"], label="train")
        ax[0].plot(hist["step"], hist["val_acc"], label="val")
        # ax[0].set_xscale("log"); 
        ax[0].set_xlabel("optimization step (log)")
        ax[0].set_ylabel("accuracy"); ax[0].legend(); ax[0].set_title("Accuracy")
        ax[1].plot(hist["step"], hist["train_loss"], label="train")
        ax[1].plot(hist["step"], hist["val_loss"], label="val")
        # ax[1].set_xscale("log"); 
        ax[1].set_xlabel("optimization step (log)")
        ax[1].set_ylabel("loss"); ax[1].legend(); ax[1].set_title("Loss")
        fig.suptitle(f"Grokking: x {args.op} y mod {args.p}, "
                     f"B{args.batch_size} LR{args.lr} split={args.train_frac}, wd={args.weight_decay}")
        fig.tight_layout()
        fig.savefig("docs/grokking/loss_curves.png", dpi=150)
        print("saved grokking_curves.png")


if __name__ == "__main__":
    main()