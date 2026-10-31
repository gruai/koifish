"""
Toy demo inspired by "Emergence of Linear Truth Encodings in Language Models"
(Ravfogel, Yehudai, Linzen, Bruna, Bietti; NeurIPS 2025, arXiv:2510.15804).

NOTE: This is my own minimal re-implementation of the *idea*, not the authors'
code, so details (sizes, document format, hyper-parameters) differ from the paper.

Idea ("Truth Co-occurrence Hypothesis"):
  * There are N entities, each with one true attribute (a fixed fact).
  * A document is a sequence of statements  s1 a1 s2 a2 ... sK aK.
  * Within a document, statements are ALL true or ALL false
    (false = a random wrong attribute).  -> true co-occurs with true.
  * A one-layer transformer is trained with next-token prediction on attributes.
  * To predict a_i well for i >= 2, the model should (1) memorise the facts and
    (2) compare memorised facts with what it saw earlier in the document,
    i.e. infer "is this a true or false document?".  That inference is
    expected to show up as a linear "truth direction" in the hidden states.

We then (a) train a linear probe on hidden states at attribute tokens
(held-out entities) and (b) compare with a control where truthfulness is
sampled independently per statement (no co-occurrence => no incentive).

Run:  python truth_encoding_demo.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
dev = "cpu"

# ---------------- config ----------------
N_ENT, N_ATT = 200, 20      # entities / attributes
K = 4                       # statements per document
D = 64                      # model width
STEPS = 6000
BATCH = 256
LR = 3e-3
P_TRUE = 0.5

V = N_ENT + N_ATT           # vocab: entities then attributes
TRUE_ATTR = torch.randint(0, N_ATT, (N_ENT,))   # the "world facts"


def sample_docs(B, cooccur=True):
    """Returns tokens (B, 2K), truth labels (B, K), entity ids (B, K)."""
    ent = torch.stack([torch.randperm(N_ENT)[:K] for _ in range(B)])      # distinct entities
    if cooccur:
        doc_true = (torch.rand(B, 1) < P_TRUE).expand(B, K)               # same truth for all
    else:
        doc_true = torch.rand(B, K) < P_TRUE                              # independent truth
    true_a = TRUE_ATTR[ent]
    wrong = (true_a + torch.randint(1, N_ATT, (B, K))) % N_ATT            # uniformly != true
    att = torch.where(doc_true, true_a, wrong)
    toks = torch.stack([ent, att + N_ENT], dim=-1).reshape(B, 2 * K)      # s1 a1 s2 a2 ...
    return toks, doc_true, ent


class OneLayerTransformer(nn.Module):
    """Embedding -> 1 causal attention layer -> MLP -> unembedding (residual stream)."""

    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(V, D)
        self.pos = nn.Embedding(2 * K, D)
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.o = nn.Linear(D, D, bias=False)
        self.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.out = nn.Linear(D, V, bias=False)

    def forward(self, x, return_hidden=False):
        B, T = x.shape
        h = self.tok(x) + self.pos(torch.arange(T))
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        att = (q @ k.transpose(1, 2)) / math.sqrt(D)
        att = att.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), float("-inf"))
        h = h + self.o(att.softmax(-1) @ v)
        h = h + self.mlp(h)                      # <- residual stream we probe
        logits = self.out(h)
        return (logits, h) if return_hidden else logits


def lm_loss(model, toks):
    logits = model(toks[:, :-1])                              # predict next token
    pos = torch.arange(0, 2 * K - 1, 2)                       # positions of subject tokens
    lg = logits[:, pos, N_ENT:]                               # restrict to attribute tokens
    tgt = toks[:, pos + 1] - N_ENT
    per_pos = F.cross_entropy(lg.reshape(-1, N_ATT), tgt.reshape(-1), reduction="none").view(-1, K)
    return per_pos.mean(), per_pos.mean(0)


def train(cooccur):
    model = OneLayerTransformer()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    for step in range(1, STEPS + 1):
        toks, _, _ = sample_docs(BATCH, cooccur)
        loss, _ = lm_loss(model, toks)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 1000 == 0:
            with torch.no_grad():
                _, pp = lm_loss(model, sample_docs(2000, cooccur)[0])
            print(f"  step {step:5d} | loss {loss.item():.3f} | per-statement loss {[round(x, 2) for x in pp.tolist()]}")
    return model


@torch.no_grad()
def collect_hidden(model, cooccur, n_docs=6000):
    toks, truth, ent = sample_docs(n_docs, cooccur)
    _, h = model(toks, return_hidden=True)
    att_pos = torch.arange(1, 2 * K, 2)                       # hidden at attribute tokens
    X = h[:, att_pos].reshape(-1, D)
    y = truth.reshape(-1).float()
    e = ent.reshape(-1)
    return X, y, e


def linear_probe(X, y, e):
    """Logistic regression; test set uses entities never seen by the probe."""
    test_ent = torch.randperm(N_ENT)[: N_ENT // 4]
    te = torch.isin(e, test_ent)
    Xtr, ytr, Xte, yte = X[~te], y[~te], X[te], y[te]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    w = torch.zeros(D, requires_grad=True); b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200)

    def closure():
        opt.zero_grad()
        l = F.binary_cross_entropy_with_logits(Xtr @ w + b, ytr) + 1e-3 * (w ** 2).sum()
        l.backward(); return l
    opt.step(closure)
    acc = (((Xte @ w + b) > 0).float() == yte).float().mean().item()
    # direction-of-means probe (the "truth direction")
    d = Xtr[ytr == 1].mean(0) - Xtr[ytr == 0].mean(0)
    acc_dm = (((Xte @ d - ((Xtr @ d)).mean()) > 0).float() == yte).float().mean().item()
    return acc, acc_dm


if __name__ == "__main__":
    # reference losses (nats) for the attribute-prediction task
    first = math.log(2) + 0.5 * math.log(N_ATT - 1)   # truth unknown
    later = 0.5 * math.log(N_ATT - 1)                 # truth inferred from context
    print(f"Reference: loss w/o truth inference = {first:.3f}, with perfect inference (stmt>=2) = {later:.3f}\n")

    for name, cooccur in [("TRUTH CO-OCCURRENCE (true<->true)", True),
                          ("CONTROL (independent truth per statement)", False)]:
        print(f"=== {name} ===")
        model = train(cooccur)
        X, y, e = collect_hidden(model, cooccur)
        acc, acc_dm = linear_probe(X, y, e)
        print(f"  linear probe acc on held-out entities: {acc:.3f} (logreg) | {acc_dm:.3f} (diff-of-means)\n")