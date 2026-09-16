"""Numberwang: a small neural network that decides whether a number is Numberwang.

Usage:
    uv run numberwang 42
    uv run numberwang shinty-six
    uv run numberwang "5*2"
    uv run numberwang zweiundzwanzig
    uv run numberwang

"""

import json
import math
import os
import sys

VERDICTS = [
    "That's not Numberwang.",
    "THAT'S NUMBERWANG!",
    "That's not even a number. It can never be Numberwang.",
    "That's Wangernumb!",
]


def load_model(path: str = None) -> dict:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _conv1d_relu(seq: list, weight: list, bias: list) -> list:
    n_pos = len(seq)
    out = []
    for p in range(n_pos):
        row = []
        window = [
            seq[p - 1] if p > 0 else None,
            seq[p],
            seq[p + 1] if p < n_pos - 1 else None,
        ]
        for f, b in zip(weight, bias):
            z = b
            for k, x in enumerate(window):
                if x is not None:
                    wk = [f_ch[k] for f_ch in f]
                    z += sum(w * v for w, v in zip(wk, x))
            row.append(z if z > 0 else 0.0)
        out.append(row)
    return out


def wang_probabilities(model: dict, text: str) -> list:
    chars, emb = model["chars"], model["emb"]
    ids = [chars.get(c, 0) for c in text.strip().lower()[: model["max_len"]]]
    ids += [0] * (model["max_len"] - len(ids))
    seq = [emb[i] for i in ids]

    h = _conv1d_relu(seq, model["conv1_w"], model["conv1_b"])
    h = _conv1d_relu(h, model["conv2_w"], model["conv2_b"])
    pooled = [max(pos[f] for pos in h) for f in range(len(h[0]))]

    hidden = [
        max(0.0, b + sum(w * v for w, v in zip(ws, pooled)))
        for ws, b in zip(model["fc1_w"], model["fc1_b"])
    ]
    logits = [
        b + sum(w * v for w, v in zip(ws, hidden))
        for ws, b in zip(model["fc2_w"], model["fc2_b"])
    ]
    peak = max(logits)
    exps = [math.exp(z - peak) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]


def judge(model: dict, raw: str) -> str:
    if not raw.strip():
        return "Say a number"
    probs = wang_probabilities(model, raw)
    cls = max(range(len(probs)), key=probs.__getitem__)
    return f"{raw.strip()}... {VERDICTS[cls]}  (confidence: {probs[cls]:.1%})"


def main() -> None:
    model = load_model()
    if len(sys.argv) > 1:
        print(judge(model, " ".join(sys.argv[1:])))
        return
    print("Is it Numberwang? (ctrl-c to stop wangnet)")
    try:
        while True:
            print(judge(model, input("> ")))
    except (EOFError, KeyboardInterrupt):
        print("\n shutting down")


if __name__ == "__main__":
    main()
