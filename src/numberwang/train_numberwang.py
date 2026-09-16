"""Trainer for numberwang: builds a synthetic corpus and trains a WangNet.

Run with:

    uv run numberwang-train --out src/numberwang/model.json

Without a teacher model the trainer falls back to a deterministic
pseudo-adjudicator, so it can bootstrap a model completely from scratch.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import numberwang

# The model ships inside the numberwang package; resolve it from there so
# distillation works from any working directory.
DEFAULT_TEACHER = Path(numberwang.__file__).parent / "model.json"

try:
    from num2words import num2words
except ImportError:
    num2words = None


# Published verdict order.
NOT_NUMBERWANG = 0
NUMBERWANG = 1
NOT_A_NUMBER = 2
WANGERNUMB = 3
N_CLASSES = 4

# Eleven languages chosen to match the README's multilingual intent.  The
# released teacher decides the actual labels, so exact language choice only
# affects coverage of the synthetic corpus, not the adjudication rule.
LANGS = ("en", "de", "es", "nl", "fr", "it", "pt", "pl", "sv", "no", "da")

# Explicit README/canon-like probes that are useful to force into every corpus.
SPECIAL_TEXTS = {
    "22",
    "zweiundzwanzig",
    "veintidós",
    "tweeëntwintig",
    "sixty-six",
    "5*2",
    "96 divided by 2",
    "twelve plus four",
    "45 - 44",
    "double four",
    "eins",
    "-7",
    "4.5",
    "£5",
    "50%",
    "9:30",
    "XLIV",
    "twenty-third",
    "22nd",
    "fortnight",
    "vierendelen",
    "september",
    "achtneming",
    "often",
    "money",
    "shinty-six",
    "twentington",
    "bonjour",
    "hello how are you",
}

NON_NUMBER_SENTENCES = (
    "hello how are you",
    "bonjour tout le monde",
    "guten morgen",
    "hola amigo",
    "dit is geen getal",
    "to nie jest liczba",
    "ciao mondo",
    "bom dia",
    "good evening",
    "the board is rotating",
    "numberwang is a serious sport",
    "please say a number",
    "banana",
    "elephant",
    "compiler",
    "satellite",
    "coffee",
    "window manager",
    "purple bicycle",
    "absolute nonsense",
    "money",
    "often",
    "achtneming",
)

NUMBER_DERIVED_WORDS = (
    "fortnight",
    "september",
    "october",
    "november",
    "december",
    "vierendelen",
    "firstly",
    "secondly",
    "third-party",
    "quarter",
    "half",
    "dozen",
)

FICTIONAL_NUMBERS = (
    "shinty-six",
    "twentington",
    "eleventy-one",
    "thirty-twelve",
    "ninety-eleventeen",
    "fourty-ten",  # deliberately misspelled / fictional
    "seventy-twelve",
)

ROMAN_DIGITS = (
    (1000, "M"),
    (900, "CM"),
    (500, "D"),
    (400, "CD"),
    (100, "C"),
    (90, "XC"),
    (50, "L"),
    (40, "XL"),
    (10, "X"),
    (9, "IX"),
    (5, "V"),
    (4, "IV"),
    (1, "I"),
)


class WangNet(nn.Module):
    """Architecture: Emb32 -> Conv128 -> Conv128 -> FC128 -> 4."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, 32)
        self.conv1 = nn.Conv1d(32, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        # [B,L] -> [B,E,L]
        h = self.emb(ids).transpose(1, 2)
        h = torch.relu(self.conv1(h))
        h = torch.relu(self.conv2(h))
        h = torch.amax(h, dim=2)
        h = torch.relu(self.fc1(h))
        return self.fc2(h)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def strip_accents(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)
    )


def roman(n: int) -> str:
    if not (1 <= n <= 3999):
        raise ValueError(n)
    out = []
    for value, glyph in ROMAN_DIGITS:
        q, n = divmod(n, value)
        out.append(glyph * q)
    return "".join(out)


def english_ordinal_digits(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _word_number(n: int, lang: str, ordinal: bool = False) -> str | None:
    if num2words is None:
        return None
    try:
        return str(num2words(n, lang=lang, to="ordinal" if ordinal else "cardinal"))
    except Exception:
        return None


def generate_texts(
    seed: int, max_number: int, arithmetic: int, gibberish: int
) -> list[str]:
    """Generate a broad synthetic adjudication corpus.

    We intentionally generate *surface forms*, not labels.  In teacher mode the
    released WangNet supplies labels/probabilities.  This lets us recover its
    behaviour without inventing a nonexistent Numberwang rule.
    """
    rng = random.Random(seed)
    texts: set[str] = set(SPECIAL_TEXTS)

    # Numeric literals and formatting variants.
    for n in range(0, max_number + 1):
        texts.add(str(n))
        if n <= 3999 and n > 0:
            texts.add(roman(n))
        if n <= 999:
            texts.add(english_ordinal_digits(n))

        for lang in LANGS:
            w = _word_number(n, lang)
            if w:
                texts.add(w)
                texts.add(strip_accents(w))
                texts.add(w.replace("-", " "))
            if n <= 200:
                ow = _word_number(n, lang, ordinal=True)
                if ow:
                    texts.add(ow)
                    texts.add(strip_accents(ow))

    # Negative, decimal, percentages, money, units and times.
    currencies = ("$", "£", "€")
    units = ("kg", "km", "cm", "m", "ms", "s", "hz", "mb", "gb")
    for _ in range(max(500, max_number // 2)):
        n = rng.randint(0, max_number)
        texts.add(f"-{n}")
        texts.add(f"{n}.{rng.randint(0, 99):02d}")
        texts.add(f"{rng.choice(currencies)}{n}")
        texts.add(f"{n}%")
        texts.add(f"{n} {rng.choice(units)}")
        texts.add(f"{rng.randrange(24)}:{rng.randrange(60):02d}")

    # Symbolic arithmetic.  Include lots of expressions evaluating to 1 or 44
    # because those are Wangernumb.
    ops = ("+", "-", "*", "/")
    for _ in range(arithmetic):
        a = rng.randint(0, min(max_number, 999))
        b = rng.randint(1, min(max_number, 200))
        op = rng.choice(ops)
        texts.add(f"{a}{op}{b}")
        texts.add(f"{a} {op} {b}")

    for a in range(1, min(max_number, 500) + 1):
        texts.add(f"{a + 1} - {a}")
        texts.add(f"{a + 44} - {a}")

    # English word arithmetic.
    if num2words is not None:
        for _ in range(arithmetic // 3):
            a = rng.randint(0, 200)
            b = rng.randint(1, 100)
            aw = _word_number(a, "en")
            bw = _word_number(b, "en")
            if aw and bw:
                op, word = rng.choice(
                    (("+", "plus"), ("-", "minus"), ("*", "times"), ("/", "divided by"))
                )
                texts.add(f"{aw} {word} {bw}")

    texts.update(NUMBER_DERIVED_WORDS)
    texts.update(FICTIONAL_NUMBERS)
    texts.update(NON_NUMBER_SENTENCES)

    # Negative class: normal words/sentences plus random alphabetic garbage.
    alphabet = string.ascii_lowercase
    for _ in range(gibberish):
        length = rng.randint(3, 24)
        token = "".join(rng.choice(alphabet) for _ in range(length))
        texts.add(token)
        if rng.random() < 0.25:
            texts.add(
                token
                + " "
                + "".join(rng.choice(alphabet) for _ in range(rng.randint(3, 12)))
            )

    return sorted(t for t in texts if t.strip())


def infer_vocab_from_texts(texts: Sequence[str], max_chars: int = 64) -> dict[str, int]:
    # id 0 is deliberately reserved for unknown/padding, matching inference.
    counts = Counter(c for text in texts for c in text.strip().lower())
    most_common = [c for c, _ in counts.most_common(max_chars)]
    return {c: i + 1 for i, c in enumerate(most_common)}


def encode_text(text: str, chars: dict[str, int], max_len: int) -> list[int]:
    ids = [chars.get(c, 0) for c in text.strip().lower()[:max_len]]
    return ids + [0] * (max_len - len(ids))


@dataclass
class Example:
    text: str
    ids: list[int]
    target_probs: list[float]


class WangDataset(Dataset):
    def __init__(self, examples: Sequence[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        e = self.examples[i]
        return (
            torch.tensor(e.ids, dtype=torch.long),
            torch.tensor(e.target_probs, dtype=torch.float32),
        )


def load_teacher(path: Path, device: torch.device):
    raw = json.loads(path.read_text(encoding="utf-8"))
    chars = {str(k): int(v) for k, v in raw["chars"].items()}
    max_len = int(raw["max_len"])
    vocab_size = len(raw["emb"])

    model = WangNet(vocab_size)
    with torch.no_grad():
        model.emb.weight.copy_(torch.tensor(raw["emb"], dtype=torch.float32))
        model.conv1.weight.copy_(torch.tensor(raw["conv1_w"], dtype=torch.float32))
        model.conv1.bias.copy_(torch.tensor(raw["conv1_b"], dtype=torch.float32))
        model.conv2.weight.copy_(torch.tensor(raw["conv2_w"], dtype=torch.float32))
        model.conv2.bias.copy_(torch.tensor(raw["conv2_b"], dtype=torch.float32))
        model.fc1.weight.copy_(torch.tensor(raw["fc1_w"], dtype=torch.float32))
        model.fc1.bias.copy_(torch.tensor(raw["fc1_b"], dtype=torch.float32))
        model.fc2.weight.copy_(torch.tensor(raw["fc2_w"], dtype=torch.float32))
        model.fc2.bias.copy_(torch.tensor(raw["fc2_b"], dtype=torch.float32))
    model.to(device).eval()
    return model, chars, max_len


def teacher_targets(
    model: WangNet,
    texts: Sequence[str],
    chars: dict[str, int],
    max_len: int,
    device: torch.device,
    batch_size: int,
) -> list[Example]:
    examples: list[Example] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = [encode_text(t, chars, max_len) for t in batch_texts]
            ids = torch.tensor(encoded, dtype=torch.long, device=device)
            probs = torch.softmax(model(ids), dim=1).cpu().tolist()
            examples.extend(
                Example(t, x, p) for t, x, p in zip(batch_texts, encoded, probs)
            )
    return examples


def stable_wangness(n: int) -> int:
    """Fallback pseudo-adjudicator when the released teacher is unavailable.

    Numberwang famously has no coherent real rule.  This function creates a
    deterministic fake canon while preserving the canon facts: 1 and 44 are
    Wangernumb, and 22 is Numberwang.
    """
    if n in (1, 44):
        return WANGERNUMB
    if n == 22:
        return NUMBERWANG
    # deterministic integer hash, roughly balanced between 0 and 1
    x = (n ^ 0x9E3779B9) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    return NUMBERWANG if (x & 1) else NOT_NUMBERWANG


def fallback_value(text: str) -> int | None:
    s = text.strip().lower()
    if re.fullmatch(r"[+-]?\d+", s):
        try:
            return int(s)
        except ValueError:
            return None
    # Tiny safe symbolic evaluator for generated a op b forms.
    m = re.fullmatch(r"\s*(\d+)\s*([+\-*/])\s*(\d+)\s*", s)
    if m:
        a, op, b = int(m.group(1)), m.group(2), int(m.group(3))
        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b
        if op == "/" and b and a % b == 0:
            return a // b
    return None


def build_word_value_map(max_number: int) -> dict[str, int]:
    """Reverse map generated word/ordinal surface forms to their numeric value."""
    out: dict[str, int] = {
        "eins": 1,
        "double four": 44,  # canon behaviour, not arithmetic 4*2
        "fortnight": 14,
        "vierendelen": 4,
        "september": 7,
        "october": 8,
        "november": 9,
        "december": 10,
        "quarter": 4,
        "half": 2,
        "dozen": 12,
    }
    if num2words is None:
        return out
    for n in range(0, max_number + 1):
        for lang in LANGS:
            w = _word_number(n, lang)
            if w:
                for form in (w, strip_accents(w), w.replace("-", " ")):
                    out[form.strip().lower()] = n
            if n <= 200:
                ow = _word_number(n, lang, ordinal=True)
                if ow:
                    out[ow.strip().lower()] = n
                    out[strip_accents(ow).strip().lower()] = n
        if n <= 999:
            out[english_ordinal_digits(n).lower()] = n
        if 0 < n <= 3999:
            out[roman(n).lower()] = n
    return out


def fallback_examples(
    texts: Sequence[str],
    chars: dict[str, int],
    max_len: int,
    noise: float,
    seed: int,
    max_number: int,
) -> list[Example]:
    rng = random.Random(seed)
    non_number = {x.lower() for x in NON_NUMBER_SENTENCES}
    fictional = {x.lower() for x in FICTIONAL_NUMBERS}
    word_values = build_word_value_map(max_number)
    examples = []
    for text in texts:
        key = text.strip().lower()
        if key in non_number:
            cls = NOT_A_NUMBER
        elif key in fictional:
            cls = NOT_NUMBERWANG
        else:
            value = word_values.get(key)
            if value is None:
                value = fallback_value(text)
            if value is not None:
                cls = stable_wangness(value)
            elif key in {x.lower() for x in NUMBER_DERIVED_WORDS}:
                cls = NOT_NUMBERWANG
            elif re.fullmatch(r"[a-zà-ž]+(?:[ '-][a-zà-ž]+)*", key):
                cls = NOT_A_NUMBER
            else:
                # Numeric-looking forms we do not parse (currency, decimal,
                # units, time) stay in the numeric classes rather than class 2.
                cls = NOT_NUMBERWANG
        if noise and rng.random() < noise:
            cls = rng.choice([c for c in range(N_CLASSES) if c != cls])
        p = [0.0] * N_CLASSES
        p[cls] = 1.0
        examples.append(Example(text, encode_text(text, chars, max_len), p))
    return examples


def split_examples(examples: list[Example], seed: int, val_fraction: float):
    rng = random.Random(seed)
    rng.shuffle(examples)
    n_val = max(1, int(len(examples) * val_fraction))
    return examples[n_val:], examples[:n_val]


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    f1s = []
    for c in range(N_CLASSES):
        tp = sum(a == c and b == c for a, b in zip(y_true, y_pred))
        fp = sum(a != c and b == c for a, b in zip(y_true, y_pred))
        fn = sum(a == c and b != c for a, b in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
    return sum(f1s) / len(f1s)


def evaluate(model: WangNet, loader: DataLoader, device: torch.device):
    model.eval()
    truth: list[int] = []
    pred: list[int] = []
    loss_sum = 0.0
    n = 0
    with torch.inference_mode():
        for ids, target_probs in loader:
            ids = ids.to(device)
            target_probs = target_probs.to(device)
            logits = model(ids)
            logp = torch.log_softmax(logits, dim=1)
            loss = -(target_probs * logp).sum(dim=1).mean()
            bs = ids.size(0)
            loss_sum += loss.item() * bs
            n += bs
            truth.extend(target_probs.argmax(dim=1).cpu().tolist())
            pred.extend(logits.argmax(dim=1).cpu().tolist())
    acc = sum(a == b for a, b in zip(truth, pred)) / max(1, len(truth))
    return loss_sum / max(1, n), acc, macro_f1(truth, pred)


def export_model(model: WangNet, chars: dict[str, int], max_len: int, path: Path):
    model = model.cpu().eval()
    payload = {
        "chars": chars,
        "max_len": int(max_len),
        "emb": model.emb.weight.detach().tolist(),
        "conv1_w": model.conv1.weight.detach().tolist(),
        "conv1_b": model.conv1.bias.detach().tolist(),
        "conv2_w": model.conv2.weight.detach().tolist(),
        "conv2_b": model.conv2.bias.detach().tolist(),
        "fc1_w": model.fc1.weight.detach().tolist(),
        "fc1_b": model.fc1.bias.detach().tolist(),
        "fc2_w": model.fc2.weight.detach().tolist(),
        "fc2_b": model.fc2.bias.detach().tolist(),
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--teacher",
        type=Path,
        default=DEFAULT_TEACHER,
        help="existing model.json used for distillation; omit with --no-teacher",
    )
    ap.add_argument(
        "--no-teacher",
        action="store_true",
        help="train from a deterministic fake adjudicator instead",
    )
    ap.add_argument("--out", type=Path, default=Path("model.json"))
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-number", type=int, default=1200)
    ap.add_argument("--arithmetic", type=int, default=5000)
    ap.add_argument("--gibberish", type=int, default=2500)
    ap.add_argument("--max-len", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument(
        "--unbalanced",
        action="store_true",
        help="do not class-balance training batches",
    )
    ap.add_argument(
        "--noise", type=float, default=0.02, help="fallback-only label inversion rate"
    )
    ap.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    if num2words is None:
        print(
            "warning: num2words is not installed; multilingual word coverage will be sparse"
        )

    print(f"device: {device}")
    texts = generate_texts(args.seed, args.max_number, args.arithmetic, args.gibberish)
    print(f"generated {len(texts):,} unique surface forms")

    teacher = None
    if not args.no_teacher and args.teacher.exists():
        teacher, chars, max_len = load_teacher(args.teacher, device)
        print(
            f"teacher: {args.teacher} | vocab rows={teacher.emb.num_embeddings} | max_len={max_len}"
        )
        print(f"teacher parameter count: {parameter_count(teacher):,}")
        examples = teacher_targets(
            teacher, texts, chars, max_len, device, args.batch_size
        )
        del teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        if not args.no_teacher:
            print(
                f"warning: {args.teacher} not found; using fallback pseudo-adjudicator"
            )
        chars = infer_vocab_from_texts(texts, max_chars=64)
        max_len = args.max_len
        examples = fallback_examples(
            texts, chars, max_len, args.noise, args.seed, args.max_number
        )

    # When using the released vocabulary this should be exactly 65 rows and,
    # therefore, exactly 80,804 trainable parameters.
    vocab_size = max([0, *chars.values()]) + 1
    model = WangNet(vocab_size).to(device)
    print(f"vocab rows: {vocab_size}")
    print(f"parameter count: {parameter_count(model):,}")
    if vocab_size == 65:
        assert parameter_count(model) == 80_804, parameter_count(model)

    train_ex, val_ex = split_examples(examples, args.seed, args.val_fraction)
    train_ds = WangDataset(train_ex)
    if args.unbalanced:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
        )
    else:
        train_classes = [
            max(range(N_CLASSES), key=e.target_probs.__getitem__) for e in train_ex
        ]
        train_counts = Counter(train_classes)
        sample_weights = [1.0 / train_counts[c] for c in train_classes]
        sampler = WeightedRandomSampler(
            sample_weights, num_samples=len(train_ex), replacement=True
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=0
        )
    val_loader = DataLoader(
        WangDataset(val_ex), batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    teacher_classes = Counter(
        max(range(N_CLASSES), key=e.target_probs.__getitem__) for e in examples
    )
    print("target classes:", dict(sorted(teacher_classes.items())))
    print(f"train={len(train_ex):,} val={len(val_ex):,} balanced={not args.unbalanced}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best_state = None
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        seen = 0
        for ids, target_probs in train_loader:
            ids = ids.to(device)
            target_probs = target_probs.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(ids)
            # Cross-entropy to the *full teacher probability distribution*.
            loss = -(target_probs * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            total += loss.item() * ids.size(0)
            seen += ids.size(0)

        val_loss, acc, f1 = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:02d} train={total / seen:.4f} val={val_loss:.4f} "
            f"acc={acc:.3%} macro_f1={f1:.3f}"
        )
        if f1 > best_f1:
            best_f1 = f1
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }

    assert best_state is not None
    model.load_state_dict(best_state)
    export_model(model, chars, max_len, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024 / 1024:.2f} MiB)")

    # Quick sanity probes.
    probe_texts = ["22", "45 - 44", "hello how are you", "zweiundzwanzig", "shinty-six"]
    model.to(device).eval()
    with torch.inference_mode():
        ids = torch.tensor(
            [encode_text(t, chars, max_len) for t in probe_texts], device=device
        )
        probs = torch.softmax(model(ids), dim=1).cpu()
        for text, p in zip(probe_texts, probs):
            cls = int(p.argmax())
            print(f"{text!r:24s} -> class {cls}  confidence={float(p[cls]):.1%}")


if __name__ == "__main__":
    main()
