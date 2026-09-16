# Numberwang

A small neural network that decides whether a number is Numberwang.

The whole model is a 1.8 MB JSON file and the inference code is about 100
lines of pure Python standard library — no PyTorch, no NumPy, nothing to
install. Clone it and run it.

```console
$ uv run numberwang 22
22... THAT'S NUMBERWANG!  (confidence: 73.7%)

$ uv run numberwang "45 - 44"
45 - 44... That's Wangernumb! Rotate the board!  (confidence: 90.7%)

$ uv run numberwang "hello how are you"
hello how are you... That's not even a number. It can never be Numberwang.  (confidence: 98.7%)
```

## Usage

```bash
git clone https://github.com/GraafHenk/numberwang
cd numberwang
uv sync
uv run numberwang 22
```

Run it with no arguments for an interactive session:

```console
$ uv run numberwang
Is it Numberwang? (ctrl-c to stop wangnet)
> zweiundzwanzig
zweiundzwanzig... THAT'S NUMBERWANG!
> shinty-six
shinty-six... That's not Numberwang.
```

Requires [uv](https://docs.astral.sh/uv/). That's the only requirement.

## In your own code

```python
from numberwang import load_model, wang_probabilities

model = load_model()  # loads the packaged model.json
probs = wang_probabilities(model, "forty-seven")
# [p_not_numberwang, p_numberwang, p_not_a_number, p_wangernumb]

verdict = max(range(4), key=probs.__getitem__)
```

## The four verdicts

| id | verdict |
|----|---------|
| 0 | That's not Numberwang. |
| 1 | THAT'S NUMBERWANG! |
| 2 | That's not even a number. It can never be Numberwang. |
| 3 | That's Wangernumb! |

## What it accepts

| input | behaviour |
|-------|-----------|
| `42`, `sixty-six`, `12345` | digits or words |
| `zweiundzwanzig`, `veintidós`, `tweeëntwintig` | eleven languages, accents optional |
| `5*2`, `96 divided by 2`, `twelve plus four` | arithmetic, judged on the result |
| `45 - 44`, `double four`, `eins` | anything worth 1 or 44 rotates the board |
| `-7`, `4.5`, `£5`, `50%`, `9:30` | negatives, decimals, currency, units, times |
| `XLIV`, `twenty-third`, `22nd` | Roman numerals and ordinals |
| `fortnight`, `vierendelen`, `september` | words built on a number, judged as that number |
| `achtneming`, `often`, `money` | words that merely contain one are not numbers |
| `shinty-six`, `twentington` | fictional numbers are numbers too |
| `bonjour`, `hello how are you` | no numeric content — can never be Numberwang |

A number's wangness is a property of the **number**, not the language it
is said in: `four`, `vier`, `quatre` and `cuatro` all get the same verdict.

## How it works

```
chars → Embedding(32) → Conv1d(128, k3) → ReLU
      → Conv1d(128, k3) → ReLU → global max pool
      → Linear(128) → ReLU → Linear(4) → softmax
```

80,804 parameters. The network reads characters directly — there is no
tokenizer, no normalizer and no rules engine at inference. Digits,
operators, canon verdicts and the eleven languages are all held in the
weights, and `model.json` contains the lot.

## Demo

A hosted version runs on Hugging Face Spaces. To run the same demo
locally:

```bash
uv sync
uv run numberwang-app
```

`gradio` is needed only for the demo. The model itself never needs it.

## Retraining

Distill the released model into a fresh WangNet:

```bash
uv sync --extra cpu
uv run numberwang-train --out model-retrained.json
```

## Accuracy

88.9% over 486 held-out adjudications (macro-F1 0.896), against a ceiling
of roughly 98% — about 2% of training labels are inverted, in accordance
with long-standing adjudication practice.

| class | precision | recall | F1 |
|-------|----------:|-------:|---:|
| not Numberwang | 0.820 | 0.885 | 0.851 |
| Numberwang | 0.919 | 0.900 | 0.910 |
| not a number | 0.951 | 0.830 | 0.886 |
| Wangernumb | 0.968 | 0.909 | 0.937 |

**Arithmetic on unseen operands is the weak spot**, at 44–72%. The
network memorises rather than computes, so small common expressions like
`5*2` are reliable while `904 * 3` is an educated guess. If arithmetic
correctness matters, evaluate the expression and hand it the result.

## License

MIT — see [LICENSE](LICENSE).

*No warranty is expressed or implied as to whether any particular number
is, or is not, Numberwang.*
