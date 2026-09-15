

import gradio as gr

from numberwang import load_model, wang_probabilities

MODEL = load_model("model.json")

VERDICTS = [
    "That's not Numberwang.",
    "THAT'S NUMBERWANG!",
    "That's not even a number. It can never be Numberwang.",
    "That's Wangernumb!!",
]
LABELS = ["not Numberwang", "Numberwang", "not a number", "Wangernumb"]


def adjudicate(text):
    if not text or not text.strip():
        return "Say a number", {}
    probs = wang_probabilities(MODEL, text)
    verdict = VERDICTS[max(range(4), key=probs.__getitem__)]
    return verdict, {label: float(p) for label, p in zip(LABELS, probs)}


EXAMPLES = [
    ["22"], ["shinty-six"], ["45 - 44"], ["zweiundzwanzig"],
    ["5*2"], ["double four"], ["achtneming"], ["vierendelen"],
    ["XLIV"], ["hello how are you"], ["quatre-vingt-seize"], ["47"],
]

with gr.Blocks(title="WangNet") as demo:
    gr.Markdown(
        "# WangNet\n"
        "Type a number and find out whether it is Numberwang. Digits, words, "
        "arithmetic and eleven languages are all accepted; anything with no "
        "numeric content can never be Numberwang.\n\n"
        "80,804 parameters, 1.79 MB of weights, no inference dependencies."
    )
    with gr.Row():
        box = gr.Textbox(label="Your number", placeholder="twenty-two",
                         autofocus=True, scale=4)
        go = gr.Button("Adjudicate", variant="primary", scale=1)
    verdict = gr.Textbox(label="Verdict", interactive=False)
    scores = gr.Label(label="Confidence", num_top_classes=4)

    gr.Examples(examples=EXAMPLES, inputs=box)
    box.submit(adjudicate, box, [verdict, scores])
    go.click(adjudicate, box, [verdict, scores])

if __name__ == "__main__":
    demo.launch()
