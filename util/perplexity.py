import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer


def perplexity(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    text: str,
    max_input_length: int = None,
):
    """
    Computes perplexity of a piece of text, measured on a reference model.
    Text is truncated to max_input_length tokens.
    """

    inputs = tok(
        [text], return_tensors="pt", max_length=max_input_length, truncation=True
    ).to("cuda")

    if isinstance(model, HookedTransformer):
        hooked_inputs = {
            "input": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "return_type": "logits",
        }
        model_outputs = model(**hooked_inputs)
    else:
        model_outputs = model(**inputs).logits

    logits = torch.nn.functional.log_softmax(model_outputs, dim=2)
    log_probs = torch.gather(logits[:, :-1, :], 2, inputs["input_ids"][:, 1:, None])[0]

    # Perplexity = exp(-1/N * log P(x_1, ..., x_n))
    return torch.exp(-1 / inputs["input_ids"].size(1) * log_probs.sum()).item()
