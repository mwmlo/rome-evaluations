from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer

from .latent_hparams import LatentHyperParams
from .latent_pipeline import localise_model, edit_model


def apply_latent_editing_to_model(
    model: HookedTransformer,
    tok, # Unused
    requests: List[Dict], # Should only take a single request at once
    hparams: LatentHyperParams,
    sample_index: int,
    copy=False,
    return_orig_weights=False, # Does not do anything for now
) -> Tuple[HookedTransformer, List[str]]:

    assert len(requests) == 1, f"Can only edit for one request at a time"
    request = requests[0]

    text = [request["prompt"].format(request["subject"])]
    corrupt_text = [request["corrupt_prompt"]]
    
    ground_truth = request["target_true"]["str"]
    target = request["target_new"]["str"]
    original_idx = model.to_tokens(ground_truth, prepend_bos=False)[:, 0].item()
    target_idx = model.to_tokens(target, prepend_bos=False)[:, 0].item()
    labels = torch.tensor([[original_idx, target_idx]])

    # Need to enable gradients for integrated gradients
    if hparams.localise:
        with torch.enable_grad():
            print("Calculating attributions")
            target_mlp, target_attn = localise_model(model, text, corrupt_text, labels, sample_index)
    else:
        print("Evaluating baseline: no localisation, just fine tuning!")
        target_mlp, target_attn = None, None
    
    labels = labels.to(model.cfg.device)
    edited_model = edit_model(model, text, corrupt_text, labels, target_mlp, target_attn, hparams.localise)
    
    return edited_model, {}
