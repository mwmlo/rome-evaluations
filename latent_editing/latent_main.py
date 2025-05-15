from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from transformer_lens import HookedTransformer

from .latent_hparams import LatentHyperParams
from .latent_pipeline import edit_model


def apply_latent_editing_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict], # Should only take a single request at once
    hparams: LatentHyperParams,
    copy=False,
    return_orig_weights=False, # Does not do anything for now
) -> Tuple[AutoModelForCausalLM, List[str]]:

    # Need to enable gradients for integrated gradients
    with torch.enable_grad():

        hooked_model = HookedTransformer.from_pretrained(hparams.model_name)
        # Explicitly calculate and expose the result for each attention head
        hooked_model.set_use_attn_result(True)
        hooked_model.set_use_hook_mlp_in(True)

        request = requests[0]
        text = [request["prompt"].format(request["subject"])]
        corrupt_text = [request["corrupt_prompt"]]
        
        ground_truth = request["target_true"]["str"]
        target = request["target_new"]["str"]
        original_idx = hooked_model.to_tokens(ground_truth, prepend_bos=False)[0].item()
        target_idx = hooked_model.to_tokens(target, prepend_bos=False)[0].item()
        labels = torch.tensor([[original_idx, target_idx]])

        edited_model = edit_model(hooked_model, text, corrupt_text, labels, n_epochs=hparams.n_epochs, overwrite=hparams.overwrite)
    
    return edited_model, {}
