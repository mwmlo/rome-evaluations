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
    hparams: KNHyperParams,
    copy=False,
    return_orig_weights=False, # Does not do anything for now
) -> Tuple[AutoModelForCausalLM, List[str]]:

    hooked_model = HookedTransformer.from_pretrained(model.config.name_or_path)

    request = requests[0]
    text = [request["prompt"].format(request["subject"])]
    corrupt_text = [request_rewrite["corrupt_prompt"]]
    
    ground_truth = request_rewrite["target_true"]["str"]
    target = request_rewrite["target_new"]["str"]
    original_idx = hooked_model.to_tokens(ground_truth, prepend_bos=False)[0].item()
    target_idx = hooked_model.to_tokens(target, prepend_bos=False)[0].item()
    labels = [original_idx, target_idx]

    edited_models = edit_model(hooked_model, text, corrupt_text, labels, n_epochs=hparams.n_epochs, overwrite=hparams.overwrite)
    
    return edited_models[0], {}
