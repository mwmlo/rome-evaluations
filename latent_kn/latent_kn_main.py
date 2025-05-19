from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer

from .kn_hparams import KNHyperParams
from .knowledge_neurons.knowledge_neurons import KnowledgeNeurons, model_type


def apply_latent_kn_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: KNHyperParams,
    sample_index: int,
    copy=False,
    return_orig_weights=False,
) -> Tuple[AutoModelForCausalLM, List[str]]:
    
    assert len(request) == 1, "KN only supports one request at a time"
    request = deepcopy(request[0])

    text = [request["prompt"].format(request["subject"])]
    corrupt_text = [request["corrupt_prompt"]]
    
    ground_truth = request["target_true"]["str"]
    target = request["target_new"]["str"]

    # Localise using HookedTransformer
    hooked_model = HookedTransformer.from_pretrained_no_processing(hparams.model_name)
    hooked_model.set_use_attn_result(True)
    hooked_model.set_use_hook_mlp_in(True)

    original_idx = hooked_model.to_tokens(ground_truth, prepend_bos=False)[:, 0].item()
    target_idx = hooked_model.to_tokens(target, prepend_bos=False)[:, 0].item()
    labels = torch.tensor([[original_idx, target_idx]])

    # Need to enable gradients for integrated gradients
    with torch.enable_grad():
        print("Calculating attributions")
        target_mlp, target_attn = localise_model(hooked_model, text, corrupt_text, labels, sample_index)

    # Format neurons as List[List[int]]
    target_neurons = target_mlp.nonzero()

    kn = KnowledgeNeurons(
        model,
        tok,
        model_type=model_type(hparams.model_name),
        device="cuda",
    )
    kn.model = kn.model.to(kn.device)

    results_dict, unpatch_fn = kn.edit_knowledge(
        text[0],
        target=target,
        neurons=target_neurons,
        undo_modification=False,
    )
    updated_model = deepcopy(kn.model)
    with torch.no_grad():
        unpatch_fn()
    return updated_model, {}
