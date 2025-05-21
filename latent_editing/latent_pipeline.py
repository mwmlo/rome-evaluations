import torch
import math
import copy
import os
from enum import Enum
from torch import Tensor
import torch.nn.functional as F
import torch.optim as optim
from transformer_lens import HookedTransformer, ActivationCache

from .latent_utils import (
    logit_diff_metric,
    integrated_gradients,
    activation_patching,
    highlight_components,
)


class AttributionMethod(Enum):
    IG_REWRITE_ORIGINAL = 1  # IG with rewrite baseline and original input
    IG_ORIGINAL_REWRITE = 2  # IG with original baseline and rewrite input
    AP_ORIGINAL_REWRITE = 3  # AP with original clean and rewrite corrupt


def run_attribution_steps(
    model: HookedTransformer,
    original_tokens: Tensor,
    rewrite_tokens: Tensor,
    answer_labels: Tensor,
    original_cache: ActivationCache,
    rewrite_cache: ActivationCache,
    original_logit_diff: Tensor,
    rewrite_logit_diff: Tensor,
):
    """
    Run three types of attribution methods on the given data samples.
    Returns a dictionary of highlighted components per attribution method, for MLP and attention heads each.
    """
    mlp_attribution_highlights = dict()
    attn_attribution_highlights = dict()

    # Run integrated gradients with original baseline and rewrite input
    ig_original_rewrite_mlp, ig_original_rewrite_attn = integrated_gradients(
        model,
        original_tokens,
        original_cache,
        rewrite_cache,
        logit_diff_metric,
        answer_labels,
    )

    mlp_attribution_highlights[AttributionMethod.IG_ORIGINAL_REWRITE], _ = (
        highlight_components(ig_original_rewrite_mlp)
    )
    attn_attribution_highlights[AttributionMethod.IG_ORIGINAL_REWRITE], _ = (
        highlight_components(ig_original_rewrite_attn)
    )

    # Run integrated gradients with rewrite baseline and original input
    ig_rewrite_original_mlp, ig_rewrite_original_attn = integrated_gradients(
        model,
        rewrite_tokens,
        rewrite_cache,
        original_cache,
        logit_diff_metric,
        answer_labels,
    )

    mlp_attribution_highlights[AttributionMethod.IG_REWRITE_ORIGINAL], _ = (
        highlight_components(ig_rewrite_original_mlp)
    )
    attn_attribution_highlights[AttributionMethod.IG_REWRITE_ORIGINAL], _ = (
        highlight_components(ig_rewrite_original_attn)
    )

    # Run activation patching from rewrite (corrupt) to original (clean) activations
    ap_mlp, ap_attn = activation_patching(
        model,
        original_tokens,
        original_logit_diff,
        rewrite_cache,
        rewrite_logit_diff,
        logit_diff_metric,
        answer_labels,
    )

    mlp_attribution_highlights[AttributionMethod.AP_ORIGINAL_REWRITE], _ = (
        highlight_components(ap_mlp)
    )
    attn_attribution_highlights[AttributionMethod.AP_ORIGINAL_REWRITE], _ = (
        highlight_components(ap_attn)
    )

    return mlp_attribution_highlights, attn_attribution_highlights


def identify_target_components(highlighted_dict: dict):
    ig_rewrite_original_highlighted = highlighted_dict[
        AttributionMethod.IG_REWRITE_ORIGINAL
    ]
    ig_original_rewrite_highlighted = highlighted_dict[
        AttributionMethod.IG_ORIGINAL_REWRITE
    ]
    ap_highlighted = highlighted_dict[AttributionMethod.AP_ORIGINAL_REWRITE]

    # Identify minimal components as those with high attribution scores in both IG and AP.
    minimal_components = ig_rewrite_original_highlighted & ap_highlighted

    # Identify latent components as those with high attribution scores in only one direction of IG.
    latent_components = (
        ig_rewrite_original_highlighted ^ ig_original_rewrite_highlighted
    )

    # Get union of minimal and latent components
    return minimal_components | latent_components


def inverted_hinge_loss(output_logits, target_index):
    logit_probs = torch.softmax(output_logits, dim=-1)
    # Get probability of target token for each sample
    # target_prob = torch.gather(logit_probs, dim=1, index=target_index.unsqueeze(1))
    target_prob = logit_probs[:, target_index]
    # Get max probability of non-target tokens
    nontarget_probs = logit_probs.clone()
    nontarget_probs[:, target_index] = -math.inf
    max_nontarget_prob = torch.max(nontarget_probs, dim=-1)[0]
    # Calculate IHL
    return (1 + target_prob - max_nontarget_prob).sum()


def optimise_edit_components(
    model: HookedTransformer,
    forget_logits: Tensor,
    retain_logits: Tensor,
    answer_indices: Tensor,
    target_mlp_components: Tensor,
    target_attn_components: Tensor,
    optimiser: optim.Optimizer,
    localise: bool = True,
):
    """
    Uses binary tensors target_mlp_components and target_attn_components to identify which components to edit.
    """
    optimiser.zero_grad()

    # Decrease prediction score on original token
    loss_forget = inverted_hinge_loss(forget_logits, answer_indices[:, 0])
    # Make prediction probability distribution similar to predictions given rewritten prompt
    loss_rewrite = F.cross_entropy(retain_logits, answer_indices[:, 1], reduction="sum")

    kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
    forget_log_prob = F.log_softmax(forget_logits, dim=1)
    rewrite_prob = F.softmax(retain_logits, dim=1)
    loss_fluency = kl_loss(forget_log_prob, rewrite_prob)

    # Because the rewrite output might not match the rewrite target token, we do not try to match the probability distribution exactly
    loss = loss_forget + loss_rewrite + 0.5 * loss_fluency
    print(f"Total loss: {loss}, forget loss: {loss_forget}, rewrite loss: {loss_rewrite}, fluency loss: {loss_fluency}")

    loss.backward()

    if localise:
        # Mask out gradients at non-target components
        with torch.no_grad():
            for layer_idx in range(model.cfg.n_layers):
                # Attention components: W_K, W_Q, W_V, W_O matrices
                # Match attention weight shape [n_heads, d_model, d_head] or [n_heads, d_head, d_model]
                layer_attn_weight_mask = target_attn_components[:, layer_idx].view(
                    model.cfg.n_heads, 1, 1
                )
                # Match attention bias shape [n_heads, d_head]
                layer_attn_bias_mask = target_attn_components[:, layer_idx].view(
                    model.cfg.n_heads, 1
                )

                model.blocks[layer_idx].attn.W_K.grad *= layer_attn_weight_mask  # shape []
                model.blocks[layer_idx].attn.b_K.grad *= layer_attn_bias_mask

                model.blocks[layer_idx].attn.W_Q.grad *= layer_attn_weight_mask
                model.blocks[layer_idx].attn.b_Q.grad *= layer_attn_bias_mask

                model.blocks[layer_idx].attn.W_V.grad *= layer_attn_weight_mask
                model.blocks[layer_idx].attn.b_V.grad *= layer_attn_bias_mask

                model.blocks[layer_idx].attn.W_O.grad *= layer_attn_weight_mask
                # Attention output biases of shape [d_model,] - no need to mask on specific head

                # MLP neuron components: W_in, W_out matrices
                layer_mlp_mask = target_mlp_components[layer_idx]  # shape [d_mlp,]
                model.blocks[layer_idx].mlp.W_in.grad *= layer_mlp_mask.view(
                    1, model.cfg.d_mlp
                )  # shape [d_model, d_mlp]
                model.blocks[layer_idx].mlp.W_out.grad *= layer_mlp_mask.view(
                    model.cfg.d_mlp, 1
                )  # shape [d_mlp, d_model]
                model.blocks[layer_idx].mlp.b_in.grad *= layer_mlp_mask  # shape [d_mlp,]
                # MLP output biases of shape [d_model,] - no need to mask on specific neuron

    # Gradient clipping and step
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    # Update weights using optimiser
    optimiser.step()

    return loss


def localise_model(
    model: HookedTransformer,
    original_prompts: list[str],
    rewrite_prompts: list[str],
    answer_labels: Tensor,
    sample_index: int,
):
    assert len(original_prompts) == len(rewrite_prompts), f"Must have same number of prompts"
    n_samples = len(original_prompts)

    # Tokenise all together to ensure shapes stay the same
    tokenised = model.to_tokens(original_prompts + rewrite_prompts, prepend_bos=False)
    original_tokens, rewrite_tokens = [tokenised[i:i + n_samples] for i in range(0, len(tokenised), n_samples)]

    original_logits, original_cache = model.run_with_cache(original_tokens)
    original_logit_diff = logit_diff_metric(original_logits, answer_labels)

    rewrite_logits, rewrite_cache = model.run_with_cache(rewrite_tokens)
    rewrite_logit_diff = logit_diff_metric(rewrite_logits, answer_labels)

    # LOCALISATION STAGE

    target_mlp_save_path = f"attribution_results/{sample_index}_target_mlp.pt"
    target_attn_save_path = f"attribution_results/{sample_index}_target_attn.pt"

    is_mlp_saved = os.path.exists(target_mlp_save_path)
    is_attn_saved = os.path.exists(target_attn_save_path)

    if is_mlp_saved:
        print(f"Loading saved attributions for neurons")
        target_mlp = torch.load(target_mlp_save_path)
    if is_attn_saved:
        print(f"Loading saved attributions for attention heads")
        target_attn = torch.load(target_attn_save_path)

    if not is_mlp_saved and not is_mlp_saved:

        mlp_highlighted, attn_highlighted = run_attribution_steps(
            model,
            original_tokens,
            rewrite_tokens,
            answer_labels,
            original_cache,
            rewrite_cache,
            original_logit_diff,
            rewrite_logit_diff
        )

        target_mlp = identify_target_components(mlp_highlighted).to(model.cfg.device)
        target_attn = identify_target_components(attn_highlighted).to(model.cfg.device)

        torch.save(target_mlp, target_mlp_save_path)
        torch.save(target_attn, target_attn_save_path)

    return target_mlp[0], target_attn[0]


def edit_model(
    model: HookedTransformer,
    original_prompt: str,
    rewrite_prompt: str,
    answer_labels: Tensor,
    target_mlp: Tensor,
    target_attn: Tensor,
    localise: bool = True,
) -> HookedTransformer:
    print(f"\nFine tuning model...")
    model_copy = copy.deepcopy(model)
    relevant_parameters = [
        p for name, p in model_copy.named_parameters() if "attn" in name or "mlp" in name
    ]
    optimiser = optim.AdamW(relevant_parameters, lr=3e-4)
    
    # Fine tune until loss is below threshold
    loss = math.inf
    max_epochs = 10
    n = 0
    while loss > 1.5 and n < max_epochs:
        forget_logits = model_copy(original_prompt)[:, -1, :]
        rewrite_logits = model_copy(rewrite_prompt)[:, -1, :]
        # answer_index = answer_labels[i, 1].unsqueeze(0)  # Aim for rewritten answer
        
        loss = optimise_edit_components(
            model_copy, forget_logits, rewrite_logits, answer_labels, target_mlp, target_attn, optimiser, localise
        )
        n += 1

    return model_copy
