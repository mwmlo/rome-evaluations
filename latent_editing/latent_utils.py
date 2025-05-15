import torch
from torch import Tensor

from captum.attr import LayerIntegratedGradients

from transformer_lens.utils import get_act_name
from transformer_lens import ActivationCache, HookedTransformer
from transformer_lens.hook_points import HookPoint


# INTEGRATED GRADIENTS #


def run_from_layer_fn(
    model: HookedTransformer,
    original_input: Tensor,
    patch_layer: HookPoint,
    patch_output: Tensor,
    metric: callable,
    metric_labels: Tensor,
    reset_hooks_end=True,
):
    """
    Runs the model with a specified input to an internal layer.
    Runs with original_input, but patches in patch_output at patch_layer.
    """

    def fwd_hook(act, hook):
        assert (
            patch_output.shape == act.shape
        ), f"Patch shape {patch_output.shape} != activation shape {act.shape}"
        return patch_output + 0 * act  # Trick to ensure gradients propagate

    logits = model.run_with_hooks(
        original_input,
        fwd_hooks=[(patch_layer.name, fwd_hook)],
        reset_hooks_end=reset_hooks_end,
    )
    diff = metric(logits, metric_labels)
    return diff


def compute_layer_to_output_attributions(
    model,
    original_input,
    layer_input,
    layer_baseline,
    target_layer,
    prev_layer,
    metric,
    metric_labels,
):
    """
    Calculates layerwise integrated gradient scores for target_layer.
    Uses layer_input / layer_baseline as IG input / baseline parameters.
    """
    n_samples = original_input.size(0)

    # Take the model starting from the target layer
    def forward_fn(x):
        return run_from_layer_fn(
            model, original_input, prev_layer, x, metric, metric_labels
        )

    # Attribute to the target_layer's output
    ig_embed = LayerIntegratedGradients(
        forward_fn, target_layer, multiply_by_inputs=True
    )
    attributions, error = ig_embed.attribute(
        inputs=layer_input,
        baselines=layer_baseline,
        internal_batch_size=n_samples,
        attribute_to_layer_input=False,
        return_convergence_delta=True,
    )
    print(f"\nError (delta) for {target_layer.name} attribution: {error}")
    return attributions


def integrated_gradients(
    model: HookedTransformer,
    baseline_tokens: torch.Tensor,
    baseline_cache: ActivationCache,
    input_cache: ActivationCache,
    metric: callable,
    metric_labels,
):
    """
    Calculates layerwise integrated gradients for every MLP neuron and
    attention head in the model. Uses corrupt_cache activations as input
    and clean_cache activations as baseline.
    """
    n_samples = baseline_tokens.size(0)

    # Gradient attribution for neurons in MLP layers
    mlp_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.d_mlp)
    # Gradient attribution for attention heads
    attn_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.n_heads)

    # Calculate integrated gradients for each layer
    for layer in range(model.cfg.n_layers):

        # Gradient attribution on heads
        hook_name = get_act_name("result", layer)
        target_layer = model.hook_dict[hook_name]
        prev_layer_hook = get_act_name("z", layer)
        prev_layer = model.hook_dict[prev_layer_hook]

        layer_input = input_cache[prev_layer_hook]
        layer_baseline = baseline_cache[prev_layer_hook]

        # Shape [batch, seq_len, d_head, d_model]
        attributions = compute_layer_to_output_attributions(
            model,
            baseline_tokens,
            layer_input,
            layer_baseline,
            target_layer,
            prev_layer,
            metric,
            metric_labels,
        )

        # Calculate score based on mean over each embedding, for each token
        per_token_score = attributions.mean(dim=3)
        score = per_token_score.mean(dim=1)
        attn_results[:, layer] = score

        # Gradient attribution on MLP neurons
        hook_name = get_act_name("post", layer)
        target_layer = model.hook_dict[hook_name]
        prev_layer_hook = get_act_name("mlp_in", layer)
        prev_layer = model.hook_dict[prev_layer_hook]

        layer_input = input_cache[prev_layer_hook]
        layer_baseline = baseline_cache[prev_layer_hook]

        # Shape [batch, seq_len, d_model]
        attributions = compute_layer_to_output_attributions(
            model,
            baseline_tokens,
            layer_input,
            layer_baseline,
            target_layer,
            prev_layer,
            metric,
            metric_labels,
        )
        score = attributions.mean(dim=1)
        mlp_results[:, layer] = score

    return mlp_results, attn_results


# ACTIVATION PATCHING #


def patch_hook(
    activations: torch.Tensor, hook: HookPoint, cache: ActivationCache, idx: int
):
    """
    Replace the activations for the target component with activations from the cached run.
    """
    cached_activations = cache[hook.name]
    activations[:, :, idx] = cached_activations[:, :, idx]
    return activations


def activation_patching(
    model: HookedTransformer,
    clean_tokens: torch.Tensor,
    clean_logit_diff,
    corrupted_cache: ActivationCache,
    corrupted_logit_diff,
    metric: callable,
    metric_labels,
):
    """
    Calculate activation patching scores for every MLP neuron and attention head in the model.
    Patches corrupted_cache activations into clean run.
    """
    n_samples = clean_tokens.size(0)

    mlp_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.d_mlp)
    attn_results = torch.zeros(n_samples, model.cfg.n_layers, model.cfg.n_heads)

    baseline_diff = clean_logit_diff - corrupted_logit_diff

    for layer in range(model.cfg.n_layers):
        # Activation patching on heads
        print(f"Activation patching on attention heads in layer {layer}")
        for head in range(model.cfg.n_heads):
            hook_name = get_act_name("result", layer)

            # Temporary hook
            def temp_hook(act, hook):
                return patch_hook(act, hook, corrupted_cache, head)

            with model.hooks(fwd_hooks=[(hook_name, temp_hook)]):
                patched_logits = model(clean_tokens)

            patched_logit_diff = metric(patched_logits, metric_labels).detach()
            # Normalise result by clean and corrupted logit difference
            attn_results[:, layer, head] = (
                patched_logit_diff - clean_logit_diff
            ) / baseline_diff

        # Activation patching on MLP neurons
        print(f"Activation patching on MLP in layer {layer}")

        for neuron in range(model.cfg.d_mlp):
            hook_name = get_act_name("post", layer)

            def temp_hook(act, hook):
                return patch_hook(act, hook, corrupted_cache, neuron)

            with model.hooks(fwd_hooks=[(hook_name, temp_hook)]):
                patched_logits = model(clean_tokens)

            patched_logit_diff = metric(patched_logits, metric_labels).detach()
            # Normalise result by clean and corrupted logit difference
            mlp_results[:, layer, neuron] = (
                patched_logit_diff - clean_logit_diff
            ) / baseline_diff

    return mlp_results, attn_results


# MISCELLANEOUS #

def highlight_components(attribution_scores):
    """
    Return a binary tensor of the same shape as attribution_scores, with 1s in components
    with high attribution scores ("important" components).
    Also returns the indices of the highlighted components.
    """
    mean_scores = torch.mean(attribution_scores, dim=(1, 2), keepdim=True)
    std_scores = torch.std(attribution_scores, dim=(1, 2), keepdim=True)

    highlighted_components = attribution_scores.abs() > (mean_scores + std_scores)
    highlighted_indices = highlighted_components.nonzero()
    return highlighted_components, highlighted_indices


def logit_diff_metric(logits, metric_labels):
    """
    Calculate difference in logit values between the correct token and the incorrect token.
    """
    correct_index = metric_labels[:, 0]
    incorrect_index = metric_labels[:, 1]
    logits_last = logits[:, -1, :]
    batch_size = logits.size(0)
    correct_logits = logits_last[torch.arange(batch_size), correct_index]
    incorrect_logits = logits_last[torch.arange(batch_size), incorrect_index]
    return correct_logits - incorrect_logits