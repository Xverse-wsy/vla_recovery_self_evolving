"""One masked, normalized action distance for FM, pull and persistent push."""
import torch


def distance_per_window(x, y, mask):
    if x.shape != y.shape or x.shape != mask.shape or x.ndim != 3:
        raise ValueError('Expect equally shaped [batch,horizon,dimension] tensors')
    mask = mask.to(dtype=torch.bool, device=x.device)
    # Select before subtraction/squaring so even masked nonfinite values cannot leak.
    delta = torch.where(mask, x, torch.zeros_like(x)) - torch.where(mask, y, torch.zeros_like(y))
    count = mask.sum(dim=(1, 2))
    values = delta.square().sum(dim=(1, 2)) / count.clamp_min(1)
    return values, count > 0


def mean_nonempty(values, valid):
    if valid.any():
        return values[valid].mean()
    return values.sum() * 0


def pair_loss_per_window(generated, positive, negative, mask, lambda_push=0.1):
    if not 0 < lambda_push < 1:
        raise ValueError('Require 0 < lambda_push < 1; no hinge or distance cutoff')
    pull, valid = distance_per_window(generated, positive, mask)
    push, valid_negative = distance_per_window(generated, negative, mask)
    if not torch.equal(valid, valid_negative):
        raise RuntimeError('Pull and push must use the SAME mask')
    return pull - lambda_push * push, valid, dict(pull=pull, push=push)


def pair_loss(generated, positive, negative, mask, lambda_push=0.1):
    values, valid, terms = pair_loss_per_window(generated, positive, negative, mask, lambda_push)
    return mean_nonempty(values, valid), terms


def fm_loss_per_window(velocity, target, mask):
    return distance_per_window(velocity, target, mask)


def eta_at_update(index, total_updates, target=0.05, warmup_fraction=0.1):
    warmup = max(1, int(total_updates * warmup_fraction + 0.999999))
    return target * min(1.0, (index + 1) / warmup)
