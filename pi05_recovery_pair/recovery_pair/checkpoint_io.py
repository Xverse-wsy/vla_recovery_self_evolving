"""Preserve trained fp32 islands when reloading a mixed-precision RLinf checkpoint."""


def restore_full_checkpoint_after_dtype_setup(policy, checkpoint):
    """Reapply full weights AFTER upstream has finalized its per-parameter dtypes.

    Upstream first loads weights, converts the whole trunk to bf16, then restores
    selected parameters to fp32. That last conversion cannot recover fp32 training
    updates already rounded away. Loading into the final dtype layout preserves
    saved norms and other fp32 islands. Safetensors base loading stays unchanged.
    """
    from rlinf.models.embodiment.openpi.checkpoint import resolve_full_weights, load_full_weights
    weights = resolve_full_weights(checkpoint)
    if weights is None:
        return False
    expect_rlt = bool(getattr(getattr(policy, 'rlt_cfg', None), 'use_rlt', False))
    load_full_weights(policy, weights, expect_rlt=expect_rlt)
    return True
