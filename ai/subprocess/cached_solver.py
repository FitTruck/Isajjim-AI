"""
Step Caching Euler Solver for Fast-SAM3D optimization.

On "full" steps: compute velocity normally via dynamics_fn (all CFG calls).
On "cached" steps: reuse previous velocity (skip all backbone calls).

This provides significant speedup for the SS Generator where each step
involves 3 backbone calls through PointmapCFG (conditional + no-pointmap + unconditional).
With cache_stride=3 and 14 steps: 42 calls → ~18 calls (2.3x fewer backbone calls).

Reference: Fast-SAM3D (arXiv:2602.05293), Section 4.1

Compatible with both FlowMatching (SLaT) and ShortCut (SS) generators.
ShortCut passes `d` as the first positional arg after `times` in solve_iter().
"""

from sam3d_objects.model.backbone.generator.flow_matching.solver import (
    ODESolver,
    linear_approximation_step,
)


class CachedEuler(ODESolver):
    """
    Euler solver with step caching for accelerated diffusion inference.

    Maintains a cache of the most recent velocity computed by a full backbone
    evaluation. On cached steps, reuses the previous velocity instead of calling
    the backbone, eliminating redundant computation on steps where the velocity
    field changes slowly.

    Args:
        cache_stride: Every Nth step (after warmup) is a "full" step.
                      Steps in between reuse cached velocity.
                      stride=3 → pattern: F,C,C,F,C,C,...
        warmup_steps: Number of initial steps that always run full computation.
                      Ensures the denoising trajectory is well-established
                      before caching begins.
    """

    def __init__(
        self,
        cache_stride: int = 3,
        warmup_steps: int = 2,
    ):
        super().__init__()
        self.cache_stride = cache_stride
        self.warmup_steps = warmup_steps

    def _is_full_step(self, step_idx: int) -> bool:
        """Whether this step should run the full backbone computation."""
        if step_idx < self.warmup_steps:
            return True
        offset = step_idx - self.warmup_steps
        return (offset % self.cache_stride) == 0

    def step(self, dynamics_fn, x_t, t, dt, *args, **kwargs):
        """Standard Euler step (for solve() compatibility)."""
        velocity = dynamics_fn(x_t, t, *args, **kwargs)
        return linear_approximation_step(x_t, dt, velocity)

    def solve_iter(self, dynamics_fn, x_init, times, *args, **kwargs):
        """
        Euler solver with step caching.

        Overrides ODESolver.solve_iter() to insert caching logic.
        On full steps: compute velocity via dynamics_fn.
        On cached steps: reuse the most recent velocity.

        Args:
            dynamics_fn: Callable that computes velocity field.
                         For SS (ShortCut): _generate_dynamics(x_t, t, d, *conds)
                         For SLaT (FlowMatching): _generate_dynamics(x_t, t, *conds)
            x_init: Initial noise tensor (or dict of tensors).
            times: Time sequence [t0, t1, ..., tN].
            *args, **kwargs: Forwarded to dynamics_fn (includes `d` for ShortCut).
        """
        x_t = x_init
        cached_velocity = None

        for step_idx, (t0, t1) in enumerate(zip(times[:-1], times[1:])):
            dt = t1 - t0
            is_full = self._is_full_step(step_idx)

            if is_full or cached_velocity is None:
                # Full step: compute velocity from backbone
                velocity = dynamics_fn(x_t, t0, *args, **kwargs)
                cached_velocity = velocity
            else:
                # Cached step: reuse previous velocity
                velocity = cached_velocity

            x_t = linear_approximation_step(x_t, dt, velocity)
            yield x_t, t0
