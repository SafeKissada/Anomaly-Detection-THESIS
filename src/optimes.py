"""Optimizer factory: builds a torch.optim.Optimizer from cfg.OPTIM.

Mirrors the pattern used by `get_criterion()` in src/losses.py — a single
switch driven entirely by config, so scripts/train.py never hardcodes an
optimizer class. Before this module existed, cfg.OPTIM was printed to the
console but never actually read: src/engine.py called `torch.optim.Adam(...)`
directly, so changing cfg.OPTIM in config.py silently had no effect. Every
call now MUST go through get_optimizer() so cfg.OPTIM is the single source
of truth for which optimizer actually runs.

Supported cfg.OPTIM values (case-insensitive): 'Adam', 'AdamW', 'SGD'
(SGD + momentum), 'RMSprop'.
"""
import torch

_VALID_OPTIMS = ('ADAM', 'ADAMW', 'SGD', 'RMSPROP')


def get_optimizer(cfg, params) -> torch.optim.Optimizer:
    """Build the optimizer named by cfg.OPTIM for the given parameters.

    Reads (all already present on Config, `momentum`/`rmsprop_*` are new):
      cfg.OPTIM             : 'Adam' | 'AdamW' | 'SGD' | 'RMSprop'
      cfg.AE_LR              : learning rate (all optimizers)
      cfg.AE_WEIGHT_DECAY    : weight decay / L2 penalty (all optimizers)
      cfg.AE_MOMENTUM         : momentum, used by SGD and RMSprop only
      cfg.AE_SGD_NESTEROV     : bool, Nesterov momentum for SGD only
                               (silently forced False if AE_MOMENTUM == 0,
                               since torch.optim.SGD requires momentum > 0
                               for nesterov=True)
      cfg.AE_RMSPROP_ALPHA    : smoothing constant, RMSprop only
      cfg.AE_RMSPROP_EPS      : numerical-stability epsilon, RMSprop only

    Adam/AdamW ignore momentum/RMSprop-specific fields entirely (they use
    their own internal beta1/beta2 defaults) — those fields are simply not
    read for those two branches.
    """
    name = cfg.OPTIM.strip().upper()

    if name == 'ADAM':
        return torch.optim.Adam(
            params,
            lr=cfg.AE_LR,
            weight_decay=cfg.AE_WEIGHT_DECAY,
        )

    elif name == 'ADAMW':
        # AdamW decouples weight decay from the gradient-based update
        # (Loshchilov & Hutter, 2019) rather than folding it into the
        # gradient like Adam/SGD do — generally the better default when
        # AE_WEIGHT_DECAY > 0.
        return torch.optim.AdamW(
            params,
            lr=cfg.AE_LR,
            weight_decay=cfg.AE_WEIGHT_DECAY,
        )

    elif name == 'SGD':
        nesterov = bool(getattr(cfg, 'AE_SGD_NESTEROV', False)) and cfg.AE_MOMENTUM > 0
        return torch.optim.SGD(
            params,
            lr=cfg.AE_LR,
            momentum=cfg.AE_MOMENTUM,
            weight_decay=cfg.AE_WEIGHT_DECAY,
            nesterov=nesterov,
        )

    elif name == 'RMSPROP':
        return torch.optim.RMSprop(
            params,
            lr=cfg.AE_LR,
            momentum=cfg.AE_MOMENTUM,
            weight_decay=cfg.AE_WEIGHT_DECAY,
            alpha=cfg.AE_RMSPROP_ALPHA,
            eps=cfg.AE_RMSPROP_EPS,
        )

    else:
        raise ValueError(
            f"Unknown cfg.OPTIM: {cfg.OPTIM!r}. Expected one of "
            f"{_VALID_OPTIMS} (case-insensitive), e.g. "
            f"Config.OPTIM = 'AdamW'.")