"""Optimizer factory: สร้าง torch.optim.Optimizer จาก cfg.OPTIM

ตาม pattern เดียวกับ `get_criterion()` ใน src/losses.py — switch เดียวที่
ขับเคลื่อนด้วย config ทั้งหมด scripts/train.py เลยไม่ต้อง hardcode
optimizer class เลย ก่อนหน้าที่โมดูลนี้จะมี cfg.OPTIM ถูก print ขึ้นจอ
console แต่ไม่เคยถูกอ่านจริง: src/engine.py เรียก `torch.optim.Adam(...)`
ตรงๆ ทำให้แก้ cfg.OPTIM ใน config.py แล้วไม่มีผลอะไรเลยแบบเงียบๆ ตอนนี้
ทุกจุดเรียก**ต้อง**ผ่าน get_optimizer() ให้ cfg.OPTIM เป็นแหล่งความจริง
แหล่งเดียวว่า optimizer ตัวไหนถูกใช้จริง

Optimizer factory: builds a torch.optim.Optimizer from cfg.OPTIM.

Mirrors the pattern used by `get_criterion()` in src/losses.py — a single
switch driven entirely by config, so scripts/train.py never hardcodes an
optimizer class. Before this module existed, cfg.OPTIM was printed to the
console but never actually read: src/engine.py called `torch.optim.Adam(...)`
directly, so changing cfg.OPTIM in config.py silently had no effect. Every
call now MUST go through get_optimizer() so cfg.OPTIM is the single source
of truth for which optimizer actually runs.

ค่า cfg.OPTIM ที่รองรับ (ไม่สนตัวพิมพ์เล็ก-ใหญ่): 'Adam', 'AdamW', 'SGD'
(SGD + momentum), 'RMSprop'

Supported cfg.OPTIM values (case-insensitive): 'Adam', 'AdamW', 'SGD'
(SGD + momentum), 'RMSprop'.
"""
import torch

_VALID_OPTIMS = ('ADAM', 'ADAMW', 'SGD', 'RMSPROP')


def get_optimizer(cfg, params) -> torch.optim.Optimizer:
    """สร้าง optimizer ตามชื่อใน cfg.OPTIM ให้กับ parameters ที่ส่งมา

    อ่านค่าต่อไปนี้ (ทั้งหมดมีอยู่ใน Config อยู่แล้ว `momentum`/`rmsprop_*`
    เป็นของใหม่):
      cfg.OPTIM             : 'Adam' | 'AdamW' | 'SGD' | 'RMSprop'
      cfg.AE_LR              : learning rate (ทุก optimizer)
      cfg.AE_WEIGHT_DECAY    : weight decay / L2 penalty (ทุก optimizer)
      cfg.AE_MOMENTUM         : momentum ใช้เฉพาะ SGD กับ RMSprop
      cfg.AE_SGD_NESTEROV     : bool, Nesterov momentum เฉพาะ SGD
                               (บังคับเป็น False อัตโนมัติแบบเงียบๆ ถ้า
                               AE_MOMENTUM == 0 เพราะ torch.optim.SGD
                               บังคับว่า momentum ต้อง > 0 ถึงจะใช้
                               nesterov=True ได้)
      cfg.AE_RMSPROP_ALPHA    : smoothing constant เฉพาะ RMSprop
      cfg.AE_RMSPROP_EPS      : numerical-stability epsilon เฉพาะ RMSprop

    Adam/AdamW ไม่สนใจ field เกี่ยวกับ momentum/RMSprop เลย (ใช้
    beta1/beta2 ภายในของตัวเองแทน) — field พวกนี้ไม่ถูกอ่านเลยใน
    สอง branch นั้น

    Build the optimizer named by cfg.OPTIM for the given parameters.

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
        # AdamW แยก weight decay ออกจากการ update ที่อิงตาม gradient
        # (Loshchilov & Hutter, 2019) แทนที่จะรวมเข้าไปใน gradient แบบที่
        # Adam/SGD ทำ — โดยทั่วไปเป็น default ที่ดีกว่าเมื่อ
        # AE_WEIGHT_DECAY > 0
        #
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