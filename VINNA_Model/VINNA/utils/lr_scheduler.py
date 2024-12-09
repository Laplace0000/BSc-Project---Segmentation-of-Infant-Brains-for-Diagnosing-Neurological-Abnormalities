import math
from typing import List
import torch
import torch.optim.lr_scheduler as scheduler

# https://detectron2.readthedocs.io/_modules/detectron2/solver/lr_scheduler.html
class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        warmup_factor: float = 0.001,
        warmup_iters: int = 1000,
        warmup_method: str = "linear",
        last_epoch: int = -1,
    ):
        self.max_iters = max_iters
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = warmup_method
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        warmup_factor = _get_warmup_factor_at_iter(
            self.warmup_method, self.last_epoch, self.warmup_iters, self.warmup_factor
        )
        # Different definitions of half-cosine with warmup are possible. For
        # simplicity we multiply the standard half-cosine schedule by the warmup
        # factor. An alternative is to start the period of the cosine at warmup_iters
        # instead of at 0. In the case that warmup_iters << max_iters the two are
        # very close to each other.
        return [
            base_lr
            * warmup_factor
            * 0.5
            * (1.0 + math.cos(math.pi * self.last_epoch / self.max_iters))
            for base_lr in self.base_lrs
        ]


    def _compute_values(self) -> List[float]:
        # The new interface
        return self.get_lr()



def _get_warmup_factor_at_iter(
    method: str, iter: int, warmup_iters: int, warmup_factor: float
) -> float:
    """
    Return the learning rate warmup factor at a specific iteration.
    See :paper:`in1k1h` for more details.

    Args:
        method (str): warmup method; either "constant" or "linear".
        iter (int): iteration at which to calculate the warmup factor.
        warmup_iters (int): the number of warmup iterations.
        warmup_factor (float): the base warmup factor (the meaning changes according
            to the method used).

    Returns:
        float: the effective warmup factor at the given iteration.
    """
    if iter >= warmup_iters:
        return 1.0

    if method == "constant":
        return warmup_factor
    elif method == "linear":
        alpha = iter / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    else:
        raise ValueError("Unknown warmup method: {}".format(method))


class CosineLR:
    def __init__(self, base_lr,  eta_min, max_epoch):
        self.base_lr = base_lr
        self.max_epoch = max_epoch
        self.eta_min = eta_min

    def lr_func_cosine(self, cur_epoch):
        """

            cur_epoch (float): the number of epoch of the current training stage.
        """
        return self.eta_min + ((self.base_lr-self.eta_min) * (math.cos(math.pi * cur_epoch / self.max_epoch) + 1.0) * 0.5)

    def set_lr(self, optimizer, epoch):
        """
        Sets the optimizer lr to the specified value.
        Args:
            optimizer (optim): the optimizer using to optimize the current network.
            new_lr (float): the new learning rate to set.
        """
        new_lr = self.get_epoch_lr(epoch)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    def get_epoch_lr(self, cur_epoch):
        """
        Retrieves the lr for the given epoch (as specified by the lr policy).
        Args:
            cur_epoch (float): the number of epoch of the current training stage.
        """
        return self.lr_func_cosine(cur_epoch)


class CosineAnnealingWarmRestartsDecay(scheduler.CosineAnnealingWarmRestarts):
    def __init__(self,  optimizer, T_0, T_mult=1, eta_min=0, last_epoch=-1):
        super(CosineAnnealingWarmRestartsDecay, self).__init__(optimizer,
                                                               T_0,
                                                               T_mult=T_mult,
                                                               eta_min=eta_min,
                                                               last_epoch=last_epoch)
        pass

    def decay_base_lr(self, curr_iter, n_epochs, n_iter):
        if self.T_cur + 1 == self.T_i:
            annealed_lrs = []
            for base_lr in self.base_lrs:
                annealed_lr = base_lr * \
                              (1 + math.cos(math.pi * curr_iter / (n_epochs * n_iter))) / 2
                annealed_lrs.append(annealed_lr)
            self.base_lrs = annealed_lrs


def get_lr_scheduler(optimzer, cfg):
    scheduler_type = cfg.OPTIMIZER.LR_SCHEDULER
    if scheduler_type == 'step_lr':
        return scheduler.StepLR(
            optimizer=optimzer,
            step_size=cfg.OPTIMIZER.STEP_SIZE,
            gamma=cfg.OPTIMIZER.GAMMA,
        )
    elif scheduler_type == "reduceLROnPlateau":
        return scheduler.ReduceLROnPlateau(
            optimizer=optimzer,
            mode=cfg.OPTIMIZER.MODE,
            factor=cfg.OPTIMIZER.FACTOR,
            patience=cfg.OPTIMIZER.PATIENCE,
            threshold=cfg.OPTIMIZER.THRESH,
            threshold_mode='rel',
            cooldown=cfg.OPTIMIZER.COOLDOWN,
            min_lr=1e-10,
            eps=1e-08,
            verbose=False
        )
    elif scheduler_type == 'cosineWarmRestarts':
        return scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimzer,
            T_0=cfg.OPTIMIZER.T_ZERO,
            T_mult=cfg.OPTIMIZER.T_MULT,
            eta_min=cfg.OPTIMIZER.ETA_MIN,
        )
    elif scheduler_type == "NoScheduler" or scheduler_type is None:
        return None
    else:
        raise ValueError(f"{scheduler_type} lr scheduler is not supported ")