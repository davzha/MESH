try:
    # PyTorch >= 2.0
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class SA_LRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer,
        warmup_steps=10000,
        decay_rate=0.5,
        decay_steps=100000,
        last_epoch=-1,
        verbose=False,
    ):
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        super(SA_LRScheduler, self).__init__(
            optimizer, last_epoch=last_epoch, verbose=verbose
        )

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            learning_rates = [
                base_lr * self._step_count / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            learning_rates = self.base_lrs
        learning_rates = [
            lr * (self.decay_rate ** (self._step_count / self.decay_steps))
            for lr in learning_rates
        ]
        return learning_rates
