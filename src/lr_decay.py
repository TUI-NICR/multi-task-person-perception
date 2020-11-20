# -*- coding: utf-8 -*-
"""
.. codeauthor:: Dominik Höchemer <dominik.hoechemer@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""


class LRPolyDecay(object):
    """
    Creates a Poly Decay learning rate scheduler

    Parameters
    ----------
    lr_init: float
        The initial learning rate to decay from.
    power: float
        The exponent in the poly decay formula.
    max_iter: int
        The number of iterations over which the initial learning rate is
        decaying to zero.
    lr_min: float
        Minimal learning rate.
    verbose: int
        For printing some information, 0: quiet, 1: update messages.

    """
    def __init__(self, lr_init, power, max_iter, lr_min, verbose=0):
        super(LRPolyDecay, self).__init__()
        self._cur_iter = 0
        if not isinstance(lr_init, (list, tuple)):
            self._lr_init = [lr_init]
        else:
            self._lr_init = lr_init
        self._power = power
        self._max_iter = max_iter
        self._lr_min = lr_min
        self._verbose = verbose
        self._lr = lr_init

    def get_current_lr(self):
        return self._lr

    def update_optimizer(self, optimizer):
        # keep track of total batch count
        self._cur_iter += 1

        # set the new learning rate
        self._lr = []
        for init_lr, param_group in zip(self._lr_init, optimizer.param_groups):
            # calculate current learning rate
            lr = init_lr * (1 - (self._cur_iter/self._max_iter)) ** self._power

            # limit learning rate
            lr = max(lr, self._lr_min)

            param_group['lr'] = lr

            # remember lr
            self._lr.append(lr)
