# -*- coding: utf-8 -*-

import chainer.links as L
import chainer.functions as F
from chainer import Chain


class Linear(L.Linear):

    def __call__(self, x):
        shape = x.shape
        if len(shape) == 3:
            x = F.reshape(x, (-1, shape[2]))
        y = super().__call__(x)
        if len(shape) == 3:
            y = F.reshape(y, (shape[0], shape[1], -1))
        return y

class GrobalAttentionNet(Chain):

    def __init__(self, n_enc, n_dec):
        super(GrobalAttentionNet, self).__init__(
            l1 = Linear(n_enc, n_dec),
            l2 = Linear(n_enc, n_dec),
            l3 = Linear(n_dec, 1),
        )

    def __call__(self, e, h):
        h = F.broadcast_to(h, e.shape)
        h = F.swapaxes(h,0,1)
        e = F.swapaxes(e,0,1)
        w1 = self.l1(e)
        w2 = self.l2(h)
        v = self.l3(F.tanh(w1+w2))
        a =  F.softmax(v)
        c = F.batch_matmul(h, a, transa=True)
        return F.swapaxes(c, 1, 2), a
