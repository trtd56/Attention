# -*- coding: utf-8 -*-

import chainer.links as L
import chainer.functions as F
from chainer import Chain, Variable

class NStepLSTM(L.NStepLSTM):

    def __init__(self, n_layers, in_size, out_size, dropout, use_cudnn):
        super(NStepLSTM, self).__init__(n_layers, in_size, out_size, dropout, use_cudnn)
        self.state_size = out_size
        self.reset_state()

    def to_cpu(self):
        super(NStepLSTM, self).to_cpu()
        if self.cx is not None:
            self.cx.to_cpu()
        if self.hx is not None:
            self.hx.to_cpu()

    def to_gpu(self, device=None):
        super(NStepLSTM, self).to_gpu(device)
        if self.cx is not None:
            self.cx.to_gpu(device)
        if self.hx is not None:
            self.hx.to_gpu(device)

    def set_state(self, cx, hx):
        assert isinstance(cx, Variable)
        assert isinstance(hx, Variable)
        cx_ = cx
        hx_ = hx
        if self.xp == numpy:
            cx_.to_cpu()
            hx_.to_cpu()
        else:
            cx_.to_gpu()
            hx_.to_gpu()
        self.cx = cx_
        self.hx = hx_

    def reset_state(self):
        self.cx = self.hx = None

    def __call__(self, xs, train=True):
        batch = len(xs)
        if self.hx is None:
            xp = self.xp
            self.hx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        if self.cx is None:
            xp = self.xp
            self.cx = Variable(
                xp.zeros((self.n_layers, batch, self.state_size), dtype=xs[0].dtype),
                volatile='auto')
        hy, cy, ys = super(NStepLSTM, self).__call__(self.hx, self.cx, xs, train)
        self.hx, self.cx = hy, cy
        return ys

class NstepLstmNet(Chain):

    def __init__(self, n_layers, n_in, n_out, gpu, dropout):
        use_cudnn = (gpu >= 0)
        super(NstepLstmNet, self).__init__(
            nstep_lstm_f = NStepLSTM(n_layers, n_in, n_out, dropout, use_cudnn),
        )

    def reset_state(self):
        self.nstep_lstm_f.reset_state()

    def __call__(self, x, train):
        x_f = [xx for xx in x]
        self.reset_state()
        h_x_f = self.nstep_lstm_f(x_f, train)
        return F.stack(h_x_f)

class BiNstepLstmNet(Chain):

    def __init__(self, n_layers, n_in, n_out, gpu, dropout):
        use_cudnn = (gpu >= 0)
        super(BiNstepLstmNet, self).__init__(
            nstep_lstm_f = NStepLSTM(n_layers, n_in, n_out, dropout, use_cudnn),
            nstep_lstm_b = NStepLSTM(n_layers, n_in, n_out, dropout, use_cudnn),
        )

    def reset_state(self):
        self.nstep_lstm_f.reset_state()
        self.nstep_lstm_b.reset_state()

    def __call__(self, x, train):
        x_f = [xx for xx in x]
        x_b = [xx[::-1] for xx in reversed(x_f)]
        self.reset_state()
        h_x_f = self.nstep_lstm_f(x_f, train)
        h_x_b = self.nstep_lstm_b(x_b, train)
        return F.stack([F.concat([f, b[::-1]]) for f,b in zip(h_x_f, h_x_b)])
