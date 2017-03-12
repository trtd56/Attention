# -*- coding: utf-8 -*-

from chainer import optimizer, optimizers

from seq2seq import Seq2Seq
from xp import XP

X_IDS = [[1,2,3,0], [3,1,2,3]]
T_IDS = [[0,3,2,1], [3,2,1,3]]
GPU   = -1

LAYER = 1
UNIT  = 3
VOCAB = 8
EPOCH = 1

if __name__ == "__main__":
    XP.set_library(gpu=GPU)
    x_ids = XP.iarray(X_IDS)
    t_ids = XP.iarray(T_IDS)

    model = Seq2Seq(LAYER,UNIT,VOCAB)
    opt = optimizers.RMSprop(lr=0.01)
    opt.setup(model)

    for epoch in range(EPOCH):
        model.cleargrads()
        loss = model(x_ids, t_ids)
        loss.backward()
        opt.update()
        loss.unchain_backward()
        print("epoch:",epoch,"/ train loss:",loss.data)
        predict, _ = model(x_ids, train=False)
        print(predict)
    _, attention = model(x_ids, train=False)
    model.plot_heatmap(attention[0],x_ids.data[0],t_ids.data[0])
