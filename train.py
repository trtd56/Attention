# -*- coding: utf-8 -*-

import os
from chainer import optimizer, optimizers
from datetime import datetime

from seq2seq import Seq2Seq
from vocab import Vocab


GPU   = -1
SEED  = 0
LAYER = 1
UNIT  = 32
VOCAB = 4
EPOCH = 1
N_DATA = 30
N_BATCH = 2

N_TEST = 10
TEST_FILE = "./test"

if __name__ == "__main__":
    model = Seq2Seq(LAYER,UNIT,VOCAB)
    opt = optimizers.RMSprop(lr=0.01)
    opt.setup(model)

    leng = model.get_limit()
    vocab = Vocab(VOCAB, leng, GPU, SEED)
    train_data = vocab.gen_train_data(N_DATA)

    # train
    print("epoch\tloss_avg\tdatetime")
    for epoch in range(EPOCH):
        loss_list = []
        for batch in vocab.gen_batch(train_data, N_BATCH):
            x, t = batch
            model.cleargrads()
            loss = model(x, t)
            loss.backward()
            opt.update()
            loss.unchain_backward()
            loss_list.append(loss.data)
        now = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        print("{0}\t{1:.6f}\t{2}".format(epoch,sum(loss_list)/len(loss_list),now))

    # test
    if not os.path.exists(TEST_FILE):
        os.mkdir(TEST_FILE)
    for i in range(N_TEST):
        test_x, test_t = vocab.get_test_data()
        eos = vocab.get_eos()
        hyp, att = model.predict(test_x, eos)
        print(i)
        print("input:  ",test_x.data[0])
        print("predict:",hyp.T[0])
        print("answer: ",test_t)
        print()
        path = TEST_FILE + "/{0:02d}.png".format(i)
        model.plot_heatmap(att[0], test_x.data[0],hyp.T[0], eos, path)
