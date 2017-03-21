# -*- coding: utf-8 -*-

import os
from chainer import optimizer, optimizers
from datetime import datetime
import chainer.links as L

from seq2seq import Seq2Seq
from vocab import Vocab


GPU   = -1
SEED  = 0
LAYER = 1
UNIT  = 32
VOCAB = 5
EPOCH = 200
N_DATA = 3000
N_BATCH = 20

MODE = "attention"

N_TEST = 10
TEST_FILE = "./test"

def plot_heatmap(model, test_batch, eos, epoch):
    for batch in test_batch:
        x, t, path = batch
        hyp, att = model.predict(x, eos)
        file_path = path + "/{0:03d}.png".format(epoch)
        model.plot_heatmap(att[0], x.data[0], hyp.T[0], eos, file_path)

if __name__ == "__main__":

    model = Seq2Seq(MODE, LAYER, UNIT, VOCAB)
    opt = optimizers.SGD()
    opt.setup(model)

    leng = model.get_limit()
    vocab = Vocab(VOCAB, leng, GPU, SEED)
    train_data = vocab.gen_train_data(N_DATA)

    # make test data
    if not os.path.exists(TEST_FILE):
        os.mkdir(TEST_FILE)
    test_batch = []
    for i in range(N_TEST):
        test_x, test_t = vocab.get_test_data()
        path = TEST_FILE + "/{0:02d}".format(i+1)
        if not os.path.exists(path):
            os.mkdir(path)
        test_batch.append((test_x, test_t, path))
    eos = vocab.get_eos()

    # train start
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
        if MODE == "attention":
            plot_heatmap(model, test_batch, eos, epoch)
