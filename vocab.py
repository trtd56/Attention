# -*- coding: utf-8 -*-

from xp import XP

class Vocab():
    __EOS = 0
    __PAD = -1

    def __init__(self, n_vocab, leng, gpu, seed):
        XP.set_library(gpu, seed)
        self.n_vocab = n_vocab
        self.leng = leng


    def gen_train_data(self, n_data):
        train_data = []
        for _ in range(n_data):
            v = self.__gen_vocab()
            train_data.append(v)
        return train_data

    def __gen_vocab(self):
        xp = XP.lib()
        n = xp.random.randint(1, self.leng-1, 1)
        x = xp.random.randint(1, self.n_vocab, n)
        t = x[::-1]
        padding = [self.__EOS]
        padding += [self.__PAD] * (self.leng - len(x))
        x = xp.append(x, padding)
        t = xp.append(t, padding)
        return (x, t)

    def gen_batch(self, train_data, n_batch):
        xp = XP.lib()
        x_batch = []
        t_batch = []
        l_data = len(train_data)
        for i in xp.random.choice(range(l_data), l_data, replace=False):
            data = train_data[i]
            x_batch.append(data[0])
            t_batch.append(data[1])
            if len(x_batch) == n_batch:
                x_ids = XP.iarray(x_batch)
                t_ids = XP.iarray(t_batch)
                yield (x_ids, t_ids)
                x_batch = []
                t_batch = []

    def get_test_data(self):
        v = self.__gen_vocab()
        x_id = XP.iarray([v[0]])
        return x_id, v[1]

    def get_eos(self):
        return self.__EOS
