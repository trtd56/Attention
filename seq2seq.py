# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from bilstm import BiNstepLstmNet, NstepLstmNet
from attention import GrobalAttentionNet

import chainer.links as L
import chainer.functions as F
from chainer import Chain

class Seq2Seq(Chain):

    __LIMIT = 15

    def __init__(self, mode, n_layer, n_unit, n_vocab, gpu=-1, dropout=0.5):
        self.mode = mode
        if mode == "normal":
            super(Seq2Seq, self).__init__(
                embed = L.EmbedID(n_vocab, n_unit, ignore_label=-1),
                bilstm = NstepLstmNet(n_layer, n_unit, n_unit, gpu, dropout),
                lstm = L.LSTM(n_unit, n_unit),
                dec = L.Linear(n_unit, n_vocab),
            )
        elif mode == "bilstm":
            super(Seq2Seq, self).__init__(
                embed = L.EmbedID(n_vocab, n_unit, ignore_label=-1),
                bilstm = BiNstepLstmNet(n_layer, n_unit, n_unit, gpu, dropout),
                lstm = L.LSTM(n_unit*2, n_unit*2),
                dec = L.Linear(n_unit*2, n_vocab),
            )
        elif mode == "attention":
            super(Seq2Seq, self).__init__(
                embed = L.EmbedID(n_vocab, n_unit, ignore_label=-1),
                bilstm = BiNstepLstmNet(n_layer, n_unit, n_unit*4, gpu, dropout),
                attention = GrobalAttentionNet(n_unit*8, n_unit*8),
                lstm = L.LSTM(n_unit*8, n_unit*8),
                dec = L.Linear(n_unit*8, n_vocab),
            )

    def __call__(self, x, t):
        # encode
        h_x = self.embed(x)
        h_x = self.bilstm(h_x, train=True)
        # attention
        h_x = F.swapaxes(h_x,0,1)
        c = h_x[-1]
        self.lstm.reset_state()
        t = F.swapaxes(t,0,1)
        loss = 0
        for tt in t:
            if self.mode == "attention":
                c , _ = self.attention(h_x, c)
            # decode
            c = self.lstm(c)
            y = self.dec(c)
            loss += F.softmax_cross_entropy(y, tt)
        return loss

    def predict(self, x, eos):
        # encode
        h_x = self.embed(x)
        h_x = self.bilstm(h_x, train=False)
        # attention
        h_x = F.swapaxes(h_x,0,1)
        c = h_x[-1]
        self.lstm.reset_state()
        hyp_list = []
        att_list = []
        for _ in range(self.__LIMIT):
            c , a = self.attention(h_x, c)
            c = self.lstm(c)
            y = self.dec(c)
            hyp = y.data.argmax(1)
            hyp_list.append(hyp)
            a = F.reshape(a,(a.shape[0],a.shape[1]))
            att_list.append(a.data)
            if hyp == eos:
                break
        att_list = self.xp.array(att_list)
        att_list = self.xp.swapaxes(att_list,0,1)
        return self.xp.array(hyp_list), att_list

    def plot_heatmap(self, a_list, row_labels, column_labels, eos, img_path=None):
        row_labels = [i for i in row_labels if not i == -1]
        a_list = a_list[:,:len(row_labels)]

        fig, ax = plt.subplots()
        heatmap = ax.pcolor(a_list, cmap=plt.cm.Blues)

        ax.set_xticks(self.xp.arange(a_list.shape[1])+0.5, minor=False)
        ax.set_yticks(self.xp.arange(a_list.shape[0])+0.5, minor=False)

        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_xticklabels(row_labels, minor=False)
        ax.set_yticklabels(column_labels, minor=False)

        if img_path:
            plt.savefig(img_path)
        else:
            plt.show()

    def get_limit(self):
        return self.__LIMIT
