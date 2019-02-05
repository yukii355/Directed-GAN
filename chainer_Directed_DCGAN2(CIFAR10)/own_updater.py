#!/usr/bin/python
# -*- coding: utf-8 -*-

import six     # Python2とPython3の互換性ライブラリ
import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
from chainer.datasets import get_mnist
from chainer.dataset import convert
from chainer.dataset import iterator as iterator_module
from chainer import optimizer as optimizer_module



import chainer.functions as F
import chainer.links as L

from chainer import Variable
from chainer.training import trainer
from chainer.training import extensions





class DirectedGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        #self.reconstruction_loss,\
        self.gen, self.dis = kwargs.pop('models')
        super(DirectedGANUpdater, self).__init__(*args, **kwargs)



    def update_core(self):
        gen_optimizer = self.get_optimizer('opt_gen')  # 生成モデル用Optimizerを用意する
        dis_optimizer = self.get_optimizer('opt_dis')  # 識別モデル用Optimizerを用意する






        batch = self._iterators['main'].next()
        in_arrays = self.converter(batch, self.device)
        x_batch = xp.array(x_batch)
        zero_fake = xp.zeros_lik(x_batch)
        fake_data = self.gen(train=True)
        input4dis = F.concat((fake_data, chainer.Variable(x_batch)), axis=0)
        dis_output = self.dis(input4dis, train=True)
        (dis_fake, dis_true) = F.split_axis(dis_output, 2, axis=0)
        dis_tmp = self.dis(chainer.Variable(zero_fake, train=True)
        zeros = chainer.Variable(xp.zeros(self.batch_size, dtype=np.int32))
        ones = chainer.Variable(xp.ones(self.batch_size, dtype=np.int32))

        loss_gen = F.softmax_cross_entropy(dis_fake, zeros)
        loss_dis = F.softmax_cross_entropy(dis_real, ones) + \
                   F.softmax_cross_entropy(dis_tmp, zeros)

        reporter.report({'gen/loss': loss_gen, 'dis/loss': loss_dis})
        loss_dic = {'gen': loss_gen, 'dis': loss_dis}

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.target.cleargrads()
        loss_dic[name].backward()
        optimizer.update()
