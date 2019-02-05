#!/usr/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList

import matplotlib.pyplot as plt
import chainer.functions as F
import chainer.links as L

from chainer import Variable

from chainer.training import extensions

'''
StandardUpdaterクラスは
1. .__init__() でiterator, optimizerなどを受け取り、
2. .update_core() でミニバッチに対する処理を定義しています。

さて、　GANのUpdaterを定義するときは上記クラスを以下のようにオーバーライドします。
'''


# kwarz gs:損失関数のための引数.


# self.reconstruction_lossが必要かと思って追加したけど、意味はわかってないので微妙

class DirectedGANUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        #self.reconstruction_loss,\
        self.gen, self.dis = kwargs.pop('models')
        super(DirectedGANUpdater, self).__init__(*args, **kwargs)



# loss_dis:discriminatorの更新
    def loss_dis(self, dis, y_fake, y_real):
        batchsize = len(y_fake)
        # 本物画像に対して本物(1)を出力させたい
        # 本物を本物と判定すればするほどL1は小さくなる.
        L1 = F.sum(F.softplus(-y_real)) / batchsize

        # 偽物画像に対して偽物(0)を出力させたい
        # 偽物を偽物と判定するほどL2は小さくなる.
        L2 = F.sum(F.softplus(y_fake)) / batchsize
        loss = L1 + L2
        chainer.report({'loss': loss}, dis)
        return loss



# loss_gen:generatorの更新
# generator in Original GAN just uses this loss function.


    '''
    def loss_gen(self, gen, y_fake):
        batchsize = len(y_fake)

        # 偽物画像を入力した時のDiscriminatorの出力を本物(1)に近づける
        # 偽物で本物と眼底するほどlossは小さくなる.
        loss = F.sum(F.softplus(-y_fake)) / batchsize
        chainer.report({'loss': loss}, gen)
        return loss
    '''
    


    '''
    chainerのドキュメントによると、softplus関数の定義は
    http://docs.chainer.org/en/stable/reference/generated/chainer.functions.softplus.html
    こんな感じ.The softplus function is the smooth approximation of ReLU.
    このグラフから，Generatorの誤差はy_fakeが1に近づくほど0に縮まります．
    一方のDiscriminatorは，y_realが正の大きな値をとるほどL1が小さく，y_fakeが0に近づくほどL2が小さくなり，
    全体の誤差L1+L2が縮まります．
    x_fake   （気づいたんですがDiscriminatorの方はネットワークの最終出力が恒等関数（sigmoidとかじゃない）ので，
    出力範囲は[0,1]とは限らないですね...）
    
    '''



# reconstruction_lossの更新.
    '''
    def reconstruction_loss(self, gen, y_fake):
        batchsize = len(y_fake)

        loss = np.norm(- y_real) / batchsize
        chainer.report({'loss': loss}, observer=None)
        return loss
    '''



    '''
    def loss_optimizer(model, loss_beta=0.99):
        optimizer = chainer.optimizers.Adam(beta=loss_beta)
        optimizer.setup(model)
        optimizer.add_hook(chainer.optimizer.WeightDecay(0.0001), 'hook_dec')
        return optimizer

    opt_gen = make_optimizer(gen)
    opt_dis = make_optimizer(dis)
    '''




    # 引数にreconstrucition_lossを入れてみたけど、必要かとか、使い方がわかってない
    # もう少し数式に沿った書き方
    def reconstruction_loss(self, gen, x_real, x_fake, y_fake, beta = 0.99, epoch = 100):

        # batchsize = len(x_real)
        # batchsize2 = len(x_fake)
        #beta = 0.99
        #epoch = 100

        '''
        print((x_real[0][0]).shape)
        print(type(x_real[0][0]))
        print(type(x_real[0][1]))
        '''


        '''
        fig, ax = plt.subplots(nrows=9, ncols=12, sharex=True, sharey=True)
        ax = ax.flatten()
        img = x_real[1][0]
        #newimg = np.dstack((img[0], img[1], img[2]))
        ax[1].imshow(img, interpolation='none')

        '''
        '''
        for i in range(108):
            img = train[i][0]
            newimg = np.dstack((img[0], img[1], img[2]))
            ax[i].imshow(newimg, interpolation='none')
        '''
        '''

        ax[0].set_xticks([])
        ax[0].set_yticks([])
        plt.tight_layout()
        plt.savefig("CIFAR-10-9x12.png")
        plt.show()
        '''
        batchsize3 = len(y_fake)

        L3 = F.sum(F.softplus(-y_fake)) / batchsize3

        a = F.batch_l2_norm_squared(x_fake - x_real) / batchsize3

        # b = x_fake[0] - x_real[0]

        # aを初期化しておきたい
        # a = Variable

        '''
        for n in range(batchsize3):
            a = a + ((x_fake[n] - x_real[n]))
        '''

        # print(sum(a))







        # a_1 = F.batch_l2_norm_squared(x_fake - x_real)

        # a_0 = np.sqrt(x_fake - x_real)
        # a_1 = np.sqrt(x_fake - x_real)

        # L4 = F.sum(a[1]) / batchsize3

        # a[i]のiをDirected-GANアルゴリズムの中のSと見て、色々変えてみれば良いのか？




        L5 = sum(a) / batchsize3

        # L4 = a[1] / batchsize3

        # L4 = F.sum(F.matmul(a_0, a_1)) / batchsize3

        # L4 = F.sum((np.linalg.norm(nparray_x_fake - nparray_x_real)) ** 2) / batchsize3

        loss = (1 - beta**epoch) * L3 + (beta**epoch) * L5
        # print('this is reconstruction loss >> %.2f' % (loss, ))
        chainer.report({'loss': loss})

        return loss




    '''
    アルゴリズムの式を素直に実装した場合：
    def reconstruction_loss(self, gen, )
        
        for j in range(S)
        j = o
        loss = norm( G(z)- x )**2  

    '''



    '''
    # GANの概念的にこんな感じ
    
    for イテレーション回数 do
	        m個のノイズサンプルzを作る
	        m個の贋作G(z)を作る
	        m個の真作を取り出す
	        見破るモデルにG(z)とxを渡して誤差を計算する

	        m個のノイズサンプルzを作る
	        m個の贋作G(z)を作る
	        見破るモデルにG(z)を渡して誤差を計算する

	        それぞれの誤差を最適化機のアップデータに渡してネットワークを更新する
    end for
    '''

# update_coreでミニバッチに対する処理を定義

    def update_core(self):
        gen_optimizer = self.get_optimizer('gen')  # 生成モデル用Optimizerを用意する
        dis_optimizer = self.get_optimizer('dis')  # 識別モデル用Optimizerを用意する

        batch = self.get_iterator('main').next()   # バッチで教師データを取り出す
        x_real = Variable(self.converter(batch, self.device)) / 255.

        xp = chainer.cuda.get_array_module(x_real.data)      # 概念コードのmにあたる
        gen, dis = self.gen, self.dis
        batchsize = len(batch)

        y_real = dis(x_real)                # 真作と識別されなければいけない識別結果

        # z = xp.randomm.uniform(-1, 1, (batch_size, self.gen.z_dim))   # ノイズの生成
        # z = z.astype(dtype=xp.float32)
        z = Variable(xp.asarray(gen.make_hidden(batchsize)))
        x_fake = gen(z)                     # 贋作を作っている
        y_fake = dis(x_fake)                # 贋作と識別されなければいけない識別結果



        # DiscriminatorとGeneratorを1回ずつ更新
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)   # 真作と贋作の識別結果で識別モデルのアップデート
        gen_optimizer.update(self.reconstruction_loss, gen, x_real, x_fake, y_fake, beta = 0.99, epoch = 100)           # 贋作の識別結果で生成モデルのアップデート

        # gen_optimizer.updateに、x_real, x_fakeを足してみた


        '''
        #上のD,G1回ずつの更新が通常のGANだが、mode collapse対処のため、Unrolled GANでは
        まずk回discriminatorを更新.この時1回目の更新で得られた重みをコピーして保存しておく。
        その後Generatorを更新してから保存したDiscriminatorの重みで、
        現在のDiscriminatorの重みを上書きを行う.これを実装したものが以下の部分.
        
        dis_optimizer.update(self.loss_dis, dis, y_fake, y_real)
        if self.k == 0:
            dis.cache_discriminator_weights()
        if self.k == dis.unrolling_steps:
            gen_optimizer.update(self.loss_gen, gen, y_fake)
            dis.restore_discriminator_weights()
            self.k = -1
        self.k += 1
        
        '''