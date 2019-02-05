#!/usr/bin/python
# -*- coding: utf-8 -*-


'''
GANの論文では学習済みモデルの出力画像のみが描かれていることが多いですが、
学習に用いていないモデルに対する評価をしたいことがあります。
具体的にはVAEとGANを組み合わせている場合などです
（実装例はdsannoさんのhttps://github.com/dsanno/chainer-vae-gan が参考になると思います）
Updaterの時と同様にchainerに実装されているEvaluatorの.__init__()と.evaluate()を
以下のような形でオーバーライドします。

'''





class GAN_Evaluator(extensions.Evaluator):

    def __init__(self, iterator, generator, discriminator, converter=convert.concat_examples,
device=None, eval_hook=None, eval_func):
        if isinstance(iterator, iterator_module.Iterator):
            iterator = {'main': iterator}
        self._iterators = iterator
        self._targets = {'gen':generator, 'dis':discriminator}

        self.converter = converter
        self.device = device
        self.eval_hook = eval_hook

    def evaluate(self):
        iterator = self._iterators['main']
        gen = self._targets['gen']
        dis = self._targets['dis']

        it = copy.copy(iterator)
        summary = reporter.DictSummary()
        for batch in it:
            observation = {}
            with reporter.report_scope(observation):
                in_arrays = self.converter(batch, self.device)
                batch_size = in_arrays.shape[0]

                x_batch = xp.array(in_arrays)
                fake_data = gen(train=False)
                input4dis = F.concat((fake_data, \
                              chainer.Variable(x_batch, volatile='on')),axis=0)
                dis_output = dis(input4dis, train=False)
                (dis_fake, dis_true) = F.split_axis(dis_output, 2, axis=0)
                dis_tmp = self.dis(chainer.Variable(zero_fake, train=True)
                zeros = chainer.Variable(xp.zeros(self.batch_size, dtype=np.int32))
                ones = chainer.Variable(xp.ones(self.batch_size, dtype=np.int32))

                loss_gen = F.softmax_cross_entropy(dis_fake, zeros)
                loss_dis = F.softmax_cross_entropy(dis_real, ones) + \
                           F.softmax_cross_entropy(dis_tmp, zeros)

                observation['dis/val/loss'] = loss_dis
                observation['gen/val/loss'] = loss_gen

            summary.add(observation)

        return summary.compute_mean()



'''
使うかわかりませんが、Evaluator、Updaterともにインスタンスが保持しているChainやoptimizerを
取得するための.get_foo()という関数が用意されているため辞書の形で渡す必要があり上のように書くのが良いと思います。
他にはchainer.Variableをvolatileにしている以外に特徴はないと思います。


'''