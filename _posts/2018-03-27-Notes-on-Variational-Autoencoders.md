---
layout:     post
title:      "变分自编码器"
subtitle:   ""
date:       2018-03-27
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Variationnal Inference
---

> “”

## 前言

## 变分推断

### 原始问题
假设数据集$$X$$是由未观测到的连续随机变量$$z$$的某个随机过程生成的。该过程分为两个步骤：

- 第一步：从某个先验分布
$$p_{\theta^*}(z)$$
生成
$$z^{(i)}$$；
- 第二步：从某个条件分布 
$$p_{\theta^*}(x|z)$$
生成 
$$x^{(i)}$$。

其中，先验
$$p_{\theta^*}(z)$$
和似然
$$p_{\theta^*}(x|z)$$
分别来自参数分布族
$$p_{\theta}(z)$$
和
$$p_{\theta}(x|z)$$
，且这两个参数分布族的概率密度函数关于
$$\theta$$
和
$$z$$
几乎处处可微。

真实参数$$\theta^*$$和隐变量$$z^{(i)}$$均是未知的。

上述这两个步骤分别相当于“编码”和“解码”，这应该就是称之为“自编码变分贝叶斯”的原因吧。一个自然的想法是用最大似然法来求解未知的参数，即最大化边际似然。然而现实情况是

+ 直接由等式
$$p_{\theta}(x)=\int p_{\theta}(x|z)p_{\theta}(z)dz$$
来估计边际似然是不可能的；
+ 真实后验
$$p_{\theta}(z|x)= p_{\theta}(x|z)/p_{\theta}(z)$$
同样难以估计。

因此，只能从另外的途径来解决这个问题。

### 变分

由于MCMC算法的复杂性（对每个数据点都要进行大量采样），在大数据下情况，可能很难得到应用。因此，对于
$$p(z|x)$$
的积分，还需要采取其他近似解决方案。

变分推理的思想是，寻找一个容易处理的分布$$q(z)$$,然后用$$q(z)$$代替
$$p(z|x)$$。
分布之间的度量采用 Kullback–Leibler divergence ，其定义

$$KL(q||p) = \int q(t)\log \frac{q(t)}{p(t)}dt=E_q(\log q-\log p)=E_q(\log q)-E_q[\log p]$$

因此，我们寻找 $$q(z)$$ 的问题，转化为一个优化问题:

$$q^*(z) = argmax_{q(z) \in Q}KL(q(z)||p(z|x))$$

$$KL(q(z)||p(z|x))$$
是关于 $$q(z)$$ 函数，而 $$q(z)\in Q$$ 是一个函数的函数，因此，这是一个泛函。正如微分是于函数求极值，而变分（variation）则是于泛函求极值，。

### 变分下届 ELBO（Evidence Lower Bound Objective）

数据集$$X=\{ x^{(i)} \}_{i=1}^N$$的边际似然

$$\log p_\theta(X) = \log p_\theta(x^{(1)},\dots,x^{(N)}) = \sum_{i=1}^N \log p_\theta(x^{(i)})
$$。

引入识别模型
$$q_{\phi}(z|x)$$
来近似真实后验分布
$$p_{\theta}(z|x)$$，这相当于一个概率编码器。这时，我们用KL散度来度量
$$q_{\phi}(z|x)$$。

$$KL(q_{\phi}(z|x)||p_{\theta}(z|x)) = E_{q_{\phi}(z|x)}[\log q_{\phi}(z|x)] -E[\log p_{\theta}(z, x)] + \log p_{\theta}(x)$$

令：

$$\mathcal{L}(\theta,\phi;x) = \mathop{E_{q_{\phi}(z|x)}}\left[\log p_\theta(x,z) -\log q_{\phi}(z|x) \right]\tag{1}$$

则有$$x$$的对数似然：

$$\log p_\theta(x) = \mathop{KL} \left[ q_{\phi}(z|x) \| p_{\theta}(x|z) \right] + \mathcal{L}(\theta,\phi;x)$$

>注：对于观测数据$$x^{(i)}$$,其对数似然可以写作：
$$\log p_\theta(x^{(i)}) = \mathop{KL} \left[ q_{\phi}(z|x^{(i)}) \| p_{\theta}(x^{(i)}|z) \right] + \mathcal{L}(\theta,\phi;x^{(i)})
$$

因为$$\mathop{KL} [*] \ge 0$$，所以：

$$\log p_\theta(x) \ge \mathcal{L}(\theta,\phi;x)
$$

$$\mathcal{L}(\theta,\phi;x)$$称为边际似然$$\log p_\theta(x)$$的**变分下界**，也叫ELBO（Evidence Lower Bound **Objective**）。

现在，最大化边际似然的问题可以转化为最大化其变分下界的问题。

ELBO经过简单的变换可以写作下式：

$$\mathcal{L}(\theta,\phi;x) = E_{q_{\phi}(z|x)}[\log p_{\theta}(x, z) - \log q_{\phi}(z)] 
\\= E_{q_{\phi}(z|x)}[\log p_{\theta}(x| z) + \log p_{\theta}(z) - \log q_{\phi}(z|x)] 
\\= - KL(q_{\phi}(z|x)||p_{\theta}(z)) + E_{q_{\phi}(z|x)}[\log p_{\theta}(x| z)] \tag{2}$$



## SGVB估计量与AEVB算法

### 再参数化（reparameterization）

作者指出，$$\mathcal{L}(\theta,\phi;x)$$对$$\phi$$的梯度方差很大，不适用于数值计算。对于选定的后验分布
$$q_{\phi}(z|x)$$，引入一个附加辅助噪声变量的可微变换$$g_{\phi}(\epsilon,x)$$来重新参数化随机变量

$$\tilde{z} = g_{\phi}(\epsilon,x),\epsilon \sim p(\epsilon)$$

这样，用Monte Carlo法来估计某个函数$$f(x)$$关于分布
$$q_{\phi}(z|x)$$ 的期望:

$$\mathop{E_{q_{\phi}(z|x^{(i)})}} \left[ f(z) \right] = \mathop{E_{p(\epsilon)}} \left[ f\left( g_{\phi}(\epsilon,x^{(i)}) \right) \right] \approx \frac{1}{L}\sum_{l=1}^L f\left( g_{\phi}(\epsilon^{(l)},x^{(i)}) \right)
$$

### SGVB（Stochastic Gradient Variational Bayes）估计量

用上述方法估计公式（1），就得到一般的 SGVB估计量$$\widetilde{\mathcal{L}}^A(\theta,\phi;x^{(i)}) \approx \mathcal{L}(\theta,\phi;x^{(i)})$$:

$$\widetilde{\mathcal{L}}^A(\theta,\phi;x^{(i)}) = \frac{1}{L}\sum_{l=1}^L \log p_{\theta} \left( x^{(i)},z^{(i,l)} \right) - \log q_{\phi} \left( z^{(i,l)}|x^{(i)} \right) $$

，其中，$$z^{(i,l)} =  g_{\phi}(\epsilon^{(l)},x^{(i)}) , \epsilon^{(l)} \sim p(\epsilon)$$

通常情况下，等式（2）中的 KL 散度
$$KL(q_{\phi}(z|x)\|p_{\theta}(z))$$能解析地计算，
$$E_{q_{\phi}(z|x)}[\log p_{\theta}(x| z)]$$需要估计。于是我们得到 SGVB 的另一个版本
$$\widetilde{\mathcal{L}}^B(\theta,\phi;x^{(i)}) \approx \mathcal{L}(\theta,\phi;x^{(i)})$$:

$$\widetilde{\mathcal{L}}^B(\theta,\phi;x^{(i)}) = -\mathop{KL} \left[ q_{\phi}(z|x) \| p_{\theta}(z) \right] + \frac{1}{L}\sum_{l=1}^L \log p_{\theta} \left( x^{(i)}|z^{(i,l)} \right) \tag{3}$$

，其中，$$z^{(i,l)} =  g_{\phi}(\epsilon^{(l)},x^{(i)}) , \epsilon^{(l)} \sim p(\epsilon)$$。

### 基于 Mini-batch 的 AEVB 算法
给定数据集$$X=\{ x^{(i)} \}_{i=1}^N$$，可基于 Mini-batch 来构造一个边际似然变分下界的估计：

$$\mathcal{L}(\theta,\phi;X) \approx \widetilde{\mathcal{L}}^M(\theta,\phi;x^M) = \frac{N}{M} \sum_{j=1}^M \widetilde{\mathcal{L}}(\theta,\phi;x^{(j)})$$

，其中 Mini-batch $$X^M=\{ x^{(j)} \}_{j=1}^M$$是从数据集 $$X$$ 中随机抽取的 $$M$$ 个数据点。只要 Mini-batch 的规模 $$M$$ 足够大，在每个数据点 $$x^{(j)}$$ 处的采样次数$$L$$可置为 1 。


## 变分自编码器
变分自编码器基于以下假设：
- 隐变量的先验为一个中心化的各向同性的高斯分布，即$$p_\theta(z) = \mathcal{N}(z;0,I)$$;
- 
$$p_\theta(x|z)$$
是一个多元高斯分布（数据为实数值时）或是一个伯努利分布（数据为二元值时），其参数通过一个MLP从z中计算；
- 真实后验
$$p_\theta(z|x)$$
的近似
$$q_{\phi}(z|x)$$
使用具有对角协方差矩阵的高斯分布，即

$$\log q_{\phi}(z|x^{(i)}) = \log \mathcal{N}(z ;\mu^{(i)},{\sigma^2}^{(i)}I)$$

，其中，均值$$\mu^{(i)}$$ 标准差 $$\sigma^{(i)}$$是编码MLP的输出。

通俗的讲就是我们把隐变量设定为标准正太分布，通过编码器
$$q_{\phi}(z|x)$$
将x编码为标准正太的z。通过解码器
$$p_\theta(z|x)$$
将z解码成新数据。

### 伯努利 MLP 作为解码器

此情况下，
$$p_\theta(x|z)$$
，是一个多元伯努利分布，其概率是通过一个单隐层全连接神经网络从 $$z$$ 中计算：

$$\log p(x|z) = \sum_{i=1}^D x_i\log y_i + (1-x_i)\log (1-y_i)$$

其中，$$y=f_\sigma(W_2 \tanh(W_1z+b_1)+b_2)$$，$$f_\sigma(*)$$是sigmoid激活函数，$$\theta =\{W_1,W_2,b_1,b_2\}$$是 MLP 的权值和偏置。

### 高斯 MLP 作为编码器或解码器

此情况下，编码器或解码器是一个具有对角协方差阵的多元高斯分布。 

1.作为编码器，
$$\log q_\phi(z|x) = \log \mathcal{N}(x;\mu,\sigma^2I)$$
，

$$\mu = W_2h+b_2$$，$$\log\sigma^2 = W_3h+b_3$$，$$h=\tanh(W_1x+b_1)$$，。

2.作为解码器时，$$\phi= (\mu, \sigma) $$，
$$\log p(x|z) = \log \mathcal{N}(z;\mu,\sigma^2I)$$

其中，$$z^{(l)} = {\mu} + {\sigma}\epsilon^{(l)}, \epsilon^{(l)} \sim \mathcal{N}(0,I)$$，$$l$$ 代表第l此采样； $$\mu = W_4h+b_4$$，$$\log\sigma^2 = W_5h+b_5$$，$$h=\tanh(W_3z+b_3)$$，$$\theta =\{W_3,W_4,W_5,b_3,b_4,b_5\}$$是 MLP 的权值和偏置。

>注：权重下标按照下午网络结构顺序编号。

![gaussian-mlp-vae](/img/in-post/post-vae/vae-gaussian-mlp.jpg)
图1：gaussian-mlp-vae

### 实现

首先给出两个高斯分布的KL(参考[详细过程](https://blog.csdn.net/NeutronT/article/details/78086340))：
$$\begin{aligned}
&\mathop{D_{KL}} \left[ \mathcal{N}(\mu_1,\Sigma_1) \| \mathcal{N}(\mu_2,\Sigma_2) \right] \\
&= \frac{1}{2} \left[ \log \frac{|\Sigma_2|}{|\Sigma_1|}- d + \mathop{tr} (\Sigma_2^{-1}\Sigma_1) + (\mu_1-\mu_2)^T\Sigma_2^{-1}(\mu_1-\mu_2) \right]
\end{aligned}$$

对于公式（3），第一部分

当$$p_\theta(z)=\mathcal{N}(0,I)$$，且近似后验
$$q_{\phi}(z|x) = \mathcal{N}(z;\mu,\sigma^2I)$$
，设 $$z$$ 的维数为 $$J$$。令 $$\mu$$ 和 $$\sigma$$ 分别表示在数据点 $$i$$处计算得到的变分均值和标准差，令 $$\mu_j$$ 和 $$\sigma_j$$ 分别表示这些向量的第$$j$$ 个元素， 则有

$$\begin{aligned}
-\mathop{KL} \left[ q_{\phi}(z|x) \| p_{\theta}(z) \right] &= -\mathop{KL} \left[ \mathcal{N}(z;\mu,\sigma^2I) \| \mathcal{N}(0,I) \right]\\
&= -\frac{1}{2} \left[ \log \frac{1}{|\sigma^2I|}- J + \mathop{tr} (\sigma^2I) + \mu^T\mu \right]\\
&= \frac{1}{2} \left[J + \log |\sigma^2I| - \mathop{tr} (\sigma^2I) - \mu^T\mu \right]\\
&= \frac{1}{2} \sum_{j=1}^J(1 + \log \sigma_j^2 - \sigma_j^2 - \mu_j^2)
\end{aligned}$$

于是有，

$$\widetilde{\mathcal{L}}^B(\theta,\phi;x^{(i)}) = \frac{1}{2} \sum_{j=1}^J(1 + \log {\sigma^{(i)}}_j^2 - {\sigma^{(i)}}_j^2 - {\mu^{(i)}}_j^2) + \frac{1}{L}\sum_{l=1}^L \log p_{\theta} \left( x^{(i)}|z^{(i,l)} \right)$$

其中，$$z^{(i,l)} = {\mu^{(i)}} + {\sigma^{(i)}}\epsilon^{(l)}, \epsilon^{(l)} \sim \mathcal{N}(0,I)$$。

![vae-process](/img/in-post/post-vae/vae-process.png)
图2：vae-process

### 高斯情况下的变分自编码器Tensorflow代码示例

``` python
 import tensorflow as tf

    class VariationalAutoencoder(object):

        def __init__(self, n_input, n_hidden, optimizer = tf.train.AdamOptimizer()):
            self.n_input = n_input
            self.n_hidden = n_hidden

            network_weights = self._initialize_weights()
            self.weights = network_weights

            # 编码器
            self.x = tf.placeholder(tf.float32, [None, self.n_input])
            self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
            self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])

            # 从高斯分布中采样
            eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype = tf.float32)
            self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
            #解码器
            self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])

            # 损失函数
            reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0)) #等式（2）第二项
            latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq
                                               - tf.square(self.z_mean)
                                               - tf.exp(self.z_log_sigma_sq), 1)#等式（2）第一项
            self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
            self.optimizer = optimizer.minimize(self.cost)

            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)

        def _initialize_weights(self):
            all_weights = dict()
            all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            all_weights['log_sigma_w1'] = tf.get_variable("log_sigma_w1", shape=[self.n_input, self.n_hidden],
                initializer=tf.contrib.layers.xavier_initializer())
            all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
            all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
            all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
            all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
            return all_weights

        def partial_fit(self, X):
            cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
            return cost

        def calc_total_cost(self, X):
            return self.sess.run(self.cost, feed_dict = {self.x: X})

        def transform(self, X):
            return self.sess.run(self.z_mean, feed_dict={self.x: X})

        def generate(self, hidden = None):
            if hidden is None:
                hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
            return self.sess.run(self.reconstruction, feed_dict={self.z: hidden})

        def reconstruct(self, X):
            return self.sess.run(self.reconstruction, feed_dict={self.x: X})

        def getWeights(self):
            return self.sess.run(self.weights['w1'])

        def getBiases(self):
            return self.sess.run(self.weights['b1'])
```

## 参考文献

- Kingma D P, Welling M. Auto-Encoding Variational Bayes[J]. stat, 2014, 1050: 10.
- [变分自编码器（Variational Autoencoder, VAE）通俗教程——邓范鑫](https://www.dengfanxin.cn/?p=334)
- [变分贝叶斯推断(Variational Bayes Inference)简介](https://blog.csdn.net/aws3217150/article/details/57072827)
- [变分自编码器（VAEs） - Gapeng的文章 - 知乎](http://zhuanlan.zhihu.com/p/25401928)
- [自编码变分贝叶斯](https://blog.csdn.net/NeutronT/article/details/78086340)





