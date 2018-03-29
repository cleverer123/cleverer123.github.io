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
    - Variational Autoencoders
---

> “”

## 前言

## 变分推断
###
由于MCMC算法的复杂性（对每个数据点都要进行大量采样），在大数据下情况，可能很难得到应用。因此，对于 $$p(z|x)$$ 的积分，还需要采取其他近似解决方案。

变分推理的思想是，寻找一个容易处理的分布$$q(z)$$,然后用$$q(z)$$代替$$p(z|x)$$。
分布之间的度量采用 Kullback–Leibler divergence ，其定义

$$KL(q||p) = \int q(t)\log \frac{q(t)}{p(t)}dt=E_q(\log q-\log p)=E_q(\log q)-E_q[\log p]$$

因此，我们寻找 $$q(z)$$ 的问题，转化为一个优化问题:

$$q^*(z) = argmax_{q(z) \in Q}KL(q(z)||p(z|x))$$

$$KL(q(z)||p(z|x))$$
是关于 $$q(z)$$ 函数，而 $$q(z)\in Q$$ 是一个函数，因此，这是一个泛函。而变分（variation）求极值于泛函，正如微分求极值于函数。

### ELBO（Evidence Lower Bound Objective）
根据 KL 的定义及
$$p(z|x) = \frac{p(z, x)}{p(x)}$$

$$KL(q(z)||p(z|x)) = E[\log q(z)] -E[\log p(z, x)] + \log p(x)$$

令：

$$ELBO(q) = E[\log p(z, x)] - E[\log q(z)]$$

根据 KL 的非负性质，我们有

$$\log p(x) = KL(q(x)||p(z|x)) + ELBO(q) \ge ELBO(q)$$

ELBO 是 $$p(x)$$ 对数似似然（即证据，evidence）的一个下限（lower bound）。

对于给定的数据集，$$p(x)$$ 为常数，最小化 KL 等价于最大化 ELBO 。


## 变分自编码器

对于观测数据$$x_{i}$$,其对数似然可以写作：

$$\log p_\theta(x_{i} = KL(q(z)||p_\theta(z|x_{i})) + ELBO(\theta, q; x_{i}))$$

根据上面的定义,ELBO经过简单的变换可以写作下式：

$$ELBO(\theta, q; x)) = E[\log p(z, x)] - E[\log q(z)] \\= 
E[\log p(x| z)] +E[\log p(z)] - E[\log q(z)] = - KL(q(z)||p(z)) + E_{q(z)}[\log p(x| z)] $$

因此，我们优化的目标可以分解成等号右边的两项。


## 参考文献

- [变分贝叶斯推断(Variational Bayes Inference)简介](https://blog.csdn.net/aws3217150/article/details/57072827)
- [变分自编码器（VAEs） - Gapeng的文章 - 知乎](http://zhuanlan.zhihu.com/p/25401928)
- [自编码变分贝叶斯](https://blog.csdn.net/NeutronT/article/details/78086340)





