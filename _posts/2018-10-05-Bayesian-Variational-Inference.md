---
layout:     post
title:      "贝叶斯变分推断"
subtitle:   ""
date:       2018-10-05
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Variational-Inference
---

# 变分

MCMC算法（对每个数据点都要进行大量采样），在大数据下情况，可能很难得到应用。

对于隐变量模型，由于隐变量的耦合，或变量分布的复杂性，模型似然常常无法获得参数的精确解（无解析形式），因此需要采用其他手段近似。

变分推断的思想是，寻找一个容易处理的隐变量分布$$q(z)$$,然后用$$q(z)$$代替
$$p(z|x)$$。
分布之间的*度量*采用 Kullback–Leibler divergence ，其定义

$$KL(q||p) = \int q(t)\log \frac{q(t)}{p(t)}dt=E_q[\log q-\log p]=E_q[\log q]-E_q[\log p]$$

因此，我们寻找 $$q(z)$$ 的问题，转化为一个优化问题:

$$q^*(z) = argmax_{q(z) \in Q}KL(q(z)||p(z|x))$$

$$KL(q(z)||p(z|x))$$
是关于 $$q(z)$$ 函数，而 $$q(z)\in Q$$ 是一个函数的函数，因此，这是一个泛函。正如微分是于函数求极值，而变分（variation）则是于泛函求极值。


## 变分下届

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


Note from Richard Yi Da Xu _Lesson six: Variational Bayes_.

# Derivation of Variational Inference

$$\ln P(X) = \underbrace{\int Q(Z) \ln P(X,Z) dZ - \int Q(Z) \ln Q(Z) dZ}_{\mathcal{L}(Q)} + \underbrace{\int Q(Z) \ln \frac{Q(Z)}{P(Z|X)} dZ}_{KL(Q \| P)} \\
= \mathcal{L}(Q) + KL(Q \| P)$$

## Evidence Lower Bound Objective

Suppose let's choose q(Z), such that:

$$Q(Z)=\prod_{i=1}^{M} q_i(Z_i)$$

Substitute this choice into ELBO:

$$\mathcal{L}(Q) = \int Q(Z) \ln P(X,Z) dZ - \int Q(Z) \ln Q(Z) dZ \\
= \underbrace{\int \prod_{i=1}^{M} q_i(Z_i) \ln P(X,Z) dZ }_{part(1)} - \underbrace{ \int \prod_{i=1}^{M} q_i(Z_i) \sum_{i=1}^{M}\ln q_i(Z_i) }_{part(2)} $$

### Simplification of part(1)

$$\begin{equation}
part(1) = \int \prod_{i=1}^{M} q_i(Z_i) \ln P(X,Z) dZ \\ 
= \int_{Z_1} \int_{Z_2} \cdots \int_{Z_M} \prod_{i=1}^{M} q_i(Z_i) \ln P(X,Z) dZ_1,dZ_2, \cdots ,dZ_M \\
= \int_{Z_j} q_j(Z_j) \left( \int\dots\int_{Z_{i \neq j}} \prod_{i \neq j}^{M} q_i(Z_i) \ln P(X,Z) \prod_{i \neq j}^{M} dZ_i \right)dZ_j \\
= \int_{Z_j} q_j(Z_j) \left( \int\dots\int_{Z_{i \neq j}} \ln P(X,Z) \prod_{i \neq j}^{M} q_i(Z_i) dZ_i \right)dZ_j
\end{equation}$$

Meaningfully, this can be put into an expectation function of a joint probability density $$\prod_{i \neq j}^{M} q_i(Z_i)$$

$$part(1)= \int_{Z_j} q_j(Z_j) E_{i \neq j}\left[ \ln P(X,Z) \right] dZ_j$$ 

### Simplification of part(2):

$$part(2) = \int \prod_{i=1}^{M} q_i(Z_i) \sum_{i=1}^{M}\ln q_i(Z_i) \\
= \sum_{i=1}^{M} \left( \int_{Z_i} q_i(Z_i) \ln q_i(Z_i) dZ_i \right) $$

For a paticular $$q_j(Z_j)$$, the rest of the right of the sum can be treated like a constant, part(2) can be written as:

$$part(2) = \int_{Z_j} q_j(Z_j) \ln q_j(Z_j) dZ_j + const. $$

where const. are the term does not involve $$Z_j$$.

### Putting part(1) and part(2) together:

$$\mathcal{L}(Q) = \int_{Z_j} q_j(Z_j) E_{i \neq j}\left[ \ln P(X,Z) \right] dZ_j - \int_{Z_j} q_j(Z_j) \ln q_j(Z_j) dZ_j + const.$$

Note that $$E_{i \neq j}\left[ \ln P(X,Z) \right]$$ would be some $$\ln P(Z_i)$$, name it $$\ln \widetilde{P}_j(X,Z_j)$$, i.e.:

$$\ln \widetilde{P}_j(X,Z_j) = E_{i \neq j}\left[ \ln P(X,Z) \right]$$

$$\mathcal{L}(Q_j) = \int_{Z_j} q_j(Z_j) \ln \left[ \frac{\widetilde{P}_j(X,Z_j)}{Q_j(Z_j)} \right]dZ_j + const.$$

This is the same as $$-KL\left( \widetilde{P}_j(X,Z_j) \| q_j(Z_j) \right)$$.

We can find approximate and optimal $$Q_i^*(Z_i)$$, such that:

$$Q_i^*(Z_i) = \widetilde{P}_j(X,Z_j) = exp\left(E_{i \neq j}\left[ \ln P(X,Z) \right]\right)$$












