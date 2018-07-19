---
layout:     post
title:      "高斯过程回归"
subtitle:   ""
date:       2018-07-16
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Gaussian Process
    - Regression
---

# 线性模型

回顾标准线性回归（高斯噪声）模型的贝叶斯分析：

$$f(\mathbf{x})=\mathbf{x}^T\mathbf{w}, y=f(\mathbf{x})+\epsilon$$

其中，$$\mathbf{x}$$是输入向量，$$\mathbf{w}$$是线性模型的权重，$$f$$是函数值，$$y$$是观测目标值。观测值与函数值之间的差值为附加噪声，服从独立同分布（i.i.d.）高斯分布。

$$\epsilon \sim N(0, \sigma^2_n)$$

似然：给定参数，观测值的概率分布。

$$p(y|\mathbf{X},\mathbf{w}) = \prod_{i=1}^n p(y_i|\mathbf{x}_i,\mathbf{w}) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi} \sigma_n}exp(- \frac{(y_i - \mathbf{x}_i^T\mathbf{w})^2}{2\sigma^2_n}) 
\\ = \frac{1}{(2\pi \sigma^2_n)^{n/2}} exp(-\frac{1}{2\sigma_n^2} |\mathbf{y}-\mathbf{X}^T\mathbf{w}|^2) = \mathcal{N}(\mathbf{X}^T\mathbf{w}, \sigma^2_n \mathbf{I})$$

其中，
$$|z|$$表示z的欧几里得长度。

贝叶斯形式下，我们给定参数一个先验。
$$\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \Sigma_p)$$

那么后验

$$p(\mathbf{w}|\mathbf{X},\mathbf{y}) \propto p(\mathbf{y}|\mathbf{X},\mathbf{w}) p(\mathbf{w})
\\ \propto  exp(-\frac{1}{2\sigma_n^2} |\mathbf{y}-\mathbf{X}^T\mathbf{w}|^2) exp(-\frac{1}{2}\mathbf{w}^T \Sigma_p^{-1} \mathbf{w}) 
\\ \propto exp(-\frac{1}{2}(\mathbf{w}-\boldsymbol{\mu}_w)^T \Sigma_w^{-1} (\mathbf{w}-\boldsymbol{\mu}_w))$$

其中，$$\boldsymbol{\mu}_w = \sigma_n^{-2} (\sigma_n^{-2}\mathbf{X}\mathbf{X}^T + \Sigma_p^{-1})^{-1} \mathbf{X}\mathbf{y} $$， $$\Sigma_w = (\sigma_n^{-2}\mathbf{X}\mathbf{X}^T + \Sigma_p^{-1} )^{-1}$$。事实上，这其实就是以$$\boldsymbol{\mu}_w $$为均值以 $$\Sigma_w$$ 为协方差矩阵的高斯分布，

$$p(\boldsymbol{w}|\mathcal{D}) \sim \mathcal{N}(\boldsymbol{\mu}_w, \Sigma_w).$$

基于先验和似然，我们求得了权重的后验概率分布，利用这个后验分布和全概率公式，数据$$\mathbf{x}_*$$的预测分布即
$$p(f_* | \mathbf{x}_*, \mathcal{D})$$ 可以表示为似然和后验对权重的卷积。

$$\begin{align}\label{eq:predictiveBayesianLinear} p(f_* | \mathbf{x}_*, \mathcal{D}) & = \int p(f_* | \mathbf{x}_*, \mathbf{w})p(\mathbf{w}|\mathcal{D})d\mathbf{w} \nonumber \\ & = \mathcal{N}(\mathbf{x}_*^{\mathrm{T}}\boldsymbol{\mu}_w, \mathbf{x}_*^{\mathrm{T}}\Sigma_w \mathbf{x}_*). \end{align}$$

>A Theorem of Gaussians (KPM book ch.4):

Given 
$$p(x) = \mathcal{N}(x|\mu_x,\Sigma_x)$$, $$p(y|x)=\mathcal{N}(y|Ax+b,\Sigma_y)$$, we get:

$$p(x|y) = \mathcal{N}(x|\mu_{x|y}, \Sigma_{x|y})$$

$$\Sigma_{x|y}^{-1} = \Sigma_x^{-1} + A^T \Sigma_y^{-1} A $$

$$\mu_{x|y} = \Sigma_{x|y}[A^T\Sigma_y^{-1}(y-b)+\Sigma_x^{-1}\mu_x]$$

$$p(y) = \int p(y|x) p(x)dx = \mathcal{N}(y|A\mu_x+b, \Sigma_y+A\Sigma_xA^T)$$

考虑noise，预测目标值：

$$\begin{equation} p(y_* | \mathbf{x}_*, \mathcal{D}) = \mathcal{N}(\mathbf{x}_*^{\mathrm{T}}\boldsymbol{\mu}_w, \mathbf{x}_*^{\mathrm{T}}\Sigma_w \mathbf{x}_* + \sigma^2_n \mathbf{I}). \end{equation}$$

## 函数空间角度

### 高斯过程回归
高斯过程可以看做无穷维多元高斯分布，由均值函数（mean function）、协方差函数（covariance function）共同决定。

考虑一个一般的带噪声回归模型$$y=f(\mathbf{x})+\epsilon$$， $$f(\mathbf{x})$$服从一个给定均值函数和协方差函数，但参数待定的高斯过程，$$f(\mathbf{x}) \sim \mathcal{GP}(\mu,k)$$，另外，噪声$$\varepsilon \sim \mathcal{N}(0, \sigma^2_n)$$

在我们给定一系列的多维（p维的）观测点 $$\{(\mathbf{x}_i,y_i)\}_{i=1}^n, \mathbf{x}_i \in \mathbb{R}^p, y_i \in \mathbb{R}$$，我们假定这些观测点满足这个模型，所以这些点的联合分布 $$[f(\mathbf{x}_1),\ldots,f(\mathbf{x}_n)] $$按照高斯过程的定义，需要满足一个多维高斯分布，即

$$[f(\mathbf{x}_1),f(\mathbf{x}_2),\ldots,f(\mathbf{x}_n)]^{\mathrm{T}} \sim \mathcal{N}(\mathbf{\mu},K),$$

这里 $$\mathbf{\mu} = [\mu(\mathbf{x}_1),\ldots,\mu(\mathbf{x}_n)]^\mathrm{T}$$ 是均值向量， K 是$$ n \times n$$ 的矩阵，其中第 $$（i,j）$$ 个元素是 $$K_{ij} = k(\mathbf{x}_i,\mathbf{x}_j)$$ 。

接下去，为了去预测 $$f^* = f(X^*)$$ ，其中 $$X_* = [\mathbf{x}^*_{1},\cdots,\mathbf{x}^*_{m}]^{\mathrm{T}}$$ ，我们考虑高斯分布的性质，即可以得到训练点和预测点的联合分布为

$$\begin{equation}\label{eq:jointdistribution} \begin{bmatrix} \mathbf{y} \\ f^* \end{bmatrix} \sim \mathcal{N} \left( \begin{bmatrix} \boldsymbol{\mu}(X) \\ \boldsymbol{\mu}(X^*) \end{bmatrix}, \begin{bmatrix} K(X,X) + \sigma^2_n \mathbf{I} \quad K(X^*,X)^{\mathrm{T}} \\ K(X^*,X) \quad \qquad K(X^*,X^*) \end{bmatrix} \right), \end{equation} $$

这里 $$\boldsymbol{\mu}(X) = \boldsymbol{\mu}$$，$$\boldsymbol{\mu}(X^*) = [\mu(\mathbf{x}^*_{1}),\ldots,\mu(\mathbf{x}^*_{m})]^\mathrm{T}$$，$$ K(X,X)= K$$ 。 $$ K(X^*,X)$$ 是个 $$m \times n$$ 的矩阵，其中第 $$(i,j)$$个元素 $$[K(X^*,X)]_{ij} = k(\mathbf{x}^*_{i},\mathbf{x}_j)$$，  $$K(X^*,X^*)$$ 是个  $$m \times m$$ 的矩阵，其中第 $$(i,j)$$个元素 $$[K(X^*,X^*)]_{ij} = k(\mathbf{x}^*_{i},\mathbf{x}^*_{j})$$。

利用高斯分布的条件分布性质，我们可以得到

$$\begin{equation}\label{predictive} p(f^*|X,\mathbf{y},X^*) = \mathcal{N}(\hat{\boldsymbol{\mu}},\hat{\Sigma}), \end{equation}$$

其中

$$\begin{align} \hat{\boldsymbol{\mu}} &= K(X^*,X)(K(X,X) + \sigma^2_n \mathbf{I})^{-1}(\mathbf{y}-\boldsymbol{\mu}(X)) + \boldsymbol{\mu}(X^*),\label{predictive1} \\ \hat{\Sigma} &= K(X^*,X^*) - K(X^*,X)(K(X,X) + \sigma^2_n \mathbf{I})^{-1}K(X^*,X)^\mathrm{T} \label{predictive2}. \end{align}$$

最后将噪声考虑进来的话就是，

$$\begin{equation}\label{predictiveY} p(\mathbf{y}^*|X,\mathbf{y},X^*) = \mathcal{N}(\hat{\boldsymbol{\mu}},\hat{\Sigma}+ \sigma^2_n \mathbf{I}), \end{equation}$$

实际操作中，将均值函数考虑为0，那么预测mean和预测variance的结果可以更为简单的表示为

$$\begin{align} \hat{\boldsymbol{\mu}} &= K(X^*,X) (K(X,X) + \sigma^2_n \mathbf{I})^{-1}\mathbf{y}, \tag{1} \\ \hat{\Sigma} &= K(X^*,X^*) - K(X^*,X)(K(X,X) + \sigma^2_n \mathbf{I})^{-1}K(X^*,X)^\mathrm{T} \tag{2}. \end{align}$$

## 代码实现

>Cholesky分解:
设A是一个n阶厄米特正定矩阵(Hermitian positive-definite matrix)。Cholesky分解的目标是把A变成:$$A = LL^{T}$$。其中，L是下三角矩阵。

>Solve a linear matrix equation:
$$b=L^{-1}y \rightarrow Lb = y \rightarrow b = L\backslash y$$

$$\alpha = (K(X,X) + \sigma^2_n \mathbf{I})^{-1}\mathbf{y} $$

$$L = Cholesky(K(X,X) + \sigma^2_n \mathbf{I})$$

$$\alpha = (L^T)^{-1}L^{-1}\mathbf{y} = L^T \backslash (L \backslash \mathbf{y})$$


```python
from __future__ import division
import numpy as np
import matplotlib.pyplot as pl

""" This is code for simple GP regression. It assumes a zero mean GP Prior """

# This is the true unknown function we are trying to approximate
f = lambda x: np.sin(0.9*x).flatten()
#f = lambda x: (0.25*(x**2)).flatten()

# Define the kernel
def kernel(a, b):
    """ GP squared exponential kernel """
    kernelParameter = 1
    sqdist = np.sum(a**2,1).reshape(-1,1) + np.sum(b**2,1) - 2*np.dot(a, b.T)
    return np.exp(-.5 * (1/kernelParameter) * sqdist)

N = 10         # number of training points.
n = 50         # number of test points.
s = 0.00005    # noise variance.

# Sample some input points and noisy versions of the function evaluated at
# these points. 
X = np.random.uniform(-5, 5, size=(N,1))
y = f(X) + s*np.random.randn(N)

K = kernel(X, X)
L = np.linalg.cholesky(K + s*np.eye(N))

# points we're going to make predictions at.
Xtest = np.linspace(-5, 5, n).reshape(-1,1)

# compute the mean at our test points.
Lk = np.linalg.solve(L, kernel(X, Xtest))
mu = np.dot(Lk.T, np.linalg.solve(L, y))

# compute the variance at our test points.
K_ = kernel(Xtest, Xtest)
s2 = np.diag(K_) - np.sum(Lk**2, axis=0)
s = np.sqrt(s2)

# PLOTS:
pl.figure(1)
pl.clf()
pl.plot(X, y, 'r+', ms=20)
pl.plot(Xtest, f(Xtest), 'b-')
pl.gca().fill_between(Xtest.flat, mu-3*s, mu+3*s, color="#dddddd")
pl.plot(Xtest, mu, 'r--', lw=2)
pl.savefig('predictive.png', bbox_inches='tight')
pl.title('Mean predictions plus 3 st.deviations')
pl.axis([-5, 5, -3, 3])

# draw samples from the prior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n))
f_prior = np.dot(L, np.random.normal(size=(n,10)))
pl.figure(2)
pl.clf()
pl.plot(Xtest, f_prior)
pl.title('Ten samples from the GP prior')
pl.axis([-5, 5, -3, 3])
pl.savefig('prior.png', bbox_inches='tight')

# draw samples from the posterior at our test points.
L = np.linalg.cholesky(K_ + 1e-6*np.eye(n) - np.dot(Lk.T, Lk))
f_post = mu.reshape(-1,1) + np.dot(L, np.random.normal(size=(n,10)))
pl.figure(3)
pl.clf()
pl.plot(Xtest, f_post)
pl.title('Ten samples from the GP posterior')
pl.axis([-5, 5, -3, 3])
pl.savefig('post.png', bbox_inches='tight')

pl.show()
```
![预测均值](/img/in-post/post-gp/mean-prediction.png)

![高斯过程先验](/img/in-post/post-gp/gp-prior.png)

![高斯过程后验](/img/in-post/post-gp/gp-posterior.png)

# 参考文献

- Rasmussen, Carl Edward, and Christopher KI Williams. Gaussian processes for machine learning. Vol. 1. Cambridge: MIT press, 2006.
- [Gaussian process regression的简洁推导——从Function-space角度看 - 蓦风星吟的文章 - 知乎](https://zhuanlan.zhihu.com/p/31203558)