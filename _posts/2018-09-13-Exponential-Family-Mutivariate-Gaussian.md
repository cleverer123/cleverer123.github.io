---
layout:     post
title:      "指数分布族之多元高斯"
subtitle:   ""
date:       2018-09-13
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Exponential-Family
    - Mutivariate-Gaussian
---

# Important Facts

$$\frac{\partial b^T a}{\partial a} = b $$

$$\frac{\partial a^TA a}{\partial a} = (A + A^T)a $$

$$\frac{\partial tr(BA)}{\partial A} = B^T $$

$$\frac{\partial \log|A|}{\partial A} = A^{-T} $$




# Exponential Family

Distributions in the exponential family can be written in the following common form:

$$P(x|\theta) = h(x)exp\{\eta(\theta)^T T(x)-A(\theta)\},$$

where $$\eta(\theta)$$ is called the natural parameter, $$T(x)$$ is called the sufficient statistic, $$h(x)$$ is the underlying measure, and $$A(\theta)$$ is the log normalizer

$$\begin{equation}A(\theta) = \log\int h(x)\exp\{\eta(\theta)^T T(x) \}dx. \end{equation}$$

An important property of exponential family can be derived :

$$\begin{equation}
\frac{\partial A(\theta)}{\partial \eta(\theta)} = \frac{\int h(x)\exp\{\eta(\theta)^T T(x) \} T(x)dx / \exp\{A(\theta)\} }{\int h(x)\exp\{\eta(\theta)^T T(x) \}dx / \exp\{A(\theta)\} } = E_{P(x|\theta)}[T(x)].
\end{equation}$$

# Mutivariate-Gaussian

$$x|\theta \sim \mathcal{N}(\mu, \Sigma), \theta=\{\mu, \Sigma\}$$

$$
\begin{align*} p(x|\theta) &= \frac{1}{(2\pi)^{D/2} \left| \Sigma \right|^{1/2}} \exp \left\{-\frac{1}{2}\left(x-\mu\right)^T \Sigma^{-1}\left(x-\mu \right) \right\}\\ &=   \frac{1}{(2\pi)^{D/2}} \exp \left\{ -\frac{1}{2} \log \left|\Sigma \right| -\frac{1}{2} x^T \Sigma^{-1} x +  \mu^T \Sigma^{-1} x -\frac{1}{2} \mu^T \Sigma^{-1} \mu \right\}\\ 
&=  \frac{1}{(2\pi)^{D/2}}  \exp \left\{ {\underbrace{\left[ \begin{array}{c} \Sigma^{-1}\mu \\ -\frac{1}{2} \Sigma^{-1} \end{array} \right]}_{\eta(\theta) = [\eta_1;\eta_2]} }^T \underbrace{ \left[ \begin{array}{c} x \\ xx^T \end{array} \right] }_{T(x)}  - \underbrace{(\frac{1}{2} \log \left|\Sigma \right| + \frac{1}{2} \mu^T \Sigma^{-1} \mu )}_{A(\theta)} \right\} \end{align*}
$$

$$A(\theta) = - \frac{\eta_1^T \eta_2^{-1} \eta_1}{4} - \frac{\log |-2\eta_2|}{2} $$

$$E_{p(x|\theta)}[x] = \frac{\partial A(\theta)}{\partial \eta_1} = - \frac{1}{4}(2\eta_2^{-1}\eta_1) = \mu $$

$$E_{p(x|\theta)}[xx^T] = \frac{\partial A(\theta)}{\partial \eta_2} = \frac{1}{4}\eta_1\eta_1^T\eta_2^{-2} + (-2\eta_2)^{-1} = \mu^T\mu + \Sigma $$







