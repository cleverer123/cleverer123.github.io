---
layout:     post
title:      "指数分布族"
subtitle:   ""
date:       2018-09-13
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Exponential-Family
    - Mutivariate-Gaussian
    - Wishart
    - Gaussian-Wishart
---

# Important Facts

$$\frac{\partial b^T a}{\partial a} = b $$

$$\frac{\partial a^TA a}{\partial a} = (A + A^T)a $$

$$\frac{\partial tr(BA)}{\partial A} = B^T $$

$$\left| A^{-1} \right| = \frac{1}{|A|}$$

$$\frac{\partial \log|A|}{\partial A} = A^{-T} $$

$$\frac{\partial \log|A^{-1}|}{\partial A} = -A^{-T} $$

$$tr(A^TB)=vec(A) \cdot vec(B) $$

$$\frac{\partial a^TX^{-1}b}{\partial X} = -X^{-T} ab^T X^{-T}.  \text{MatrixCookBook No.61}$$

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

$$A(\eta) = - \frac{\eta_1^T \eta_2^{-1} \eta_1}{4} - \frac{\log |-2\eta_2|}{2} $$

$$E_{p(x|\theta)}[x] = \frac{\partial A(\theta)}{\partial \eta_1} = - \frac{1}{4}(2\eta_2^{-1}\eta_1) = \mu $$

$$E_{p(x|\theta)}[xx^T] = \frac{\partial A(\theta)}{\partial \eta_2} = \frac{1}{4}(-\eta_2^{-T}\eta_1\eta_1^T\eta_2^{-T}) + (-2\eta_2)^{-1} = \mu\mu^T + \Sigma $$

# Wishart Distribution

>In statistics, the Wishart distribution is a generalization to multiple dimensions of the gamma distribution. It is named in honor of John Wishart, who first formulated the distribution in 1928. It is a family of probability distributions defined over symmetric, nonnegative-definite matrix-valued random variables (“random matrices”).

$$\Lambda|W_0,\nu_0 \sim \mathcal{W}(\Lambda|W_0,\nu_0) $$

$$P(\Lambda|W_0,\nu_0) = \frac{1}{2^{\nu_0 D/2}\left|{ W_0 }\right|^{\nu_0/2}\Gamma_{D}\left({\frac {\nu_0}{2}}\right)} \left| \Lambda \right|^{(\nu_0-D-1)/2} \exp\left\{-\frac{1}{2}\operatorname {tr} ({ W_0 }^{-1} \Lambda )\right\}$$

where $$\Gamma _{D}(\cdot)$$ is the multivariate gamma funtion defined as 

$${\displaystyle \Gamma _{D}\left({\frac {\nu_0}{2}}\right)=\pi ^{D(D-1)/4}\prod _{d=1}^{D}\Gamma \left({\frac {\nu_0}{2}}-{\frac {d-1}{2}}\right).}$$

$$P(\Lambda|W_0,\nu_0) = \exp\left\{-\frac{1}{2} vec(W_0^{-1})vec(\Lambda) + \log \left| \Lambda \right|^{(\nu_0-D-1)/2} - \log 2^{\nu_0 D/2} - \log \left|{ W_0 }\right|^{\nu_0/2} - \log \Gamma_{D}\left({\frac {\nu_0}{2}}\right) \right\}\\= \exp\left\{ \left[ \begin{array}{c} -\frac{1}{2}W_0^{-1} \\ \frac{\nu_0-D-1}{2} \end{array} \right] ^T \left[ \begin{array}{c} \Lambda \\ \log \left| \Lambda \right| \end{array} \right] - \left( \frac{\nu_0}{2} \left( D\log2 + \log \left|{ W_0 }\right| \right) + \log \Gamma_{D}\left({\frac {\nu_0}{2}}\right) \right)\right\} $$

Note: Use the fact that $$tr(A^TB)=vec(A) \cdot vec(B)$$, i.e. the trace of a matrix product is much like a dot product. The matrix parameters are assumed to be vectorized (laid out in a vector) when inserted into the exponential form. Also, $$W_0$$ and $$\Lambda$$ are symmetric, so e.g. $$W_0^T=W_0$$.

$$\eta_1=-\frac{1}{2}W_0^{-1}$$

$$\eta_2=\frac{\nu_0-D-1}{2}$$

Here $$\eta_1$$ is the matrix form.

$$A(\eta) = (\eta_2 + \frac{D+1}{2})\left(D\log 2 + \log \left| -\frac{1}{2} \eta_1^{-1} \right|\right) + \log \Gamma_{D}(\eta_2 + \frac{D+1}{2})$$

$$E[\log \left| \Lambda \right|] = \frac{\partial A(\eta)}{\partial \eta_2 } = \psi_D(\frac{\nu_0}{2}) + d\log 2 + \log \left|W_0\right| $$

$$E[\Lambda] = \frac{\partial A(\eta)}{\partial \eta_1 } = \nu_0 W_0 $$

# Gaussian-Wishart (Dependent)

$$\Lambda|W_0,\nu_0 \sim \mathcal{W}(\Lambda|W_0,\nu_0) $$

$$\mu \sim \mathcal{N}(\mu_0, (\kappa_0\Lambda)^{-1})$$

$$P(\mu|\Lambda) = \frac{ \left| \kappa_0 \Lambda \right|^{\frac{1}{2}} }{ (2\pi)^{D/2} } \exp\left\{ -\frac{\kappa_0}{2} \left(\mu - \mu_0 \right)^T \Lambda \left(\mu-\mu_0 \right) \right\} $$

$$P(\mu,\Lambda) = \frac{ \kappa_0^{1/2} \left| \Lambda \right|^{(\nu_0-D)/2} }{ (2\pi)^{D/2} 2^{\nu_0 D/2}\left|{ W_0 }\right|^{\nu_0/2}\Gamma_{D}\left({\frac {\nu_0}{2}}\right) } \exp\left\{ -\frac{\kappa_0}{2} \left(\mu - \mu_0 \right)^T \Lambda \left(\mu-\mu_0 \right) -\frac{1}{2}\operatorname {tr} ({ W_0 }^{-1} \Lambda ) \right\} \\
= \exp\left\{ -\frac{\kappa_0}{2}\mu^T\Lambda\mu + \kappa_0\mu_0^T\Lambda\mu - \frac{\kappa_0}{2}\mu_0^T\Lambda\mu_0 - \frac{1}{2}vec(W_0^{-1})vec(\Lambda) + \frac{\nu_0-D}{2}\log\left| \Lambda \right| \\ - \left( -\frac{1}{2}\log\kappa_0 + \frac{\nu_0}{2}(D\log2 + \log \left| W_0 \right| ) + \frac{D}{2}\log2\pi + \log \Gamma_{D}\left({\frac {\nu_0}{2}}\right) \right) \right\}  \\
= \exp\left\{ \left[ \begin{array}{c} -\frac{\kappa_0}{2} \\ \kappa_0\mu_0 \\ -\frac{W_0^{-1} + \kappa_0\mu_0\mu_0^T }{2} \\ \frac{\nu_0-D}{2} \end{array} \right]^T \left[ \begin{array}{c} \mu^T\Lambda\mu \\ \Lambda\mu \\ \Lambda \\ \log\left| \Lambda \right| \end{array} \right] - \left( -\frac{1}{2}\log\kappa_0 + \frac{\nu_0}{2}(D\log2 + \log \left| W_0 \right| ) + \frac{D}{2}\log2\pi + \log \Gamma_{D}\left({\frac {\nu_0}{2}}\right) \right) \right\}$$

$$\eta_1 = -\frac{\kappa_0}{2}$$

$$\eta_2 = \kappa_0\mu_0$$

$$\eta_3 = -\frac{W_0^{-1} + \kappa_0\mu_0\mu_0^T }{2}$$

$$\eta_4 = \frac{\nu_0-D}{2}$$

Here $$\eta_2$$ is a vector, $$\eta_3$$ is a matrix.

$$A(\eta) = -\frac{1}{2}\log(-2\eta_1) + (\eta_4+D/2)(D\log2 + \log| (\frac{1}{2\eta_1}\eta_2\eta_2^T - 2\eta_3)^{-1}|) + \log \Gamma_{D}\left( \eta_4 + D/2 \right) + \frac{D}{2}\log2\pi $$

$$E[\mu^T\Lambda\mu ] = \frac{\partial A(\eta)}{\partial \eta_1 } =  $$

$$E[\Lambda\mu] = \frac{\partial A(\eta)}{\partial \eta_2 } = $$

$$E[\Lambda] = \frac{\partial A(\eta)}{\partial \eta_3 } = $$

$$E[\log\left| \Lambda \right|] = \frac{\partial A(\eta)}{\partial \eta_4 } = $$

# Gaussian-Wishart (Independent)

$$GW(\mu,\Lambda|\mu_0,\Lambda_0,\nu_0, W_0) = \mathcal{N}(\mu|\mu_0, \Lambda_0^{-1})\mathcal{W}(\Lambda|\nu_0, W_0) \\
= \frac{ \left| \Lambda_0 \right|^{1/2} \left| \Lambda \right|^{(\nu_0-D-1)/2} }{ (2\pi)^{D/2} 2^{\nu_0 D/2} \left| W_0 \right|^{\nu_0/2}\Gamma_{D}\left({\frac {\nu_0}{2}}\right) } \exp\left\{ -\frac{1}{2} \left(\mu - \mu_0 \right)^T \Lambda_0 \left(\mu-\mu_0 \right) -\frac{1}{2}\operatorname {tr} ({ W_0 }^{-1} \Lambda ) \right\} 
$$

As the Gaussian and Wishart are independent, the natrual parameters are the same form as them, so is the expectations. 


# 参考

- Wainwright M J, Jordan M I. Graphical models, exponential families, and variational inference[J]. Foundations and Trends® in Machine Learning, 2008, 1(1–2): 1-305.
- [Murphy K P. Conjugate Bayesian analysis of the Gaussian distribution[J]. def, 2007, 1(2σ2): 16.](http://cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
- [Exponential Family - Table of Distributions](https://www.wikiwand.com/en/Exponential_family#/Table_of_distributions)
- [Ardeshiri T, Özkan E, Orguner U. On reduction of mixtures of the exponential family distributions[M]. Linköping University Electronic Press, 2013.](http://liu.diva-portal.org/smash/get/diva2:661064/FULLTEXT02.pdf)
- [The Exponential Family of Distributions](http://www.cs.columbia.edu/~jebara/4771/tutorials/lecture12.pdf)
- [Exponential Families, Robert L. Wolpert, Duke University](https://www2.stat.duke.edu/courses/Spring11/sta114/lec/expofam.pdf)
- [Machine Learning (2015 HomeWork) Carnegie Mellon University](http://alex.smola.org/teaching/10-701-15/homework/hw5.pdf)




