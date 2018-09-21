---
layout:     post
title:      "矩阵分解及变分"
subtitle:   ""
date:       2018-08-26
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Probabilistc Matrix Factorization
    - Variational-Inference
---

# PMF 
假设有N个用户和M个物品，那么就形成了一个$$N\times M$$评分矩阵$$\textbf{R}$$，通常$$\textbf{R}$$非常稀疏，只有不到1%的元素是已知的，而我们要估计出缺失元素的值。PMF假设评分矩阵中的元素$$\textbf{R}_{i,j}$$ 是由用户的潜在偏好向量$$U_i$$和物品的潜在属性向量$$V_j$$的内积决定的，即：

$$\textbf{R}_{i,j}\sim \mathbf{N}(U_i^TV_j,\sigma^2)$$
其中$$\textbf{N}$$表示正态分布。则观测到的评分矩阵条件概率为：

$$p(\textbf{R}|U,V,\sigma^2)\sim \prod_{i=1}^{N}\prod_{j=1}^{M}\mathcal{N}(U_i^TV_j,\sigma^2)^{I_{ij}}$$

$$I_{i,j}$$是指示函数，若观测到$$\textbf{R}_{i,j}$$则其值为1，否则为0。再假设用户偏好向量和物品偏好向量也都服从正态分布，即：

$$p(U|\sigma_U)\sim \prod_{i=1}^N\mathcal{N}(0,\sigma_U^2\textbf{I}), \\ p(V|\sigma_V)\sim  \prod_{j=1}^M\mathcal{N}(0,\sigma_V^2\textbf{I})$$

根据 后验=先验 X 似然，可以得出潜变量U,V的后验概率为：

$$p(U,V|\textbf{R},\sigma_U^2,\sigma_V^2,\sigma^2)\propto p(\textbf{R}|U,V,\sigma^2)p(U|\sigma_U^2)p(V|\sigma_V^2)$$

两边取对数得到:

$$\ln{p(U,V|\textbf{R},\sigma_U^2,\sigma_V^2,\sigma^2)}=-\frac{1}{2\sigma^2}\sum_{i=1}^N\sum_{j=1}^MI_{i,j}(\textbf{R}_{i,j}-U_i^TV_j)^2-\frac{1}{2\sigma_U^2}\sum_{i=1}^MU_i^TU_i-\frac{1}{2\sigma_V^2}\sum_{j=1}^NV_j^TV_i \\
-\frac{1}{2}((\sum_{i=1}^N\sum_{j=1}^MI_{i,j})\ln\sigma^2-NK\ln\sigma_U^2-MK\ln\sigma_V^2)+C $$

其中K是潜变量的维度，C是无关常数。最大后验概率等价最大化目标函数：

$$E=-\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^MI_{i,j}(\textbf{R}_{i,j}-U_i^TV_j)^2-\frac{\lambda_U}{2}\sum_{i=1}^MU_i^TU_i-\frac{\lambda_V}{2}\sum_{j=1}^NV_j^TV_i$$

其中$$\lambda_U=\frac{\sigma^2}{\sigma_U^2}$$,$$\lambda_V=\frac{\sigma^2}{\sigma_V^2}$$,这就变成我们熟悉的最小化平方差和正则化项之和的形式。

对$$U_i,V_j$$求导:

$$\frac{\partial E}{ \partial U_i}=(\textbf{R}_{i,j}-U_i^TV_j)V_j-\lambda_UU_i \\
\frac{\partial E}{ \partial V_j}=(\textbf{R}_{i,j}-U_i^TV_j)U_i-\lambda_VV_j$$

用SGD更新U_i,V_j

$$U_i=U_i+\alpha\frac{\partial E}{ \partial U_i}\\
V_j=V_j+\alpha\frac{\partial E}{ \partial V_j}$$

直到收敛或达到最大迭代次数.

# Bayesian PMF

$$p(U) \sim \mathcal{N}(U|\mu_U, \Lambda_U^{-1}) = \prod_i \mathcal{N}(U_i|\mu_U,\Lambda_U^{-1})$$

$$p(V) \sim \mathcal{N}(V|\mu_V, \Lambda_V^{-1}) = \prod_j \mathcal{N}(V_j|\mu_V,\Lambda_V^{-1})$$

所有用户共享一组超参数$$\mu_U,\Lambda_U$$，尺寸为D×1,D×D。所有物品同样共享一组超参数$$\mu_V,\Lambda_V$$。 超参数$$\mu,\Lambda$$服从Gaussian-Wishart分布，表示为一个均值的高斯分布，以及一个协方差的威沙特分布的乘积。

$$p\left(\mu,\Lambda \right)\sim \mathcal{N}(\mu|\mu_0,(\kappa_0\Lambda)^{-1}) \cdot \mathcal{W}(\Lambda|W_0,\nu_0)$$

其中，$$\mu_0=0, W_0=I,\nu_0=D$$。未书写简便，$$\Theta_U=\{\mu_U,\Lambda_U\}$$,$$\Theta_V=\{ \mu_V,\Lambda_V \}$$。

## 后验参数求解 

按照
$$p(\Theta_U|U),p(\Theta_V|V)$$
对$$\Theta_U,\Theta_V$$进行采样。 

$$p(\Theta_U|U) \sim p(U|\Theta_U) \cdot p(\Theta_U)$$

$$p(\Theta_U|U) = \mathcal{NW} (\mathbf{\mu_U}, \boldsymbol{\Lambda_U} | \boldsymbol{\mu_U^*}, \kappa_U^*, \nu_U^*, \mathbf{W_U^*}) \\
=\mathcal{N}(\boldsymbol{\mu} | \boldsymbol{\mu^*}, (\kappa^* \boldsymbol{\Lambda})^{-1}) \mathcal{W}(\boldsymbol{\Lambda} | \mathbf{W}^*, \nu^*)$$

经过求解(见高斯威沙特后验推导)：

$$\boldsymbol{\mu_U}^* = \frac{\sum_{i=1}^Nx_i + \kappa_0\boldsymbol{\mu_0}}{\kappa_0+N}$$

$$\kappa_U^* = N + \kappa_0$$

$$\begin{align}
[\mathbf{W_U^*}]^{-1} 
& = \mathbf{W}_0^{-1} + \frac{N\kappa_0}{\kappa_0 + N} (\boldsymbol{\bar{x}} - \boldsymbol{\mu}_0) (\boldsymbol{\bar{x}} - \boldsymbol{\mu}_0)^T + N \sum_{i=1}^N \left( \boldsymbol{x}_i - \boldsymbol{\bar{x}} \right) \left( \boldsymbol{x}_i - \boldsymbol{\bar{x}} \right)^T  
\end{align}$$

$$\nu_U^* = \nu_0 + N $$

## 根据超参数更新特征

根据 
$$p(U|R,V,\Theta_U)$$
，对U进行采样。根据 
$$p(V|R,U,\Theta_V)$$
，对U进行采样。

$$p\left(  U|R,V,\Theta_U \right) \cdot p\left(R|V,\Theta_U \right) = p(U,R|V,\Theta_U)$$

$$p\left( U|R,V,\Theta_U \right) \sim p( R|U,V ) \cdot p(U|\Theta_U)$$

由于右侧两个高斯函数的乘积仍然是高斯函数，故待求概率也是高斯分布。需要求其参数$$\mu^*, \Lambda^*$$。

$$\Lambda^* = \alpha \cdot \sum_{j}  V_j V_j^T + \Lambda_U$$

$$\mu^* = \left( \Lambda^*\right)^{-1} \cdot \left(  \alpha \sum_{j} R_{ij}  V_j + \Lambda_U \mu_U \right)$$

## Variational Inference PMF

$$P(R,U,V,\mu_U,\Lambda_U,\mu_V,\Lambda_V) = P(R|U,V)P(U|\mu_U,\Lambda_U)P(V|\mu_V,\Lambda_V)P(\mu_U,\Lambda_U)P(\mu_V,\Lambda_V)
$$

记$$Z={U,V,\mu,\Lambda}$$，
利用$$Q(Z)$$近似后验$$P(Z|R)$$.

$$Q(Z)=Q(U)Q(V)Q(\mu_U,\Lambda_U)Q(\mu_V,\Lambda_V)$$

$$Q(U)=\prod_{i} N(U_i|\mu^{*(U)}_{i},[\Lambda^{*(U)}_{i}]^{-1})$$

$$Q(V)=$$

$$Q(\mu_U,\Lambda_U)=\mathcal{NW} (\mathbf{\mu_U}, \boldsymbol{\Lambda_U} | \boldsymbol{\mu_U^*}, \kappa_U^*, \nu_U^*, \mathbf{W_U^*}) \\
=\mathcal{N}(\boldsymbol{\mu} | \boldsymbol{\mu^*}, (\kappa^* \boldsymbol{\Lambda})^{-1}) \mathcal{W}(\boldsymbol{\Lambda} | \nu^*, \mathbf{W}^*)$$



$$\mathcal{L}(Q(Z)) = E_{Q(Z)}[\log P(R,Z)] - E_{Q(Z)}[\log Q(Z)] \\
= E
$$










参考
- [PMF:概率矩阵分解 - 追溯星霜的文章 - 知乎](https://zhuanlan.zhihu.com/p/27399967)
- [推荐系统算法 Probabilistic Matrix Factorization](https://blog.csdn.net/shenxiaolu1984/article/details/50372909)
- [推荐系统算法 Bayesian Probabilistic Matrix Factorization](https://blog.csdn.net/shenxiaolu1984/article/details/50405659)
- [推荐系统算法 Dependent Probabilistic Matrix Factorization](https://blog.csdn.net/shenxiaolu1984/article/details/50382566)
- Incorporating side information in probabilistic matrix factorization with gaussian processes
- [推荐算法 Non-negtive Matrix Factorization](https://blog.csdn.net/google19890102/article/details/51190313)
- Algorithm for Non-negative Matrix Factorization
- [Projected gradient methods for non-negative matrix factorization](https://www.csie.ntu.edu.tw/~cjlin/nmf/)