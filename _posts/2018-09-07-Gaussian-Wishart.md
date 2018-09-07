---
layout:     post
title:      "高斯威沙特分布后验推导"
subtitle:   ""
date:       2018-09-07
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Gaussian-Wishart
---

模型表示如下：

$$\begin{align}
x_i &\sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Lambda})\\
\boldsymbol{\mu} &\sim \mathcal{N}(\boldsymbol{\mu_0}, (\kappa_0 \boldsymbol{\Lambda})^{-1})\\
\boldsymbol{\Lambda} &\sim \mathcal{W}(\upsilon_0, \mathbf{W}_0)
\end{align}$$

展开形式如下：
似然：

$$
\begin{align}
\mathcal{N}(\mathbf{x}_i &| \boldsymbol{\mu}, \boldsymbol{\Lambda}) 
\propto\notag\\ 
&|\boldsymbol{\Lambda}|^{N/2} 
\exp{\left(-\frac{1}{2}\sum_{i=1}^N \left( \mathbf{x}_i^T\boldsymbol{\Lambda}\mathbf{x}_i - 2 \boldsymbol{\mu}^T \boldsymbol{\Lambda}\mathbf{x}_i + \boldsymbol{\mu}^T\boldsymbol{\Lambda}\boldsymbol{\mu}\right) \right)}
\end{align}
$$

正态先验：
$$
\begin{align}
\mathcal{N}(\boldsymbol{\mu} &| \boldsymbol{\mu}_0, (\kappa_0 \boldsymbol{\Lambda})^{-1}) 
\propto\notag\\  
&|\boldsymbol{\Lambda}|^{1/2} 
\exp{\left(-\frac{1}{2}\left( \boldsymbol{\mu}^T\kappa_0 \boldsymbol{\Lambda}\boldsymbol{\mu} 
    - 2 \boldsymbol{\mu}^T \kappa_0 \boldsymbol{\Lambda}\boldsymbol{\mu_0} +
    \boldsymbol{\mu_0}^T \kappa_0 \boldsymbol{\Lambda}\boldsymbol{\mu_0}\right) \right)}
\end{align}
$$

威沙特先验：
$$
\begin{align}
\mathcal{W}(\boldsymbol{\Lambda} | \nu_0, \mathbf{W}_0) 
\propto
|\boldsymbol{\Lambda}|^{\frac{\nu_0-D-1}{2}} 
\exp{\left(-\frac{1}{2} tr(\mathbf{W}_0^{-1} \boldsymbol{\Lambda})\right)}
\end{align}
$$

求解正态-威沙特后验 
$$\mathcal{NW} (\mathbf{\mu}, \boldsymbol{\Lambda} | \boldsymbol{\mu^*}, \kappa^*, \nu^*, \mathbf{W^*})$$
也就是

$$\begin{align}
& \mathcal{N}(\boldsymbol{\mu^*} | \boldsymbol{\mu}, \kappa^* \boldsymbol{\Lambda}) \mathcal{W}(\boldsymbol{\Lambda} | \nu^*, \mathbf{W}^*) \\
& = (\kappa_0^*\Lambda )^{1/2} \cdot \exp{ \left[- \frac{1}{2} \left( \mu-\mu_0^* \right)^T\kappa_0^*\Lambda \left( \mu -\mu_0^* \right) \right]} 
 |\Lambda|^{(\nu_0^*-D-1)/2} \cdot \exp{\left[ -  \frac{1}{2}Tr \left( (W_0^*)^{-1}\Lambda_U \right) \right]} 
\end{align}$$

考察$$\Lambda$$的指数：
$$N+1+\nu_0-D-1 = 1 + \nu_0^*- D-1 \Rightarrow \nu_0^* = \nu_0 + N$$

考察exp指数中$$\boldsymbol{\mu}^T\boldsymbol{\Lambda}\boldsymbol{\mu}$$:
$$\kappa_0^* = N + \kappa_0$$

考察$$\boldsymbol{\mu}^T \boldsymbol{\Lambda}$$项：
$$\sum_{i=1}^Nx_i + \kappa_0\mu_0 = \kappa_0^* \mu_0^*\Rightarrow \mu_0^* = \frac{\sum_{i=1}^Nx_i + \kappa_0\mu_0}{\kappa_0+N}$$

考察剩余项：
$$\begin{align}
& tr(\mathbf{W^*}^{-1} \boldsymbol{\Lambda}) \\
& = tr(\mathbf{W}_0^{-1} \boldsymbol{\Lambda}) + \sum_{i=1}^{N}\mathbf{x}_i^T\boldsymbol{\Lambda}\mathbf{x}_i + \boldsymbol{\mu_0}^T \kappa_0 \boldsymbol{\Lambda}\boldsymbol{\mu_0} - \boldsymbol{\mu^*}^T \kappa^* \boldsymbol{\Lambda} \boldsymbol{\mu}^*  
\end{align}$$

其中：

$$\begin{align}
& \boldsymbol{\mu_0}^T \kappa_0 \boldsymbol{\Lambda}\boldsymbol{\mu_0} - \boldsymbol{\mu^*}^T \kappa^*\boldsymbol{\Lambda} \boldsymbol{\mu}^* \\
& = \kappa_0 \boldsymbol{\mu}_0^T \boldsymbol{\Lambda} \boldsymbol{\mu}_0 - \frac{1}{\kappa_0 + N}(\kappa_0^2 \boldsymbol{\mu}_0^T \boldsymbol{\Lambda} \boldsymbol{\mu}_0 + N \kappa_0 \boldsymbol{\mu}_0^T \boldsymbol{\Lambda} \boldsymbol{\bar{x}}_0 + N \kappa_0 \boldsymbol{\bar{x}}^T \boldsymbol{\Lambda} \boldsymbol{\mu}_0 + N^2 \boldsymbol{\bar{x}}^T \boldsymbol{\Lambda} \boldsymbol{\bar{x}}) \\
& = \frac{N\kappa_0}{\kappa_0 + N} \left( \boldsymbol{\bar{x}}^T \boldsymbol{\Lambda} \boldsymbol{\bar{x}} - \boldsymbol{\bar{x}}^T \boldsymbol{\Lambda} \boldsymbol{\mu}_0 - \boldsymbol{\mu}_0^T \boldsymbol{\Lambda} \boldsymbol{\bar{x}} + \boldsymbol{\mu}_0^T \boldsymbol{\Lambda} \boldsymbol{\mu}_0 \right) - N \boldsymbol{\bar{x}}^T \boldsymbol{\Lambda} \boldsymbol{\bar{x}}  
\end{align}$$




