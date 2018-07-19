---
layout:     post
title:      "核函数与再生核希尔伯特空间"
subtitle:   ""
date:       2018-07-15
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Kernel
    - RKHS
---

### 向量內积与函数內积

### 向量內积

在$$R^n$$空间中，我们可以用n个独立（线性无关）向量的线性组合来表示空间中任一向量。这n个独立向量则可视为一组基（basis）。该空间有无数组基，其中基向量互相垂直的一组基成为正交基。

向量內积衡量两个向量之间的相似度。也可看做其中一个向量对另一向量的投影。

![向量內积](/img/in-post/post-kernel/inner-product.png)
图1：向量內积

$$<x,y>= |x||y|cos\theta$$

如果$$x=(x_1,x_2,...,x_n),y=(y_1,y_2,...y_n)$$, 我们可以得到
$$<x,y>=\sum_{i=1}^n x_i y_i.$$

### 函数內积

函数可以视为无穷维向量，如图：

![函数](/img/in-post/post-kernel/function-vector.png)
图2：函数

以下，用$$\mathbf{x}$$表示$$R^n$$中的一个向量，$$f$$ 代表函数本身，也就是无穷向量。$$f(\mathbf{x})$$表示点$$\mathbf{x}$$处的函数值（evaluation）。

与向量內积类似，我们可以定义函数內积：
$$<f,g>=\lim_{\Delta x\rightarrow 0}\sum_{i} f(x_i) g(x_i)\Delta x=\int f(x)g(x)dx.$$

>注：函数的维度是连续的。

函数內积常常以这样的方式出现：如果$$X$$是一个随机变量，其概率密度函数（PDF）$$f(x),i.e.,f(x)>0, \int f(x) dx = 1$$，那么期望
$$E[g(x)]=\int f(x) g(x) dx=<f,g>$$

与向量基类似，我们可以使用函数基表示其他函数。与向量基不同的是，在向量空间中我们只需要有限个向量去构造一组向量基，函数空间中则需要无限个基函数。

## 特征值分解（Eigen Decomposition）
对于实对称矩阵A，存在实数$$\lambda$$及向量$$\mathbf{x}$$:
$$\mathbf{A} \mathbf{x} = \lambda \mathbf{x}.$$

$$\lambda$$ 是A的特征值，$$\mathbf{x}$$是对应的特征向量。如果A有两个不同的特征值$$\lambda_1,\lambda_2, \lambda_1 \neq \lambda_2$$，对应的特征向量$$\mathbf{x}_1,\mathbf{x}_2$$,有：
$$\lambda_1 \mathbf{x}_1^T \mathbf{x}_2 = \mathbf{x}_1^T \mathbf{A}^T \mathbf{x}_2 = \mathbf{x}_1^T \mathbf{A} \mathbf{x}_2 = \lambda_2 \mathbf{x}_1^T \mathbf{x}_2$$
由于$$\lambda_1 \neq \lambda_2$$, 有$$\mathbf{x}_1^T \mathbf{x}_2 = 0$$，即$$\mathbf{x}_1,\mathbf{x}_2$$正交。

对于$$\mathbf{A} \in \mathcal{R}^{n \times n}$$, 我们可以找到n个特征值和特征向量。 $$\mathbf{A}$$ 可以被分解成

$$\mathbf{A} = \mathbf{Q} \mathbf{D} \mathbf{Q}^T,$$

其中$$\mathbf{Q}$$是正交矩阵($$\mathbf{Q} \mathbf{Q}^T = \mathbf{I}$$)，$$\mathbf{Q}=\left( \mathbf{q}_1, \mathbf{q}_2, \cdots, \mathbf{q}_n \right)$$，且$$\mathbf{D} = \mbox{diag} (\lambda_1, \lambda_2, \cdots, \lambda_n)$$。这里$$\{ \mathbf{q}_i \}_{i=1}^n$$是$$\mathcal{R}^n$$的一组正交基。

$$\begin{eqnarray*} \mathbf{A}=\mathbf{Q} \mathbf{D} \mathbf{Q}^T &=& \left( \mathbf{q}_1, \mathbf{q}_2, \cdots, \mathbf{q}_n \right) \begin{pmatrix}
\lambda_1\\
&\lambda_2\\
&&\ddots\\
&&&\lambda_n\\
\end{pmatrix} \begin{pmatrix} \mathbf{q}_1^T \\ \mathbf{q}_2^T \\ \vdots \\ \mathbf{q}_n^T \end{pmatrix} \\ & = & \left( \lambda_1 \mathbf{q}_1, \lambda_2 \mathbf{q}_2, \cdots, \lambda_n \mathbf{q}_n \right) \begin{pmatrix} \mathbf{q}_1^T \\ \mathbf{q}_2^T \\ \vdots \\ \mathbf{q}_n^T \end{pmatrix} \\ & = & \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T  \end{eqnarray*}.$$

## 核函数（Kernel Function）

$$f(\mathbf{x})$$可视为一个无穷维向量，那么一个二元函数$$K(\mathbf{x},\mathbf{y})$$可以视为一个无穷维矩阵。
对于满足对称性及正定性的$$K(\mathbf{x},\mathbf{y})$$：
$$K(\mathbf{x},\mathbf{y}) = K(\mathbf{y},\mathbf{x}),$$
$$\int \int f(\mathbf{x}) K(\mathbf{x},\mathbf{y}) f(\mathbf{y}) d\mathbf{x} d\mathbf{y} \geq 0,$$
我们称之为核函数。

与矩阵特征值和特征向量相似，存在特征值$$\lambda$$和特征函数$$\psi(\mathbf{x})$$使得
$$\int K(\mathbf{x},\mathbf{y}) \psi(\mathbf{x}) d\mathbf{x} = \lambda \psi(\mathbf{y}).$$

对于不同的特征值$$\lambda_1,\lambda_2, \lambda_1 \neq \lambda_2$$，对应的特征函数$$\psi_1(\mathbf{x}),\psi_2(\mathbf{x})$$，可以得到：
$$\begin{eqnarray*} \int \lambda_1 \psi_1(\mathbf{x}) \psi_2(\mathbf{x}) d\mathbf{x} & = & \int \int K(\mathbf{y},\mathbf{x}) \psi_1(\mathbf{y}) d\mathbf{y} \psi_2(\mathbf{x}) d\mathbf{x} \\ & = & \int \int K(\mathbf{x},\mathbf{y}) \psi_2(\mathbf{x}) d\mathbf{x} \psi_1(\mathbf{y}) d\mathbf{y} \\ & = & \int \lambda_2 \psi_2(\mathbf{y}) \psi_1(\mathbf{y}) d\mathbf{y} \\ & = & \int \lambda_2 \psi_2(\mathbf{x}) \psi_1(\mathbf{x}) d\mathbf{x} \end{eqnarray*}.$$

因此，
$$< \psi_1, \psi_2 > = \int \psi_1(\mathbf{x}) \psi_2(\mathbf{x}) d\mathbf{x} = 0.$$

可见，特征函数互相垂直。这里$$\psi$$代表函数本身，即无穷为向量。

对一个核函数，可以找到无穷个特征值$$\{ \lambda_i \}_{i=1}^{\infty}$$及对应无穷个特征函数$$\{ \psi_i \}_{i=1}^{\infty}$$。

类似的，可以得到(Mercer's theorem)：
$$K(\mathbf{x},\mathbf{y}) = \sum_{i=0}^{\infty} \lambda_i \psi_i (\mathbf{x}) \psi_i (\mathbf{y}).$$

这里$$< \psi_i, \psi_j > = 0, i \neq j$$。  $$\{ \psi_i \}_{i=1}^{\infty}$$ 构成函数空间的一组正交基。

常见的核函数：
多项式核：$$K(\mathbf{x},\mathbf{y}) = ( \gamma \mathbf{x}^T \mathbf{y} + C)^d$$
Gaussian radial basis kernel：$$K(\mathbf{x},\mathbf{y}) = \exp (-\gamma \Vert \mathbf{x} - \mathbf{y} \Vert^2 )$$
Sigmoid kernel:$$K(\mathbf{x},\mathbf{y}) = \tanh (\gamma \mathbf{x}^T \mathbf{y} + C )$$

## 再生核希尔伯特空间（Reproducing Kernel Hilbert Space）

原本函数之间的內积需要计算无穷维的积分，利用再生核希尔伯特空间只需计算核函数。

### 希尔伯特空间
线性空间即定义了数乘和加法的空间
定义了距离的空间叫度量空间
定义了距离的线性空间叫线性度量空间

范数$$\|x\|$$：
 - 非负性 $$\|x\| \ge 0$$
 -  $$\|\alpha x\| = |\alpha| \|x\|$$
 -  $$\|x\|+ \|y\| \ge \|x + y\|$$

定义了范数的线性空间，叫赋范线性空间。

具有线性结构同时定义了内积，同时还具有完备性的空间就叫希尔伯特空间。

### RKHS

RKHS是由核函数构成的空间。其基为$$\{ \sqrt{\lambda_i} \psi_i \}_{i=1}^{\infty}$$。空间中的任一向量或函数可以表示为基的线性组合:
$$f = \sum_{i=1}^{\infty} f_i \sqrt{\lambda_i} \psi_i.$$

我们可以将$$f$$表示成$$H$$中的向量：
$$f = (f_1, f_2, ...)_\mathcal{H}^T.$$

对于另一个函数$$g = (g_1, g_2, ...)_\mathcal{H}^T$$，有
$$< f,g >_\mathcal{H} = \sum_{i=1}^{\infty} f_i g_i.$$

对于核函数K，我们用$$K(\mathbf{x},\mathbf{y})$$表示点$$(\mathbf{x},\mathbf{y})$$处的函数值（evaluation ），这是一个标量；用$$K(\cdot,\cdot)$$表示函数本身，即无穷矩阵；用$$K(\mathbf{x},\cdot)$$表示矩阵的$$\mathbf{x}$$“列”。

$$K(\mathbf{x},\cdot) = \sum_{i=0}^{\infty} \lambda_i \psi_i (\mathbf{x}) \psi_i$$

在空间H中
$$K(\mathbf{x},\cdot) = (\sqrt{\lambda_1} \psi_1 (\mathbf{x}), \sqrt{\lambda_2} \psi_2 (\mathbf{x}), \cdots )_\mathcal{H}^T$$

因此，
$$< K(\mathbf{x},\cdot), K(\mathbf{y},\cdot) >_\mathcal{H}  = \sum_{i=0}^{\infty} \lambda_i \psi_i (\mathbf{x}) \psi_i(\mathbf{y}) = K(\mathbf{x},\mathbf{y})$$

这就是核的可再生性，即用核函数再生两个核函数的內积。H 被称为再生核希尔伯特空间（RKHS）。

### Kernel trick

定义映射
$$\boldsymbol{\Phi} (\mathbf{x}) = K(\mathbf{x},\cdot) = (\sqrt{\lambda_1} \psi_1 (\mathbf{x}), \sqrt{\lambda_2} \psi_2 (\mathbf{x}), \cdots )^T$$

即将$$\mathbf{x}$$映射到H。

$$< \boldsymbol{\Phi} (\mathbf{x}), \boldsymbol{\Phi} (\mathbf{y}) >_\mathcal{H} = < K(\mathbf{x},\cdot), K(\mathbf{y},\cdot) >_\mathcal{H} = K(\mathbf{x},\mathbf{y})$$

这样，我们无需知道这个映射及特征空间的具体形式，只需要一个对称半正定的核函数，就必然存在映射$$\boldsymbol{\Phi}$$和特征空间H，使得
$$< \boldsymbol{\Phi} (\mathbf{x}), \boldsymbol{\Phi} (\mathbf{y}) > = K(\mathbf{x},\mathbf{y})$$

这就是Kernel trick。

# 参考文献

- [A Story of Basis and Kernel - Part I: Function Basis](http://songcy.net/posts/story-of-basis-and-kernel-part-1/)
- [A Story of Basis and Kernel - Part II: Reproducing Kernel Hilbert Space](http://songcy.net/posts/story-of-basis-and-kernel-part-2/)