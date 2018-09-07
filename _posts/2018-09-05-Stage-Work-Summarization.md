---
layout:     post
title:      "阶段工作总结"
subtitle:   ""
date:       2018-09-05
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:

---

# 项目

## 863结题

- 2017年11-12月提交结题材料。陈榴撰写结题报告、实验室同学整理财务资料，陈星宇、李鹏伟配合修改程序，周黎、刘浩、孙晓玉编写模糊本体推理试题。
- 2018年8月14日，**北大计算所**结题汇报顺利完成。

# 论文

## 非参贝叶斯（Nonparametric Baysian）相关理论：
- Probalistic Graphcal Model，隐含狄利克雷分布模型  (LDA, Latent Dirichlet Allocation) 建模与求解
- 概率PCA, 概率矩阵分解
- 马尔可夫链-蒙特卡罗（MCMC，Markov Chain Monte Carlo），吉布斯采样（GS，Gibbs Sampling）
- 变分贝叶斯推断（VI, Variational Baysian Inference）
- 用变分推断求解LDA ，用LaTex编写笔记《Variational Inference for Latent Dirichlet Allocation》。

## 概率图模型编程框架Edward（Python，TnesorFlow）：
- 贝叶斯线性回归
- 高斯混合模型，吉布斯采样和变分推断
- 概率PCA，概率矩阵分解
- 高斯过程分类

## 研究问题

Knowledge Graph Completion / Link Prediction. 
（head, relation, ?） (head, ?, tail)  (?, relation, tail)
求解或预测？

## Stocastic Block Models 二元关系建模，关系为隐含量
依据结构建模
- Infinite Relational Model (Kemp2006)
- Mixed Membership Stochastic Block Model (Airoldi2008)
- Nonparametric Latent Feature Model (Miller2009)
- ...
- 扩展隐含关系以对三元组进行建模，利用变分贝叶斯求解。
- 撰写《Mixed Membership Stocastic Blocks for Knowledge Completion》中。

## Factorization（Embedding）
实体和关系用向量表示。
- Tranlating (not full-expressive)
    + TransE（Bordes2013）, TransH（Wang2014）, TransR (Lin2015), TransD(Ji2015)
- Neural Network (High complexity)
    + NTN (Socher2013), MLP (Dong2014), RN (Santoro2017)
- Muticative 
    + RESCAL (Nickel2011, Nickel2016a), Distmult (Yang2014), HoloE (Nickel2016), ComplEx (Trouillon2017), SimplE (Kazemi2018)
- 数据本身的稀疏性，对Muticative Factorization的评分函数产生影响， 结合 Muticative 和 概率模型进行概率张量分解，求解方法与概率矩阵分解类似。
- 撰写《Knowledge Graph Completion via Probabilistic Tensor Factorization》中。



