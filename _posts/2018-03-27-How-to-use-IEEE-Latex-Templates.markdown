---
layout:     post
title:      "如何使用IEEE Latex模板"
subtitle:   "Diary 2018 03 27"
date:       2018-03-27
author:     "Leo"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - Latex
---

> “Let's do it!”

# 前言

模板可以从[官网](https://www.ieee.org)获得。
Publications => Author Resources => Create your IEEE Article => IEEE Article Templates => Templates for Transactions => [WIN or MAC LaTeX2e Transactions Style File](https://ieeeauthor.wpengine.com/wp-content/uploads/WIN-or-MAC-LaTeX2e-Transactions-Style-File.zip).

文件夹包含`bare_conf.tex`（conference）,
 `bare_jrnl.tex`（journal）, 
 `bare_jrnl_comsoc.tex`（IEEE Communications Society journal）, 
 `bare_conf_compsoc.tex`（IEEE Computer Society conference）, 
 `bare_jrnl_compsoc.tex`（IEEE Computer Society journal） , 
 `bare_jrnl_transmag.tex`（IEEE TRANSACTIONS ON MAGNETICS）
 及`IEEEtran_HOWTO.pdf`（帮助文档）。

本篇博客为阅读*`IEEEtran_HOWTO.pdf`*后整理。

# Class Options
```
\documentclass[10pt,journal,final,]{IEEEtran}
```
A. 9pt, **10pt**, 11pt, 12pt

B. draft, draftcls, draftclsnofoot, **final**
- draft : 草稿模式，双倍行距，四面页边距均为1英寸，不显示图片，但是留空。
- draftcls : 草稿模式，跟draft一样，不过可以显示图片
- draftclsnofoot : 跟draftcls 一样，不过在脚注里不显示“DRAFT”字样，或者说，没有脚注
- final：最终模式，默认选项

C. conference, **journal**, technote, peerreview, peerreviewca
- conference ： 会议格式
- journal 和 technote : 格式与正常发表的论文一样，双栏，摘要、作者什么的都有
- peerreview : 审稿模式，单栏，title, author names and abstract 被自动隐藏（审稿需要），可用 IEEEpeerreviewmaketitle 命令（需写在maketitle后面 ）生成单独的封面（一般写在abstract之前）
- peerreviewca： 标题下会显示作者名字，其他跟peerreview一样

D. comsoc, compsoc, transmag

  *IEEE Communications Society*, *IEEE Computer Society*, *IEEE TRANSACTIONS ON MAGNETICS* 专用格式，默认不生效。

E. **letterpaper**, a4paper, cspaper

  页面类型，默认为美国通用的 US letter (8.5in 11in)， 也是IEEE通用的，也可以改为A4 (210mm 297mm)。
  cspaper为*IEEE Computer Society journals*专用。

F. **oneside**, twoside

  设置单双面打印，默认为单面。

G. onecolumn, **twocolumn**

  单栏，双栏，双栏为默认，单栏一般用于草稿

H. romanappendices

  把附录默认的编号方式由A,B,C 改为 罗马数字。

I. captionsoff

  图片、表格大写隐藏。部分期刊要求。

J. nofonttune

  减少字内连接符的使用，使整体看起来更舒适，特别适用于双栏模式。

## THE CLASSINPUT, CLASSOPTION AND CLASSINFO CONTROLS
  
  一般用不到

## THE TITLE PAGE

A. Paper Title
```
\title{Bare Demo of IEEEtran.cls\\\\ for IEEE Journals}
```
  标题一般每词首字母大写，除了非首词介词。断行使用\\\\以等分标题长度。

B. Author Names
- Journal/Technote Mode
```
\author{Michael~Shell,~\IEEEmembership{Member,~IEEE,}
        John~Doe,~\IEEEmembership{Fellow,~OSA,}
        and~Jane~Doe,~\IEEEmembership{Life~Fellow,~IEEE}% <-this % stops a space
\thanks{M. Shell was with the Department
of Electrical and Computer Engineering, Georgia Institute of Technology, Atlanta,
GA, 30332 USA e-mail: (see http://www.michaelshell.org/contact.html).}% <-this % stops a space
\thanks{J. Doe and J. Doe are with Anonymous University.}% <-this % stops a space
\thanks{Manuscript received April 19, 2005; revised August 26, 2015.}}
```
- `\author{}`内先列作者，后写`\thanks{}`，**最后一个作者**和**第一个`\thanks{}`**之间不能有空格，**各`\thanks{}`之间**也不能有空格，这里IEEE模板用了一个很机智的方法来避免不小心敲进去的空格，即在末尾加注释符%.
- `\thanks{} :` 该命令在`\author{}`命令内部使用，说明文稿的录用时间和作者通讯方式，放在footnote（脚注）处。命令内部不支持多个段落，所以如果要分段的话只能多用几次`\thanks{}`就OK了。

- Conference Mode
```
\author{\IEEEauthorblockN{Michael Shell}
\IEEEauthorblockA{School of Electrical and\\Computer Engineering\\
Georgia Institute of Technology\\
Atlanta, Georgia 30332--0250\\
Email: http://www.michaelshell.org/contact.html}
\and
\IEEEauthorblockN{Homer Simpson}
\IEEEauthorblockA{Twentieth Century Fox\\
Springfield, USA\\
Email: homer@thesimpsons.com}
\and
\IEEEauthorblockN{James Kirk\\ and Montgomery Scott}
\IEEEauthorblockA{Starfleet Academy\\
San Francisco, California 96678--2391\\
Telephone: (800) 555--1212\\
Fax: (888) 555--1212}}
```
  `\IEEEauthorblockN{}`内写姓名，`\IEEEauthorblockA{}` 内写单位信息。三个以内单位的文章倾向于使用多列模式。

关于会议模式及其他模式的更多信息此处不多涉及。

C. Running Headings（页眉）

  在页眉显示期刊名称和文章名称，初稿一般用不到

D. Publication ID Marks

  文章出版ID，初稿用不到，录用之后才会有，但是之前可以在论文中给它留空。

E. Special Paper Notices

  特殊文章备注，如受邀文章：`IEEEspecialpapernotice{(Invited Paper)}`

## ABSTRACT AND INDEX TERMS

```
\begin{abstract}
The abstract goes here.
\end{abstract}

\begin{IEEEkeywords}
IEEEtran, journal, \LaTeX, paper, template.
\end{IEEEkeywords}
```
  
  利用以上代码（模板bare_jrnl中有），摘要和关键词会在双栏排版中位于第一栏，在正文第一段之前。但有的期刊要求摘要和关键词紧挨作者横跨两栏，如 The Computer Society and TRANSACTIONS ON MAGNETICS。
  此时使用组合`\IEEEtitleabstractindextext{}`和`\IEEEdisplaynontitleabstractindextext{}` 可以根据`\document` 中要求的文本环境自动改变摘要和关键词的位置，前者使用时需将摘要命令和关键词命令放到其括号里，后者放在`\maketitle` 和 `\IEEEpeerreviewmaketitle` 命令之间。

```
\IEEEtitleabstractindextext{  
\begin{abstract}
The abstract goes here.
\end{abstract}

\begin{IEEEkeywords}
IEEEtran, journal, \LaTeX, paper, template.
\end{IEEEkeywords}}

% make the title area
\maketitle
\IEEEdisplaynontitleabstractindextext
\IEEEpeerreviewmaketitle
```
 
  以上是bare_jrnl_transmag 中的代码(注意第一行末由于编译问题去掉一个%)，如果使用`\documentclass[journal,transmag]{IEEEtran}`则摘要关键词不分栏，使用`\documentclass[journal]{IEEEtran}`则分栏。

# SECTIONS
  `\section`, `\subsection`, `\subsubsection`, `\paragraph`
  在non-compso模式下，段落层次依次以罗马数字，大写字母，阿拉伯数字、小写字母编号。
  technotes or compsoc conferences不允许层次太深，故不允许使用`\paragraph`。如果需要，可以在文章序言中使用`\setcounter{secnumdepth}{4}`恢复`\paragraph`（即section计数层次）.

A. Initial Drop Cap Letter
  首词大写，首字母下沉：`\IEEEPARstart{W}{ith}`


B. 计数器计数形式修改

  IEEE模板中Section的编号是罗马数字，要是改投其他刊物的话可能得用阿拉伯数字，所以可以在导言部分做如下修改（放在导言区宏包调用之后）:
  ```
  \renewcommand\thesection{\arabic{section}} 
  ```
  - arabic 阿拉伯数字 
  - roman 小写的罗马数字 
  - Roman 大写的罗马数字 
  - alph 小写字母 
  - Alph 大写字母

C. 计数器速查表（部分）
 
计数器名 | 用途
----------- | -------
part  | 部序号
chapter | 章序号
section | 节
subsection | 小节
subsubsection | 小小节
paragraph | 段
subparagraph | 小段
figure | 插图序号
table | 表格序号
equation | 公式序号
page | 页码计数器
footnote | 脚注序号
mpfootnote | 小页环境中脚注计数器

## CITATIONS
  To be continued...

## EQUATIONS
```
\begin{equation}
\label{eqn_example}
x = \sum\limits_{i=0}^{z} 2^{i}Q
\end{equation}
```
![equation example](/img/in-post/post-latex/latex-equation-example.png)

To be continued...

## 其他部分待更新
