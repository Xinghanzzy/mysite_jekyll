---
layout:     post   				    # 使用的布局（不需要改）
title:      Softmax and Normalization # 标题 
subtitle:   Softmax and Normalization理解 #副标题
date:       2019-03-11 				# 时间
author:     XH 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - NMT
    - TensorFlow
    - NLP

---


> [三分钟带你对 Softmax 划重点](https://blog.csdn.net/red_stone1/article/details/80687921/)
>
> [我的的博客](https://xinghanzzy.github.io/)



# Softmax

在数学，尤其是概率论和相关领域中，Softmax函数，或称归一化指数函数，是逻辑函数的一种推广。

Softmax函数实际上是有限项离散概率分布的梯度对数归一化。

它能将一个含任意实数的K维向量  “压缩”到另一个K维实向量  中，使得每一个元素的范围都在$(0, 1)$之间，并且所有元素的和为$1$。
$$
S_i=\frac {e^{V_i}} {\sum_{i}^{c} {e^{V_i}}}
$$

在实际应用中遇到提前$padding$时候将$V_i$设置为$-2^{32}+1$这种情况

为了是经过$Softmax$之后可以把值变为0