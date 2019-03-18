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
> [深度学习中 Batch Normalization为什么效果好？-知乎](https://www.zhihu.com/question/38102762/answer/85238569?utm_source=wechat_session&utm_medium=social&utm_oi=629421655403925504/ )
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



# Normalization

## Batch Normalization

​	顾名思义，batch normalization嘛，就是“批规范化”咯。Google在ICML文中描述的非常清晰，即在每次SGD时，通过mini-batch来对相应的activation做规范化操作，**使得结果（输出信号各个维度）的均值为0，方差为1.** 

​	最后的“scale and shift”操作则是为了让因训练所需而“刻意”加入的BN能够有可能还原最初的输入（即当![\gamma^{(k)}=\sqrt{Var[x^{(k)}]}, \beta^{(k)}=E[x^{(k)}]](https://www.zhihu.com/equation?tex=%5Cgamma%5E%7B%28k%29%7D%3D%5Csqrt%7BVar%5Bx%5E%7B%28k%29%7D%5D%7D%2C+%5Cbeta%5E%7B%28k%29%7D%3DE%5Bx%5E%7B%28k%29%7D%5D)   $\gamma^{(k)}=\sqrt{Var[x^{(k)}]}, \beta^{(k)}=E[x^{(k)}]$ 时候，相当于未做normalization），从而保证整个network的capacity（容量）。（有关capacity的解释：实际上BN可以看作是在原模型上加入的“新操作”，这个新操作很大可能会改变某层原来的输入。当然也可能不改变，不改变的时候就是“还原原来输入”。如此一来，既可以改变同时也可以保持原输入，那么模型的容纳能力（capacity）就提升了。）




ax^{2} + by^{2} + c = 0



$$
\gamma^{(k)}=\sqrt{Var[x^{(k)}]}, \beta^{(k)}=E[x^{(k)}]
$$





