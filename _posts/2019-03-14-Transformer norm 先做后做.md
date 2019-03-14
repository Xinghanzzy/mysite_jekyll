---
layout:     post
title:      Transformer norm 先做后做
subtitle:   Transformer layer normalization 先做后做
date:       2019-03-14
author:     XH
header-img: img/post-bg-debug.png
catalog: true
tags:
    - Transformer
    - NMT
---


>并不适合阅读的个人文档。

# Normalization  先做 和 后做 

![](https://ws1.sinaimg.cn/large/4ac7f217ly1g12hedto9jj20d208e3yt.jpg)

- 后做是传统做法

  后做是 input x ,  x residual 到后面, x 进行function(multi-head attention),  f(x) 进行dropout，df(x) + x  （residual）, layer normalization(df(x) + x  )

  **x->LN(df(x) + x)**

- 先做是当前性能好的做法

  后做是 input x ,  x residual 到后面, x 进行LN，然后进行function(multi-head attention), 最后进行dropout在和残差相加，dfLN(x) + x  （residual）

  **x->x + (df(LN(x))**

