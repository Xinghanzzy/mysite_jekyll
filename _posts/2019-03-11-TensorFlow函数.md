---
layout:     post   				    # 使用的布局（不需要改）
title:      TensorFlow函数 				# 标题 
subtitle:   TensorFlow常用函数 #副标题
date:       2019-03-11 				# 时间
author:     XH 						# 作者
header-img: img/post-bg-2015.jpg 	#这篇文章标题背景图片
catalog: true 						# 是否归档
tags:								#标签
    - NMT
    - TensorFlow
    - NLP

---


> [我的的博客](https://xinghanzzy.github.io/)



# TensorFlow函数

## tf.tile

```python 
# key_masks：(N, T_k) N:batch_size T:maxlen
tf.tile(key_masks, [num_heads, 1]) # (h*N, T_k) 
```

在input的每一维 复制对应的次数

```python
tile([x, y], [a, b])
# x 复制 a 次
# y 复制 b 次
```



## tf.reduce_sum

```python 
tf.reduce_sum(keys, axis=-1)
# input按照第n维加和 
# 感觉就是那一维度没了
# [N, T, num_units] -> [N, T]
def reduce_sum(input_tensor,
               axis=None,
               keepdims=None,
               name=None,
               reduction_indices=None,
               keep_dims=None):
  """Computes the sum of elements across dimensions of a tensor.

  Reduces `input_tensor` along the dimensions given in `axis`.
  Unless `keepdims` is true, the rank of the tensor is reduced by 1 for each
  entry in `axis`. If `keepdims` is true, the reduced dimensions
  are retained with length 1.

  If `axis` has no entries, all dimensions are reduced, and a
  tensor with a single element is returned.
"""
```

