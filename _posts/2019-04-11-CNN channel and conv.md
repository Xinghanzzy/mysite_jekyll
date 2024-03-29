---
layout:     post
title:      CNN 多通道卷积核 channel and conv
subtitle:   多通道卷积计算理解
date:       2019-04-11
author:     XH
header-img: img/ind123123ex.jpg
catalog: true
tags:
    - 深度学习
    - CNN
---


>自己看的记录笔记
>
>写的很差 勿看


>参考文献
>
>好多博客都是类似的内容 希望没有写错
>
><https://blog.csdn.net/jacke121/article/details/80188821> 
>
><https://blog.csdn.net/haoji007/article/details/81981846> 

# CNN 多通道卷积计算理解

对于单通道图像，若利用10个卷积核进行卷积计算，可以得到10个特征图；若输入为多通道图像，则输出特征图的个数依然是卷积核的个数（10个）。 



以图片来举例：假设图片的宽度为width:W，高度为height:H，图片的通道数为D，一般目前都用RGB三通道D=3，为了通用性，通道数用D表示； 

卷积核：卷积核大小为K\*K，由于处理的图片是D通道的，因此卷积核其实也就是K\*K\*D大小的，因此，对于RGB三通道图像，在指定kernel_size的前提下，真正的卷积核大小是kernel_size*kernel_size\*3。 

**对于D通道图像的各通道而言，是在每个通道上分别执行二维卷积，然后将D个通道加起来，得到该位置的二维卷积输出**

对于RGB三通道图像而言，就是在R，G，B三个通道上分别使用对应的每个通道上的kernel_size\*kernel_size大小的核去卷积每个通道上的W*H的图像，然后将三个通道卷积得到的输出相加，得到二维卷积输出结果。因此，若有M个卷积核，可得到M个二维卷积输出结果，在有padding的情况下，能保持输出图片大小和原来的一样，因此是output(W,H,M)。



下面的图动态形象地展示了三通道图像卷积层的计算过程：

下图是7*7*3的图像，3通道，有2个3*3*3的卷积核，也称3*2=6个卷积核

![å¾çæè¿°](https://segmentfault.com/img/bVW1tf?w=860&h=690) 

有教程为7*7*M通道的图片，输出通道是n，则卷积核共m*n个卷积核。

原版动图地址：

http://cs231n.github.io/convolutional-networks/



```
四个通道上的卷积操作:
	有两个卷积核，生成两个通道。
	其中需要注意的是，四个通道上每个通道对应一个2*2的卷积核
	这4个2*2的卷积核上的参数是不一样的，之所以说它是1个卷积核，是因为把它看成了一个4*2*2的卷积核，4代表一开始卷积的通道数，2*2是卷积核的尺寸，实际卷积的时候其实就是4个2*2的卷积核（这四个22的卷积核的参数是不同的）分别去卷积对应的4个通道，然后相加，再加上偏置b，注意b对于这四通道而言是共享的，所以b的个数是和最终的featuremap的个数相同的，然后再取激活函数
    
	输出层的卷积核个数为 feature map 的个数。也就是说卷积核的个数=最终的featuremap的个数，卷积核的大小=开始进行卷积的通道数每个通道上进行卷积的二维卷积核的尺寸（此处就是4（2*2）），b（偏置）的个数=卷积核的个数=featuremap的个数。

4个通道卷积得到2个通道的过程中，参数的数目为4×（2×2）×2+2个，其中4表示4个通道，第一个2*2表示卷积核的大小，第三个2表示生成的featuremap个数，也就是得到的2通道的feature map，也就是生成的通道数，最后的2代表偏置b的个数。

```

