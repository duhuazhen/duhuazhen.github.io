---
layout:     post
title:      基于TensorFlow的鉴黄
date:       2018-05-04 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - 深度学习
    - TensorFlow
---
本文是基于ubuntu14.04+tensorflow1.6(cpu)+python3.6+anaconda  
## 简介
yahoo开源了用于检测图片是否包含不适宜工作场所（NSFW）内容的深度神经网络项目[https://github.com/yahoo/open_nsfw，GitHub](  yahoo开源了用于检测图片是否包含不适宜工作场所（NSFW）内容的深度神经网络项目https://github.com/yahoo/open_nsfw，GitHub 库中包含了网络的 Caffe 模型的代码。检测具有攻击性或成人内容的图像是研究人员进行了几十年的一个难题。随着计算机视觉技术和深度学习的发展，算法已经成熟，雅虎的这个模型能以更高的精度分辨色情图像。 由于 NSFW 界定其实是很主观的，有的人反感的东西可能其他人并不觉得如何。雅虎的这个深度神经网络只关注NSFW内容的一种类型，即色情图片。) 库中包含了网络的 Caffe 模型的代码。检测具有攻击性或成人内容的图像是研究人员进行了几十年的一个难题。随着计算机视觉技术和深度学习的发展，算法已经成熟，雅虎的这个模型能以更高的精度分辨色情图像。 由于 NSFW 界定其实是很主观的，有的人反感的东西可能其他人并不觉得如何。雅虎的这个深度神经网络只关注NSFW内容的一种类型，即色情图片。 实质上是利用了CNN的一些图像分类模型ResNet来实现二分类问题（色情与否）。  


## 安装必要的库
anaconda安装和tensorflow1.6+python3.6版本见前文
[https://duhuazhen.github.io/2016/01/29/object-detection/](https://duhuazhen.github.io/2016/01/29/object-detection/)  
激活环境（我的tensorflow1.6+python3.6环境在tensorflow环境下）  
``` python
source activate tensorflow  
``` 
* scipy
``` python
conda install scipy
```
* numpy
``` python
conda install numpy
```
* scikit-image
``` python
conda install scikit-image
```
## 下载源码
``` python
git clone https://github.com/mdietrichstein/tensorflow-open_nsfw

``` 

## 用法  
``` python
python classify_nsfw.py -m data/open_nsfw-weights.npy test.jpg  
``` 

![test4.jpg](https://upload-images.jianshu.io/upload_images/11573595-e217680eb5840168.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
``` python
 Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
Results for 'test4.jpg'
	SFW score:	0.08396992832422256
	NSFW score:	0.9160301089286804
``` 

![test1.jpg](https://upload-images.jianshu.io/upload_images/11573595-b676628c96eb378e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
``` python
Results for 'test1.jpg'
	SFW score:	0.554294228553772
	NSFW score:	0.44570574164390564
``` 

![test2.jpg](https://upload-images.jianshu.io/upload_images/11573595-c2a212986eefab3e.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
``` python
Results for 'test2.jpg'
	SFW score:	0.9743315577507019
	NSFW score:	0.025668391957879066
``` 
用了一张较为裸漏的照片，结果如下
``` python
Results for 'test6.jpg'
	SFW score:	0.004062295891344547
	NSFW score:	0.995937705039978
```
总体来说还是较为准确的。  

##### 参考 
[https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for](https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for)  
[https://github.com/yahoo/open_nsfw](https://github.com/yahoo/open_nsfw)  
[https://github.com/mdietrichstein/tensorflow-open_nsfw](https://github.com/mdietrichstein/tensorflow-open_nsfw)  
