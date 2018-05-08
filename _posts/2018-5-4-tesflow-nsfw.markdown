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
yahoo开源了用于检测图片是否包含不适宜工作场所（NSFW）内容的深度神经网络项目[https://github.com/yahoo/open_nsfw，GitHub](  yahoo开源了用于检测图片是否包含不适宜工作场所（NSFW）内容的深度神经网络项目https://github.com/yahoo/open_nsfw，GitHub 库中包含了网络的 Caffe 模型的代码。检测具有攻击性或成人内容的图像是研究人员进行了几十年的一个难题。随着计算机视觉技术和深度学习的发展，算法已经成熟，雅虎的这个模型能以更高的精度分辨色情图像。 由于 NSFW 界定其实是很主观的，有的人反感的东西可能其他人并不觉得如何。雅虎的这个深度神经网络只关注NSFW内容的一种类型，即色情图片。) 库中包含了网络的 Caffe 模型的代码。检测具有攻击性或成人内容的图像是研究人员进行了几十年的一个难题。随着计算机视觉技术和深度学习的发展，算法已经成熟，雅虎的这个模型能以更高的精度分辨色情图像。 由于 NSFW 界定其实是很主观的，有的人反感的东西可能其他人并不觉得如何。雅虎的这个深度神经网络只关注NSFW内容的一种类型，即色情图片。   


## 安装必要的库
##### 参考 
[https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for](https://yahooeng.tumblr.com/post/151148689421/open-sourcing-a-deep-learning-solution-for)  
[https://github.com/yahoo/open_nsfw](https://github.com/yahoo/open_nsfw)  
[https://github.com/mdietrichstein/tensorflow-open_nsfw](https://github.com/mdietrichstein/tensorflow-open_nsfw)  
