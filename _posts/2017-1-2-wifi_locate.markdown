---
layout:     post
title:     (转)TensorFlow练习6: 基于WiFi指纹的室内定位（autoencoder）
date:       2017-01-02 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - tensorflow
    - 深度学习
---
本帖基于论文：[Low-effort place recognition with WiFi fingerprints using Deep Learning](https://arxiv.org/pdf/1611.02049v1.pdf)  

室内定位有很多种方式，利用WiFi指纹就是是其中的一种。在室内，可以通过WiFi信号强度来确定移动设备的大致位置，参看：[https://www.zhihu.com/question/20593603](室内定位有很多种方式，利用WiFi指纹就是是其中的一种。在室内，可以通过WiFi信号强度来确定移动设备的大致位置，参看：https://www.zhihu.com/question/20593603)
