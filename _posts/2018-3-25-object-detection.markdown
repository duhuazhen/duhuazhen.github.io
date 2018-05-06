---
layout:     post
title:      TensorFlow目标检测API和OpenCV实现实时目标检测和视频处理(1)
subtitle:   环境安装及实时处理
date:       2016-01-29 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - 深度学习
    - TensorFlow
---

#### 参考：
[https://github.com/tensorflow/models/tree/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection](https://github.com/tensorflow/models/tree/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection)

[Building a Real-Time Object Recognition App with Tensorflow and OpenCV](https://towardsdatascience.com/building-a-real-time-object-recognition-app-with-tensorflow-and-opencv-b7a2b4ebdc32)  
github:[https://github.com/datitran/object_detector_app](https://github.com/datitran/object_detector_app)  

[How to train your own Object Detector with TensorFlow’s Object Detector API](https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9) 
github:[https://github.com/datitran/raccoon_dataset](https://github.com/datitran/raccoon_dataset)  

[Real-time and video processing object detection using Tensorflow, OpenCV and Docker.](https://towardsdatascience.com/real-time-and-video-processing-object-detection-using-tensorflow-opencv-and-docker-2be1694726e5)  
github:[https://github.com/lbeaucourt/Object-detection](https://github.com/lbeaucourt/Object-detection)  

本文是基于ubuntu14.04+tensorflow1.2(cpu)+python3.5+opencv3.1
#### 安装anaconda
##### 1、首先在anaconda官网上下载anaconda
这里下载的是anconda linux版本（pyton3.6版本）[https://www.anaconda.com/download/#linux](https://www.anaconda.com/download/#linux)， 另外安装了python3.5的环境[https://conda.io/docs/user-guide/tasks/manage-python.html](https://conda.io/docs/user-guide/tasks/manage-python.html)，

