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
conda 常用命令 [ttps://www.jianshu.com/p/2f3be7781451](https://www.jianshu.com/p/2f3be7781451)
[ttps://blog.csdn.net/menc15/article/details/71477949](https://blog.csdn.net/menc15/article/details/71477949)
##### 1、首先在anaconda官网上下载anaconda
这里下载的是anconda linux版本（pyton3.6版本）[https://www.anaconda.com/download/#linux](https://www.anaconda.com/download/#linux)， 另外安装了python3.5的环境[https://conda.io/docs/user-guide/tasks/manage-python.html](https://conda.io/docs/user-guide/tasks/manage-python.html)，
``` python
conda create -n py35 python=3.5 anaconda
```
上述命令中py35为环境名字，python=3.5为我们要创造的python版本，anaconda包含了所需要的所有python包。

![screenshot-conda.io-2018-05-06-14-07-13.png](https://upload-images.jianshu.io/upload_images/11573595-49645abd962c203c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
##### 2 激活环境
通过以下命令来激活我们所需要的环境
``` python
source activate py35
```
##### 3 安装及运行目标检测库
通过下面链接中的[https://github.com/tensorflow/models/tree/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection](https://github.com/tensorflow/models/tree/477ed41e7e4e8a8443bc633846eb01e2182dc68a/object_detection)命令
``` python
conda env create -f environment.yml
```
来创造我们需要的环境，可是会出现如下错误
``` python
(py35) hehe@hehe-OptiPlex-5040:~/anaconda3/envs/object_detector_app-master$ conda env create -f environment.yml
Solving environment: failed

ResolvePackageNotFound: 
  - opencv3==3.0.0=py35_0
  - tbb==4.3_20141023=0
```  
可能是源的关系，找不到一些相应的库，于是我们删除找不到的库，具体参考我fork的自己的仓库，由于找不到相应的opencv库，我们通过手动安装相应的库  
https://github.com/duhuazhen/object_detector_app
``` python
conda env create -f environment.yml
```
``` python
 conda install -c https://conda.anaconda.org/menpo opencv3  
 ```  
 然后通过运行
``` python
python object_detection_app.py Optional arguments (default value):

    Device index of the camera --source=0
    Width of the frames in the video stream --width=480
    Height of the frames in the video stream --height=360
    Number of workers --num-workers=2
    Size of the queue --queue-size=5

```
来启动物体识别程序。整体来说效果还是可以的  
![image.png](https://upload-images.jianshu.io/upload_images/11573595-432d0147c1bbedea.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
下一步我们将来训练自己的模型。
