---
layout:     post
title:     (转)生成音乐
date:       2017-01-08 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - tensorflow
    - 深度学习
---
环境：ubuntu14.04+python3.5+anaconda+midi

我在GitHub看到了一个使用RNN生成经典音乐的项目：[biaxial-rnn-music-composition](https://github.com/hexahedria/biaxial-rnn-music-composition)，它是基于Theano的。本帖改为使用TensorFlow生成音乐，代码逻辑在很大程度上基于前者。  


    https://deeplearning4j.org/restrictedboltzmannmachine.html  
    https://magenta.tensorflow.org/2016/06/10/recurrent-neural-network-generation-tutorial/  
    https://deepmind.com/blog/wavenet-generative-model-raw-audio  
    Google的项目Magenta：生成音乐、绘画或视频  
    http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/  
    TensorFlow练习7: 基于RNN生成古诗词  

数据集：首先准备一些MIDI音乐，可以去freemidi.org下载。  

另一个关于音乐的数据集 MusicNet  

我下载了50多个MIDI文件（貌似有点少）。  

有了MIDI音乐，我们还需要一个可以操作MIDI的Python库：[python-midi](https://github.com/vishnubob/python-midi)。

安装python-midi：

对于python2  
```python
git clone https://github.com/vishnubob/python-midi
cd python-midi
# $ git checkout feature/python3   # 如果使用Python3，checkout对应分支
python setup.py install

```

对于python3  
```python
git clone https://github.com/louisabraham/python3-midi
cd python-midi
# $ git checkout feature/python3   # 如果使用Python3，checkout对应分支
python setup.py install

```


