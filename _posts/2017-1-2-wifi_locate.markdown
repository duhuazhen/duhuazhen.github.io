---
layout:     post
title:     (转)基于WiFi指纹的室内定位（autoencoder）
date:       2017-01-03 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - tensorflow
    - 深度学习
---
本帖基于论文：[Low-effort place recognition with WiFi fingerprints using Deep Learning](https://arxiv.org/pdf/1611.02049v1.pdf)  

室内定位有很多种方式，利用WiFi指纹就是是其中的一种。在室内，可以通过WiFi信号强度来确定移动设备的大致位置，参看：[https://www.zhihu.com/question/20593603](室内定位有很多种方式，利用WiFi指纹就是是其中的一种。在室内，可以通过WiFi信号强度来确定移动设备的大致位置，参看：https://www.zhihu.com/question/20593603)

## 使用WiFi指纹定位的简要流程

首先采集WiFi信号，这并不需要什么专业的设备，几台手机即可。Android手机上有很多检测WiFi的App，如Sensor Log。

把室内划分成网格块(对应位置)，站在每个块内分别使用Sensor Log检测WiFi信号，数据越多越好。如下：
```python
location1 : WiFi{"BSSID":"11:11:11:11:11:11",...."level":-33,"....} # 所在位置对应的AP,RSSI信号强度等信息
location2 : WiFi{"BSSID":"11:11:11:11:11:11",...."level":-27,"....}
location2 : WiFi{"BSSID":"22:22:22:22:22:22",...."level":-80,"....}
location3 : WiFi{"BSSID":"22:22:22:22:22:22",...."level":-54,"....}
...
```

 无线信号强度是负值，范围一般在0<->-90dbm。值越大信号越强，-50dbm强于-70dbm，  
 数据采集完成之后，对数据进行预处理，制作成WiFi指纹数据库，参考下面的UJIIndoorLoc数据集。

开发分类模型(本帖关注点)。

最后，用户上传所在位置的wifi信息，分类模型返回预测的位置。  

## TensorFlow练习: 基于WiFi指纹的室内定位

使用的数据集：[https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc](https://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc)
下载数据集： 
```python
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00310/UJIndoorLoc.zip
unzip UJIndoorLoc.zip
```
代码
```python
 
import tensorflow as tf
import numpy as np
import cv2
 
inception_model = 'tensorflow_inception_graph.pb'
 
# 加载inception模型
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
 
X = tf.placeholder(np.float32, name='input')
with tf.gfile.FastGFile(inception_model, 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
imagenet_mean = 117.0
preprocessed = tf.expand_dims(X-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':preprocessed})
 
layers = [op.name for op in graph.get_operations() if op.type=='Conv2D' and 'import/' in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+':0').get_shape()[-1]) for name in layers]
 
print('layers:', len(layers))   # 59
print('feature:', sum(feature_nums))  # 7548
 
# deep dream
def deep_dream(obj, img_noise=np.random.uniform(size=(224,224,3)) + 100.0, iter_n=10, step=1.5, octave_n=4, octave_scale=1.4):
	score = tf.reduce_mean(obj)
	gradi = tf.gradients(score, X)[0]
 
	img = img_noise
	octaves = []
 
	def tffunc(*argtypes):
		placeholders = list(map(tf.placeholder, argtypes))
		def wrap(f):
			out = f(*placeholders)
			def wrapper(*args, **kw):
				return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
			return wrapper
		return wrap
	def resize(img, size):
		img = tf.expand_dims(img, 0)
		return tf.image.resize_bilinear(img, size)[0,:,:,:]
 
	resize = tffunc(np.float32, np.int32)(resize)
	for _ in range(octave_n-1):
		hw = img.shape[:2]
		lo = resize(img, np.int32(np.float32(hw)/octave_scale))
		hi = img-resize(lo, hw)
		img = lo
		octaves.append(hi)
 
	def calc_grad_tiled(img, t_grad, tile_size=512):
		sz = tile_size
		h, w = img.shape[:2]
		sx, sy = np.random.randint(sz, size=2)
		img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
		grad = np.zeros_like(img)
		for y in range(0, max(h-sz//2, sz),sz):
			for x in range(0, max(w-sz//2, sz),sz):
				sub = img_shift[y:y+sz,x:x+sz]
				g = sess.run(t_grad, {X:sub})
				grad[y:y+sz,x:x+sz] = g
		return np.roll(np.roll(grad, -sx, 1), -sy, 0)   
 
	for octave in range(octave_n):
		if octave>0:
			hi = octaves[-octave]
			img = resize(img, hi.shape[:2])+hi
		for _ in range(iter_n):
			g = calc_grad_tiled(img, gradi)
			img += g*(step / (np.abs(g).mean()+1e-7))
 
		# 保存图像
		output_file = 'output' + str(octave+1) + '.jpg'
		cv2.imwrite(output_file, img)
		print(output_file)
 
# 加载输入图像
input_img = cv2.imread('input.jpg')
input_img = np.float32(input_img)
 
# 选择层
layer = 'mixed4c'
 
deep_dream(tf.square(graph.get_tensor_by_name("import/%s:0"%layer)), input_img)
```
训练过程：
```python
Epoch:  0  Loss:  0.813012358957
Epoch:  1  Loss:  0.745993956116
Epoch:  2  Loss:  0.730452445426
Epoch:  3  Loss:  0.722375573073
Epoch:  4  Loss:  0.71705802879
```

转自：[ttp://blog.topspeedsnail.com/archives/10468](http://blog.topspeedsnail.com/archives/10468)
