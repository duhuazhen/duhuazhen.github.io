---
layout:     post
title:     (转)实现谷歌Deep Dream
date:       2017-01-04 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - tensorflow
    - 深度学习
---

本帖使用谷歌的预训练的[Inception模型](https://github.com/tensorflow/models/tree/master/inception)生成带有艺术感的图片。  
Inception模型是Google用两个星期，使用上百万张带分类的图片训练出的模型，在做[图像识别](https://tensorflow.org/tutorials/image_recognition/)时，为了节省时间，通常使用预训练的Inception模型做为训练基础。  
Deep Dream是取预训练模型的某一层（[神经网络](blog.topspeedsnail.com/archives/10377)有59层，前几层学会底层特性，像线、角，经过层层抽象，最后几层可以表示更高层次的特性），然后最大化我们提供的图像和某个层相似的特性，最后生成非常有意思的图像。

### 关于Deep Dream：

 
    [https://github.com/google/deepdream](https://github.com/google/deepdream)  
    [http://www.alanzucconi.com/2016/05/25/generating-deep-dreams/]( http://www.alanzucconi.com/2016/05/25/generating-deep-dreams/)  
    [http://ryankennedy.io/running-the-deep-dream/](http://ryankennedy.io/running-the-deep-dream/)  


[https://open_nsfw.gitlab.io](https://open_nsfw.gitlab.io)（未满18岁，请绕行；自备钛合金）
### 下载预训练的Inception模型：
``` python 
wget https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
# 解压
unzip inception5h.zip
```
代码：
```python
# -*- coding: utf-8 -*-
 
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
将测试图片保存在工作目录并命名为input_jpg。  
输入图像  

![input.jpg](https://upload-images.jianshu.io/upload_images/11573595-e0e4f474e73cc68a.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

输出图像  


![output4.jpg](https://upload-images.jianshu.io/upload_images/11573595-da6155c2f57c4190.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

原文链接：[http://blog.topspeedsnail.com/archives/10667](http://blog.topspeedsnail.com/archives/10667)
