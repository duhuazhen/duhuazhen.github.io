---
layout:     post
title:     (转)图像分类器 – retrain谷歌Inception模型
date:       2017-01-21 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - tensorflow
    - 深度学习
---
环境：ubuntu14.04+python3.5+anaconda+opencv3.1   

本帖就基于Inception模型retrain一个图像分类器。  

图像分类器应用广泛，连农业都在使用，如判断黄瓜种类。  

本帖使用的训练数据是[《TensorFlow练习9: 生成妹子图（PixelCNN）》](blog.topspeedsnail.com/archives/10660)一文中使用的妹子图，最后训练出的分类器可以判断图片是不是妹子图。

首先下载tensorflow源代码： 

```python
git clone https://github.com/tensorflow/models
```

在retrain自己的图像分类器之前，我们先来测试一下Google的Inception模型：  

```python
cd models/tutorials/image/imagenet/
python classify_image.py --image_file bigcat.jpg  
# 自动下载100多M的模型文件
# 参数的解释, 查看源码文件
```
但是由于网络原因可能下载失败，我们可以直接下载好模型[http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz)  

然后将其中的下载函数注释掉。并修改模型目录，默认目录在 /tmp/imagenet，也可以通过添加参数的方式修改,具体看源文件参数设置。

```python
#maybe_download_and_extract()
```
然后执行
```python
python classify_image.py --image_file bigcat.jpg  
```
结果

![bigcat.jpg](https://upload-images.jianshu.io/upload_images/11573595-80e3df4ecd2f89a3.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  


```python
lion, king of beasts, Panthera leo (score = 0.97022)
cheetah, chetah, Acinonyx jubatus (score = 0.00060)
leopard, Panthera pardus (score = 0.00048)
hyena, hyaena (score = 0.00043)
zebra (score = 0.00033)

```
[https://github.com/tensorflow/hub/tree/master/examples/image_retraining](https://github.com/tensorflow/hub/tree/master/examples/image_retraining)  

[https://github.com/tensorflow/tensorflow/tree/cda36b817e9998906da37ec87c525f1b278c71a7/tensorflow/examples/image_retraining](https://github.com/tensorflow/tensorflow/tree/cda36b817e9998906da37ec87c525f1b278c71a7/tensorflow/examples/image_retraining)  

使用examples中的image_retraining。 

训练

```python
python tensorflow/tensorflow/examples/image_retraining/retrain.py --bottleneck_dir bottleneck --how_many_training_steps 4000 --model_dir model --output_graph output_graph.pb --output_labels output_labels.txt --image_dir girl_types/
```
参数解释参考retrain.py源文件。

大概训练了半个小时：
```python
Final test accuracy =99.8%
```
生成的模型文件和labels文件：  

![屏幕快照-2016-11-29-下午12.49.32.png](https://upload-images.jianshu.io/upload_images/11573595-62ffdfd1c34ceaa4.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)   


使用训练好的模型：


```python
import tensorflow as tf
import sys
 
# 命令行参数，传入要判断的图片路径
image_file = sys.argv[1]
#print(image_file)
 
# 读取图像
image = tf.gfile.FastGFile(image_file, 'rb').read()
 
# 加载图像分类标签
labels = []
for label in tf.gfile.GFile("output_labels.txt"):
	labels.append(label.rstrip())
 
# 加载Graph
with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
	graph_def = tf.GraphDef()
	graph_def.ParseFromString(f.read())
	tf.import_graph_def(graph_def, name='')
 
with tf.Session() as sess:
	softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
	predict = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
 
	# 根据分类概率进行排序
	top = predict[0].argsort()[-len(predict[0]):][::-1]
	for index in top:
		human_string = labels[index]
		score = predict[0][index]
		print(human_string, score)
```

执行结果：

```python
big ass 0.999341
```


参考：
[https://www.tensorflow.org/versions/r0.11/how_tos/image_retraining/index.html](https://www.tensorflow.org/versions/r0.11/how_tos/image_retraining/index.html)   
[http://blog.topspeedsnail.com/archives/10451](http://blog.topspeedsnail.com/archives/10451)
[How Convolutional Neural Networks work](https://www.youtube.com/watch?v=FmpDIaiMIeA)


原文链接：[http://blog.topspeedsnail.com/archives/10685](http://blog.topspeedsnail.com/archives/10685)








