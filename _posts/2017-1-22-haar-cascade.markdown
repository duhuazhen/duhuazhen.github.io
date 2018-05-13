---
layout:     post
title:     (转)OpenCV之使用Haar Cascade进行对象检测
date:       2017-01-03 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - tensorflow
    - 深度学习
---
环境：ubuntu14.04+python3.5+anaconda+opencv3.1  
Haar Cascade常用来做人脸检测，其实它可以检测任何对象。OpenCV项目源码中有很多训练好的Haar分类器。  
```python
hehe@hehe-OptiPlex-5040:~/opencv-3.1.0/data/haarcascades$ ls
haarcascade_eye_tree_eyeglasses.xml
haarcascade_eye.xml
haarcascade_frontalcatface_extended.xml
haarcascade_frontalcatface.xml
haarcascade_frontalface_alt2.xml
haarcascade_frontalface_alt_tree.xml
haarcascade_frontalface_alt.xml
haarcascade_frontalface_default.xml
haarcascade_fullbody.xml
haarcascade_lefteye_2splits.xml
haarcascade_licence_plate_rus_16stages.xml
haarcascade_lowerbody.xml
haarcascade_profileface.xml
haarcascade_righteye_2splits.xml
haarcascade_russian_plate_number.xml
haarcascade_smile.xml
haarcascade_upperbody.xml

```
OpenCV自带的haarcascade文件  
本帖开始先了解怎么使用这些现成的分类器，最后再训练自己的Haar分类器。如果你要检测什么物体，先Google，也许已经有训练好的Haar分类器了（像汽车、猫，狗之类的）。

使用OpenCV自带的Haar分类器检测脸和眼睛，代码：  
```python
import cv2
import sys

img = cv2.imread(sys.argv[1])

# 加载分类器
face_haar = cv2.CascadeClassifier("/home/hehe/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
eye_haar = cv2.CascadeClassifier("/home/hehe/opencv-3.1.0/data/haarcascades/haarcascade_eye.xml")
# 把图像转为黑白图像
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 检测图像中的所有脸
faces = face_haar.detectMultiScale(gray_img, 1.3, 5)
for face_x,face_y,face_w,face_h in faces:
	cv2.rectangle(img, (face_x, face_y), (face_x+face_w, face_y+face_h), (0,255,0), 2)
        # 眼长在脸上
	roi_gray_img = gray_img[face_y:face_y+face_h, face_x:face_x+face_w]
	roi_img = img[face_y:face_y+face_h, face_x:face_x+face_w]
	eyes = eye_haar.detectMultiScale(roi_gray_img, 1.3, 5)
	for eye_x,eye_y,eye_w,eye_h in eyes:
		cv2.rectangle(roi_img, (eye_x,eye_y), (eye_x+eye_w, eye_y+eye_h), (255,0,0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)

cv2.destroyAllWindows()
```
```python
python face_detect.py lena.jpg
```
![屏幕快照-2016-11-20-下午5.37.48.png](https://upload-images.jianshu.io/upload_images/11573595-ad89578540af393d.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  


使用摄像头做为输入，实时检测：  

```python
import cv2
 
face_haar = cv2.CascadeClassifier("data/haarcascades/haarcascade_frontalface_default.xml")
eye_haar = cv2.CascadeClassifier("data/haarcascades/haarcascade_eye.xml")
 
cam = cv2.VideoCapture(0)
 
while True:
	_, img = cam.read()
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
	faces = face_haar.detectMultiScale(gray_img, 1.3, 5)
	for face_x,face_y,face_w,face_h in faces:
		cv2.rectangle(img, (face_x, face_y), (face_x+face_w, face_y+face_h), (0,255,0), 2)
 
		roi_gray_img = gray_img[face_y:face_y+face_h, face_x:face_x+face_w]
		roi_img = img[face_y:face_y+face_h, face_x:face_x+face_w]
		eyes = eye_haar.detectMultiScale(roi_gray_img, 1.3, 5)
		for eye_x,eye_y,eye_w,eye_h in eyes:
			cv2.rectangle(roi_img, (eye_x,eye_y), (eye_x+eye_w, eye_y+eye_h), (255,0,0), 2)
 
	cv2.imshow('img', img)
	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break
 
cam.release()
cv2.destroyAllWindows()
```
上面我们使用的是训练好的分类器文件，如果你要检测的物体没有现成的Haar分类器，我们只能自己训练了，其中最费事的部分就是制作训练样本。  

#### 训练Haar分类器的主要步骤：

1. 搜集制作成千上万张”消极”图像，什么图片都行，但是确保要检测的对象不在图像中  
2. 搜集制作成千上万张”积极”图像，确保这些图像中包含要检测的对象  
3. [http://image-net.org](http://image-net.org)是不错的图像资源站  
4. 创建”积极”向量文件  
5. 使用OpenCV训练Haar分类器  

为了简单，我使用一张图片制作”积极”图像：  

![mouse.jpg](https://upload-images.jianshu.io/upload_images/11573595-6e196bca92ee235d.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)   

做一个能检测我鼠标的Haar分类器  
这是我的鼠标，我就使用这一张图片制作”积极”图像，没错，最后训练出来的Haar分类器只能识别这个特定鼠标。如果你想要识别各种各样的鼠标，你需要搜集整理包含各种鼠标的图片（标记出图片中鼠标所在位置-ROI），即使有工具的帮助，这个工作也是相当痛苦的。  

#### 下载”消极”图像

找点和鼠标不想干的图片：image-net  


![屏幕快照-2016-11-21-下午12.01.27.png](https://upload-images.jianshu.io/upload_images/11573595-07e4e9ec4c7c57ee.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  
Downloads中包含图像地址：  
![屏幕快照-2016-11-21-下午12.04.57.png](https://upload-images.jianshu.io/upload_images/11573595-97dd972afd147df8.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

写一个简单的Python脚本下载图片：
```python
# Python3
 
import urllib.request
import cv2
import os
 
# 创建图片保存目录
if not os.path.exists('neg'):
    os.makedirs('neg')
 
neg_img_url = ['http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00523513', 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152']
 
urls = ''
for img_url in neg_img_url:
	urls += urllib.request.urlopen(img_url).read().decode()
 
img_index = 1
for url in urls.split('\n'):
    try:
    	print(url)
    	urllib.request.urlretrieve(url, 'neg/'+str(img_index)+'.jpg')
        # 把图片转为灰度图片
    	gray_img = cv2.imread('neg/'+str(img_index)+'.jpg', cv2.IMREAD_GRAYSCALE)
        # 更改图像大小
    	image = cv2.resize(gray_img, (150, 150))
        # 保存图片
    	cv2.imwrite('neg/'+str(img_index)+'.jpg', image)
    	img_index += 1
    except Exception as e:
        print(e)
 
# 判断两张图片是否完全一样
def is_same_image(img_file1, img_file2):
    img1 = cv2.imread(img_file1)
    img2 = cv2.imread(img_file2)
    if img1.shape == img2.shape and not (np.bitwise_xor(img1, img2).any()):
        return True
    else:
        return False
 
# 去除重复图片
"""
file_list = os.listdir('neg')
try:
	for img1 in file_list:
		for img2 in file_list:
			if img1 != img2:
				if is_same_image('neg/'+img1, 'neg/'+img2) is True:
					print(img1, img2)
					os.remove('neg/'+img1)
		file_list.remove(img1)
except Exception as e:
	print(e)
"""
```


    很多url被墙，你可能需要使用代理。(参考：[使用Tor的匿名Python爬虫](http://blog.topspeedsnail.com/archives/7258))
    下载的文件很多，为了提速，你可以把上面代码改为多线程。
![Screenshot-from-2016-11-21-13-05-56.png](https://upload-images.jianshu.io/upload_images/11573595-3894009a724ced23.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  

创建消极图片列表：  
```python
import os
import numpy as np
 
with open('neg.txt', 'w') as f:
    for img in os.listdir('neg'):
        line = 'neg/'+img+'\n'
        f.write(line)
```
创建的neg.txt内容如下：  


![屏幕快照-2016-11-21-下午3.23.01.png](https://upload-images.jianshu.io/upload_images/11573595-03d80eeadbb3f55a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  


我下载了2000+图片  

#### 制作”积极”图像

我使用OpenCV提供的opencv_createsamples命令创建pos.txt文件。它会把要识别的图片嵌入到消极图像中，允许我们快速创建”积极”图像：
```python
opencv_createsamples -img mouse.jpg -bg neg.txt -info pos.txt -maxxangle 0.5 -maxyangle -0.5 -maxzangle 0.5 -num 2000

````
生成的pos.txt文件：  

![屏幕快照-2016-11-22-上午9.16.34.png](https://upload-images.jianshu.io/upload_images/11573595-5aa427011a451090.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  


第一列代表“积极”图像路径；后面数字代表图像中有几个要识别对象和对象所在位置

你可以看看生成的“积极”图像，这些图像中嵌入了要识别的鼠标。

上面的”积极图像”是自动生成的，这要是手工制作，那工作量可想而知。  

#### 创建向量文件  
不管你用什么方法制作”积极”图像，都需要把它转换为向量格式：  
```python
opencv_createsamples -info pos.txt -num 2000 -w 20 -h 30 -vec pos.vec
```
#### 开始训练
```python
mkdir data
opencv_traincascade -data data -vec pos.vec -bg neg.txt -numPos 1800 -numNeg 900 -numStages 15 -w 20 -h 30  # pos一般是neg的1倍
```

大概需要几个小时，我电脑不给力，上面参数设置的都比较小。  
![屏幕快照-2016-11-22-上午10.19.38.png](https://upload-images.jianshu.io/upload_images/11573595-7fe21d49bdd51582.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

训练完成之后生成的haar分类器(cascade.xml)保存在data目录。  

#### 测试生成的haar分类器

```python
import cv2
 
mouse_haar = cv2.CascadeClassifier("data/cascade.xml")
 
cam = cv2.VideoCapture(0)
 
while True:
	_, img = cam.read()
	gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #http://docs.opencv.org/2.4/modules/objdetect/doc/cascade_classification.html
	mouse = mouse_haar.detectMultiScale(gray_img, 1.2, 3) # 调整参数
 
	for mouse_x,mouse_y,mouse_w,mouse_h in mouse:
		cv2.rectangle(img, (mouse_x, mouse_y), (mouse_x+mouse_w, mouse_y+mouse_h), (0,255,0), 2)
 
	cv2.imshow('img', img)
	key = cv2.waitKey(30) & 0xff
	if key == 27:
		break
 
cam.release()
cv2.destroyAllWindows()
```
结果：  
![Screenshot-from-2016-11-22-11-20-18.png](https://upload-images.jianshu.io/upload_images/11573595-0c5d3a351797b47a.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  


转自：[http://blog.topspeedsnail.com/archives/10511](http://blog.topspeedsnail.com/archives/10511)
