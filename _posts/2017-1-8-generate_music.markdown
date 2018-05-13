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
[https://deeplearning4j.org/restrictedboltzmannmachine.html](https://deeplearning4j.org/restrictedboltzmannmachine.html)     
[https://magenta.tensorflow.org/2016/06/10/recurrent-neural-network-generation-tutorial/](https://magenta.tensorflow.org/2016/06/10/recurrent-neural-network-generation-tutorial/)   
[https://deepmind.com/blog/wavenet-generative-model-raw-audio](https://deepmind.com/blog/wavenet-generative-model-raw-audio)    
Google的项目[Magenta](https://github.com/tensorflow/magenta)：生成音乐、绘画或视频  
[http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)    

数据集：首先准备一些MIDI音乐，可以去freemidi.org下载。  

另一个关于音乐的数据集 [MusicNet](http://www.hexahedria.com/2015/08/03/composing-music-with-recurrent-neural-networks/)   

我下载了50多个MIDI文件（貌似有点少）。   
![屏幕快照-2016-11-25-下午1.51.59.png](https://upload-images.jianshu.io/upload_images/11573595-16e3871b2c5f2b1f.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)  

有了MIDI音乐，我们还需要一个可以操作MIDI的Python库：[python-midi](https://github.com/vishnubob/python-midi)。

#### 安装python-midi：  

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
tensorflow while_loop用法：  
```python
# 代码取自Stack OverFlow
import tensorflow as tf
import numpy as np
 
def body(x):
    a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
    b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
    c = a + b
    return tf.nn.relu(x + c)
 
def condition(x):
    return tf.reduce_sum(x) < 100
 
x = tf.Variable(tf.constant(0, shape=[2, 2]))
 
with tf.Session():
    tf.initialize_all_variables().run()
    result = tf.while_loop(condition, body, [x])
    print(result.eval())
```
TensorFlow生成mid音乐完整代码（已修改）：
```python
import tensorflow as tf
import midi
import numpy as np
import os
import io as stringIOModule
lower_bound = 24
upper_bound = 102
span = upper_bound - lower_bound

# midi文件转Note(音符)
def midiToNoteStateMatrix(midi_file_path, squash=True, span=span):
	pattern = midi.read_midifile(midi_file_path)

	time_left = []
	for track in pattern:
		time_left.append(track[0].tick)
	
	posns = [0 for track in pattern]

	statematrix = []
	time = 0

	state = [[0,0] for x in range(span)]
	statematrix.append(state)
	condition = True
	while condition:
		if time % (pattern.resolution / 4) == (pattern.resolution / 8):
			oldstate = state
			state = [[oldstate[x][0],0] for x in range(span)]
			statematrix.append(state)
		for i in range(len(time_left)):
			if not condition:
				break
			while time_left[i] == 0:
				track = pattern[i]
				pos = posns[i]

				evt = track[pos]
				if isinstance(evt, midi.NoteEvent):
					if (evt.pitch < lower_bound) or (evt.pitch >= upper_bound):
						pass
					else:
						if isinstance(evt, midi.NoteOffEvent) or evt.velocity == 0:
							state[evt.pitch-lower_bound] = [0, 0]
						else:
							state[evt.pitch-lower_bound] = [1, 1]
				elif isinstance(evt, midi.TimeSignatureEvent):
					if evt.numerator not in (2, 4):
						out =  statematrix
						condition = False
						break
				try:
					time_left[i] = track[pos + 1].tick
					posns[i] += 1
				except IndexError:
					time_left[i] = None

			if time_left[i] is not None:
				time_left[i] -= 1

		if all(t is None for t in time_left):
			break

		time += 1

	S = np.array(statematrix)
	statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
	statematrix = np.asarray(statematrix).tolist()
	return statematrix

# Note转midi文件
def noteStateMatrixToMidi(statematrix, filename="output_file", span=span):
    statematrix = np.array(statematrix)
    if not len(statematrix.shape) == 3:
        statematrix = np.dstack((statematrix[:, :span], statematrix[:, span:]))
    statematrix = np.asarray(statematrix)
    pattern = midi.Pattern()
    track = midi.Track()
    pattern.append(track)
    
    span = upper_bound-lower_bound
    tickscale = 55
    
    lastcmdtime = 0
    prevstate = [[0,0] for x in range(span)]
    for time, state in enumerate(statematrix + [prevstate[:]]):  
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(midi.NoteOffEvent(tick=(time-lastcmdtime)*tickscale, pitch=note+lower_bound))
            lastcmdtime = time
        for note in onNotes:
            track.append(midi.NoteOnEvent(tick=(time-lastcmdtime)*tickscale, velocity=40, pitch=note+lower_bound))
            lastcmdtime = time
            
        prevstate = state
    
    eot = midi.EndOfTrackEvent(tick=1)
    track.append(eot)

    midi.write_midifile("{}.mid".format(filename), pattern)

# 读取midi数据
def get_songs(midi_path):
	files = os.listdir(midi_path)
	songs = []
	for f in files:
		f = midi_path+'/'+f
		print('加载:', f)
		try:
			song = np.array(midiToNoteStateMatrix(f))
			if np.array(song).shape[0] > 64:
				songs.append(song)
		except Exception as e:
			print('数据无效: ', e)
	print("读取的有效midi文件个数: ", len(songs))
	return songs

# midi目录中包含了下载的midi文件
songs = get_songs('midi')

note_range = upper_bound - lower_bound
# 音乐长度
n_timesteps = 128
n_input = 2 * note_range * n_timesteps
n_hidden = 64

X = tf.placeholder(tf.float32, [None, n_input])
W = None
bh = None
bv = None

def sample(probs):
	return tf.floor(probs + tf.random_uniform(tf.shape(probs), 0, 1))

def gibbs_sample(k):
	def body(count, k, xk):
		hk = sample(tf.sigmoid(tf.matmul(xk, W) + bh))
		xk = sample(tf.sigmoid(tf.matmul(hk, tf.transpose(W)) + bv))
		return count+1, k, xk

	count = tf.constant(0)
	def condition(count,  k, xk):
		return count < k
	[_, _, x_sample] = tf.while_loop(condition, body, [count, tf.constant(k), X])

	x_sample = tf.stop_gradient(x_sample) 
	return x_sample

#定义神经网络
def neural_network():
	global W
	W  = tf.Variable(tf.random_normal([n_input, n_hidden], 0.01))
	global bh
	bh = tf.Variable(tf.zeros([1, n_hidden],  tf.float32))
	global bv
	bv = tf.Variable(tf.zeros([1, n_input],  tf.float32))

	x_sample = gibbs_sample(1)
	h = sample(tf.sigmoid(tf.matmul(X, W) + bh))
	h_sample = sample(tf.sigmoid(tf.matmul(x_sample, W) + bh))

	learning_rate = tf.constant(0.005, tf.float32)
	size_bt = tf.cast(tf.shape(X)[0], tf.float32)
	W_adder  = tf.multiply(learning_rate/size_bt, tf.subtract(tf.matmul(tf.transpose(X), h), tf.matmul(tf.transpose(x_sample), h_sample)))
	bv_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(X, x_sample), 0, True))
	bh_adder = tf.multiply(learning_rate/size_bt, tf.reduce_sum(tf.subtract(h, h_sample), 0, True))
	update = [W.assign_add(W_adder), bv.assign_add(bv_adder), bh.assign_add(bh_adder)]
	return update

# 训练神经网络
def train_neural_network():
	update = neural_network()

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		saver = tf.train.Saver(tf.all_variables())

		epochs = 256
		batch_size = 64
		for epoch in range(epochs):
			for song in songs:
				song = np.array(song)
				song = song[:int(np.floor(song.shape[0]/n_timesteps) * n_timesteps)]
				song = np.reshape(song, [song.shape[0]//n_timesteps, song.shape[1] * n_timesteps])
		
			for i in range(1, len(song), batch_size): 
				train_x = song[i:i+batch_size]
				sess.run(update, feed_dict={X: train_x})
			print(epoch)
			# 保存模型
			if epoch == epochs - 1:
				saver.save(sess, 'midi.module')

		# 生成midi
		sample = gibbs_sample(1).eval(session=sess, feed_dict={X: np.zeros((1, n_input))})
		S = np.reshape(sample[0,:], (n_timesteps, 2 * note_range))
		noteStateMatrixToMidi(S, "auto_gen_music")
		print('生成auto_gen_music.mid文件')

train_neural_network()
```
生成的mid音乐：auto_gen_music  
音乐分类：[https://github.com/despoisj/DeepAudioClassification](https://github.com/despoisj/DeepAudioClassification)  

原文链接：[http://blog.topspeedsnail.com/archives/10508](http://blog.topspeedsnail.com/archives/10508)
