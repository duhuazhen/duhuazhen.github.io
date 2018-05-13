---
layout:     post
title:     (转)基于RNN生成古诗词
date:       2017-01-03 12:00:00
author:     "DuHuazhen"
header-img: "img/post-bg-2015.jpg"
catalog: true
tags:
    - tensorflow
    - 深度学习
---
我的环境：ubuntu14.04+python3.5+anaconda+pygame  
RNN不像传统的神经网络-它们的输出输出是固定的，而RNN允许我们输入输出向量序列。RNN是为了对序列数据进行建模而产生的。    
样本序列性：样本间存在顺序关系，每个样本和它之前的样本存在关联。比如说，在文本中，一个词和它前面的词是有关联的；在气象数据中，一天的气温和前几天的气温是有关联的。   
例如本帖要使用RNN生成古诗，你给它输入一堆古诗词，它会学着生成和前面相关联的字词。如果你给它输入一堆姓名，它会学着生成姓名；给它输入一堆古典乐/歌词，它会学着生成古典乐/歌词，甚至可以给它输入源代码。   

本帖代码移植自[char-rnn](https://github.com/karpathy/char-rnn)，它是基于Torch的洋文模型，稍加修改即可应用于中文。char-rnn使用文本文件做为输入、训练RNN模型，然后使用它生成和训练数据类似的文本。  

使用的数据集：全唐诗(43030首)：[https://pan.baidu.com/s/1o7QlUhO](https://pan.baidu.com/s/1o7QlUhO)  
训练：
```python
import collections
import numpy as np
import tensorflow as tf
 
#-------------------------------数据预处理---------------------------#
 
poetry_file ='poetry.txt'
 
# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8',) as f:
	for line in f:
		try:
			title, content = line.strip().split(':')
			content = content.replace(' ','')
			if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
				continue
			if len(content) < 5 or len(content) > 79:
				continue
			content = '[' + content + ']'
			poetrys.append(content)
		except Exception as e: 
			pass
 
# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))
 
# 统计每个字出现次数
all_words = []
for poetry in poetrys:
	all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)
 
# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]
 
# 每次取64首诗进行训练
batch_size = 64
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
	start_index = i * batch_size
	end_index = start_index + batch_size
 
	batches = poetrys_vector[start_index:end_index]
	length = max(map(len,batches))
	xdata = np.full((batch_size,length), word_num_map[' '], np.int32)
	for row in range(batch_size):
		xdata[row,:len(batches[row])] = batches[row]
	ydata = np.copy(xdata)
	ydata[:,:-1] = xdata[:,1:]
	"""
	xdata             ydata
	[6,2,4,6,9]       [2,4,6,9,9]
	[1,4,2,8,5]       [4,2,8,5,5]
	"""
	x_batches.append(xdata)
	y_batches.append(ydata)
 
 
#---------------------------------------RNN--------------------------------------#
 
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
	if model == 'rnn':
		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell
 
	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
 
	initial_state = cell.zero_state(batch_size, tf.float32)
 
	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
		softmax_b = tf.get_variable("softmax_b", [len(words)+1])
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
 
	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	output = tf.reshape(outputs,[-1, rnn_size])
 
	logits = tf.matmul(output, softmax_w) + softmax_b
	probs = tf.nn.softmax(logits)
	return logits, last_state, probs, cell, initial_state
#训练
def train_neural_network():
	logits, last_state, _, _, _ = neural_network()
	targets = tf.reshape(output_targets, [-1])
	loss = 	tf.contrib.legacy_seq2seq.sequence_loss_by_example([logits], [targets], [tf.ones_like(targets, dtype=tf.float32)], len(words))
	cost = tf.reduce_mean(loss)
	learning_rate = tf.Variable(0.0, trainable=False)
	tvars = tf.trainable_variables()
	grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), 5)
	optimizer = tf.train.AdamOptimizer(learning_rate)
	train_op = optimizer.apply_gradients(zip(grads, tvars))
 
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
 
		saver = tf.train.Saver(tf.all_variables())
 
		for epoch in range(50):
			sess.run(tf.assign(learning_rate, 0.002 * (0.97 ** epoch)))
			n = 0
			for batche in range(n_chunk):
				train_loss, _ , _ = sess.run([cost, last_state, train_op], feed_dict={input_data: x_batches[n], output_targets: y_batches[n]})
				n += 1
				print(epoch, batche, train_loss)
			if epoch % 7 == 0:
				saver.save(sess, 'poetry.module', global_step=epoch)
 
train_neural_network()
```
大概要训练50次也就是目录下产生poetry.module-50文件。
生成古诗  
```python
import collections
import numpy as np
import tensorflow as tf
 
#-------------------------------数据预处理---------------------------#
 
poetry_file ='poetry.txt'
 
# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8',) as f:
	for line in f:
		try:
			title, content = line.strip().split(':')
			content = content.replace(' ','')
			if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
				continue
			if len(content) < 5 or len(content) > 79:
				continue
			content = '[' + content + ']'
			poetrys.append(content)
		except Exception as e: 
			pass
 
# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))
 
# 统计每个字出现次数
all_words = []
for poetry in poetrys:
	all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)
 
# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]
 
batch_size = 1
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
	start_index = i * batch_size
	end_index = start_index + batch_size
 
	batches = poetrys_vector[start_index:end_index]
	length = max(map(len,batches))
	xdata = np.full((batch_size,length), word_num_map[' '], np.int32)
	for row in range(batch_size):
		xdata[row,:len(batches[row])] = batches[row]
	ydata = np.copy(xdata)
	ydata[:,:-1] = xdata[:,1:]
	"""
	xdata             ydata
	[6,2,4,6,9]       [2,4,6,9,9]
	[1,4,2,8,5]       [4,2,8,5,5]
	"""
	x_batches.append(xdata)
	y_batches.append(ydata)
 
 
#---------------------------------------RNN--------------------------------------#
 
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
	if model == 'rnn':
		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell
 
	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
 
	initial_state = cell.zero_state(batch_size, tf.float32)
 
	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
		softmax_b = tf.get_variable("softmax_b", [len(words)+1])
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
 
	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	output = tf.reshape(outputs,[-1, rnn_size])
 
	logits = tf.matmul(output, softmax_w) + softmax_b
	probs = tf.nn.softmax(logits)
	return logits, last_state, probs, cell, initial_state
 
#-------------------------------生成古诗---------------------------------#
# 使用训练完成的模型
 
def gen_poetry():
	def to_word(weights):
		t = np.cumsum(weights)
		s = np.sum(weights)
		sample = int(np.searchsorted(t, np.random.rand(1)*s))
		return words[sample]
 
	_, last_state, probs, cell, initial_state = neural_network()
 
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
 
		saver = tf.train.Saver(tf.all_variables())
		saver.restore(sess, 'poetry.module-0')
 
		state_ = sess.run(cell.zero_state(1, tf.float32))
 
		x = np.array([list(map(word_num_map.get, '['))])
		[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
		word = to_word(probs_)
		#word = words[np.argmax(probs_)]
		poem = ''
		while word != ']':
			poem += word
			x = np.zeros((1,1))
			x[0,0] = word_num_map[word]
			[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
			word = to_word(probs_)
			#word = words[np.argmax(probs_)]
		return poem
 
print(gen_poetry())
```
替换你自己的 poetry.module-50训练模型。

生成藏头诗
```python
import collections
import numpy as np
import tensorflow as tf
 
#-------------------------------数据预处理---------------------------#
 
poetry_file ='poetry.txt'
 
# 诗集
poetrys = []
with open(poetry_file, "r", encoding='utf-8',) as f:
	for line in f:
		try:
			title, content = line.strip().split(':')
			content = content.replace(' ','')
			if '_' in content or '(' in content or '（' in content or '《' in content or '[' in content:
				continue
			if len(content) < 5 or len(content) > 79:
				continue
			content = '[' + content + ']'
			poetrys.append(content)
		except Exception as e: 
			pass
 
# 按诗的字数排序
poetrys = sorted(poetrys,key=lambda line: len(line))
print('唐诗总数: ', len(poetrys))
 
# 统计每个字出现次数
all_words = []
for poetry in poetrys:
	all_words += [word for word in poetry]
counter = collections.Counter(all_words)
count_pairs = sorted(counter.items(), key=lambda x: -x[1])
words, _ = zip(*count_pairs)
 
# 取前多少个常用字
words = words[:len(words)] + (' ',)
# 每个字映射为一个数字ID
word_num_map = dict(zip(words, range(len(words))))
# 把诗转换为向量形式，参考TensorFlow练习1
to_num = lambda word: word_num_map.get(word, len(words))
poetrys_vector = [ list(map(to_num, poetry)) for poetry in poetrys]
#[[314, 3199, 367, 1556, 26, 179, 680, 0, 3199, 41, 506, 40, 151, 4, 98, 1],
#[339, 3, 133, 31, 302, 653, 512, 0, 37, 148, 294, 25, 54, 833, 3, 1, 965, 1315, 377, 1700, 562, 21, 37, 0, 2, 1253, 21, 36, 264, 877, 809, 1]
#....]
 
batch_size = 1
n_chunk = len(poetrys_vector) // batch_size
x_batches = []
y_batches = []
for i in range(n_chunk):
	start_index = i * batch_size
	end_index = start_index + batch_size
 
	batches = poetrys_vector[start_index:end_index]
	length = max(map(len,batches))
	xdata = np.full((batch_size,length), word_num_map[' '], np.int32)
	for row in range(batch_size):
		xdata[row,:len(batches[row])] = batches[row]
	ydata = np.copy(xdata)
	ydata[:,:-1] = xdata[:,1:]
	"""
	xdata             ydata
	[6,2,4,6,9]       [2,4,6,9,9]
	[1,4,2,8,5]       [4,2,8,5,5]
	"""
	x_batches.append(xdata)
	y_batches.append(ydata)
 
 
#---------------------------------------RNN--------------------------------------#
 
input_data = tf.placeholder(tf.int32, [batch_size, None])
output_targets = tf.placeholder(tf.int32, [batch_size, None])
# 定义RNN
def neural_network(model='lstm', rnn_size=128, num_layers=2):
	if model == 'rnn':
		cell_fun = tf.nn.rnn_cell.BasicRNNCell
	elif model == 'gru':
		cell_fun = tf.nn.rnn_cell.GRUCell
	elif model == 'lstm':
		cell_fun = tf.nn.rnn_cell.BasicLSTMCell
 
	cell = cell_fun(rnn_size, state_is_tuple=True)
	cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=True)
 
	initial_state = cell.zero_state(batch_size, tf.float32)
 
	with tf.variable_scope('rnnlm'):
		softmax_w = tf.get_variable("softmax_w", [rnn_size, len(words)+1])
		softmax_b = tf.get_variable("softmax_b", [len(words)+1])
		with tf.device("/cpu:0"):
			embedding = tf.get_variable("embedding", [len(words)+1, rnn_size])
			inputs = tf.nn.embedding_lookup(embedding, input_data)
 
	outputs, last_state = tf.nn.dynamic_rnn(cell, inputs, initial_state=initial_state, scope='rnnlm')
	output = tf.reshape(outputs,[-1, rnn_size])
 
	logits = tf.matmul(output, softmax_w) + softmax_b
	probs = tf.nn.softmax(logits)
	return logits, last_state, probs, cell, initial_state
 
#-------------------------------生成古诗---------------------------------#
# 使用训练完成的模型
def gen_poetry_with_head(head):
	def to_word(weights):
		t = np.cumsum(weights)
		s = np.sum(weights)
		sample = int(np.searchsorted(t, np.random.rand(1)*s))
		return words[sample]
 
	_, last_state, probs, cell, initial_state = neural_network()
 
	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())
 
		saver = tf.train.Saver(tf.all_variables())
		saver.restore(sess, 'poetry.module-0')
 
		state_ = sess.run(cell.zero_state(1, tf.float32))
		poem = ''
		i = 0
		for word in head:
			while word != '，' and word != '。':
				poem += word
				x = np.array([list(map(word_num_map.get, word))])
				[probs_, state_] = sess.run([probs, last_state], feed_dict={input_data: x, initial_state: state_})
				word = to_word(probs_)
				#time.sleep(1)
			if i % 2 == 0:
				poem += '，'
			else:
				poem += '。'
			i += 1
		return poem
 
print(gen_poetry_with_head('一二三四'))

```
同样：替换你自己的 poetry.module-50训练模型。
读取模型的方法：
```python
	module_file = tf.train.latest_checkpoint('.')
	#print(module_file)
	saver.restore(sess, module_file)
```

[https://github.com/ryankiros/neural-storyteller](https://github.com/ryankiros/neural-storyteller)（根据图片生成故事）

转自：[http://blog.topspeedsnail.com/archives/tag/tensorflow/page/3](http://blog.topspeedsnail.com/archives/tag/tensorflow/page/3)
