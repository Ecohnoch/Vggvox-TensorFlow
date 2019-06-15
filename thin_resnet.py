import tensorflow as tf 
import numpy as np 

def identity_block2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
	filters1, filters2, filters3 = filters

	conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
	bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'

	x = tf.layers.conv2d(input_tensor, filters1, (1, 1), use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_1, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = tf.layers.conv2d(x, filters2, kernel_size, use_bias=False, padding='SAME', kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = tf.layers.conv2d(x, filters3, (1,1),  use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)
	
	x = tf.add(input_tensor, x)
	x = tf.nn.relu(x)
	return x


def conv_block_2d(input_tensor, kernel_size, filters, stage, block, is_training, reuse, strides=(2, 2), kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
	filters1, filters2, filters3 = filters

	conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
	bn_name_1   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_reduce'
	x = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides=strides, use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_1, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_1, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
	bn_name_2   = 'bn'   + str(stage) + '_' + str(block) + '_3x3'
	x = tf.layers.conv2d(x, filters2, kernel_size, padding='SAME', use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_2, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_2, reuse=reuse)
	x = tf.nn.relu(x)

	conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
	bn_name_3   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_increase'
	x = tf.layers.conv2d(x, filters3, (1,1), use_bias=False, kernel_initializer=kernel_initializer, name=conv_name_3, reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name=bn_name_3, reuse=reuse)

	conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_shortcut'
	bn_name_4   = 'bn'   + str(stage) + '_' + str(block) + '_1x1_shortcut'
	shortcut = tf.layers.conv2d(input_tensor, filters3, (1,1), use_bias=False, strides=strides, kernel_initializer=kernel_initializer, name=conv_name_4, reuse=reuse)
	shortcut = tf.layers.batch_normalization(shortcut, training=is_training, name=bn_name_4, reuse=reuse)

	x = tf.add(shortcut, x)
	x = tf.nn.relu(x)
	return x


def resnet34(input_tensor, is_training=True, pooling_and_fc=True, reuse=False, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)):
	x = tf.layers.conv2d(input_tensor, 64, (7,7), strides=(1,1), kernel_initializer=kernel_initializer, use_bias=False, padding='SAME', name='voice_conv1_1/3x3_s1', reuse=reuse)
	x = tf.layers.batch_normalization(x, training=is_training, name='voice_bn1_1/3x3_s1', reuse=reuse)
	x = tf.nn.relu(x)
	x = tf.layers.max_pooling2d(x, (2,2), strides=(2,2), name='voice_mpool1')

	x1 = conv_block_2d(x, 3, [48, 48, 96], stage=2, block='voice_1a', strides=(1,1), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x1 = identity_block2d(x1, 3, [48, 48, 96], stage=2, block='voice_1b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	x2 = conv_block_2d(x1, 3, [96, 96, 128], stage=3, block='voice_2a', strides=(2,2), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [96, 96, 128], stage=3, block='voice_2b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x2 = identity_block2d(x2, 3, [96, 96, 128], stage=3, block='voice_2c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	x3 = conv_block_2d(x2, 3, [128, 128, 256], stage=4, block='voice_3a', strides=(2,2), is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [128, 128, 256], stage=4, block='voice_3b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x3 = identity_block2d(x3, 3, [128, 128, 256], stage=4, block='voice_3c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	x4 = conv_block_2d(x3, 3, [256, 256, 512], stage=5, block='voice_4a', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x4 = identity_block2d(x4, 3, [256, 256, 512], stage=5, block='voice_4b', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)
	x4 = identity_block2d(x4, 3, [256, 256, 512], stage=5, block='voice_4c', is_training=is_training, reuse=reuse, kernel_initializer=kernel_initializer)

	if pooling_and_fc:
		pooling_output = tf.layers.max_pooling2d(x4, (3,1), strides=(2,2), name='voice_mpool2')
		# fc_output      = tf.layers.conv2d(pooling_output, 512, (7, 1), name='fc1', reuse=reuse)
	return pooling_output