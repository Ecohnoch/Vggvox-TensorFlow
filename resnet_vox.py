import tensorflow as tf 
import numpy as np 

def voicenet(input_x, reuse=False, is_training=True):
    with tf.name_scope('audio_embedding_network'):
        # input_x = tf.layers.batch_normalization(input_x, training=is_training, name='bbn0', reuse=reuse)
        with tf.name_scope('block0'):
            conv0_1 = tf.layers.conv2d(input_x, filters=64, kernel_size=[7,7], strides=[2,2], padding='SAME', reuse=reuse, name='conv1')
            conv0_1 = tf.layers.batch_normalization(conv0_1, training=is_training, name='bbn0', reuse=reuse)
            conv0_1 = tf.nn.relu(conv0_1)
            conv0_1 = tf.layers.max_pooling2d(conv0_1, pool_size=[3,3], strides=[2,2], name='mpool1')
        with tf.name_scope('block1'):
            with tf.variable_scope('block1_conv1') as scope:
                conv_block1_conv1_shortcut = tf.layers.conv2d(conv0_1, filters=256, kernel_size=[1,1], strides=[1,1], padding='VALID', reuse=reuse, name='conv_block1_conv1_shortcut_conv')
                conv_block1_conv1_shortcut = tf.layers.batch_normalization(conv_block1_conv1_shortcut, training=is_training, name='conv_block1_conv1_shortcut_bn', reuse=reuse)

                conv_block1_conv1_1 = tf.layers.conv2d(conv0_1, filters=64, kernel_size=[1,1], strides=[1,1], padding='VALID', reuse=reuse, name='conv_block1_conv1_1')
                conv_block1_conv1_1 = tf.layers.batch_normalization(conv_block1_conv1_1, training=is_training, name='conv_block1_conv1_1_bn', reuse=reuse)
                conv_block1_conv1_1 = tf.nn.relu(conv_block1_conv1_1)

                conv_block1_conv1_2 = tf.layers.conv2d(conv_block1_conv1_1, filters=64, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block1_conv1_2')
                conv_block1_conv1_2 = tf.layers.batch_normalization(conv_block1_conv1_2, training=is_training, name='conv_block1_conv1_2_bn', reuse=reuse)
                conv_block1_conv1_2 = tf.nn.relu(conv_block1_conv1_2)

                conv_block1_conv1_3 = tf.layers.conv2d(conv_block1_conv1_2, filters=256, kernel_size=[1,1], strides=[1,1], padding='VALID', reuse=reuse, name='conv_block1_conv1_3')
                conv_block1_conv1_3 = tf.layers.batch_normalization(conv_block1_conv1_3, training=is_training, name='conv_block1_conv1_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block1_output1 = conv_block1_conv1_shortcut + conv_block1_conv1_3
                conv_block1_output1 = tf.nn.relu(conv_block1_output1)

            with tf.variable_scope('block1_conv2') as scope:
                # conv_block1_conv2_shortcut = conv_block1_output1
                # conv_block1_conv2_shortcut = tf.layers.batch_normalization(conv_block1_conv1_shortcut, training=is_training, name='conv_block1_conv1_shortcut_bn', reuse=reuse)

                conv_block1_conv2_1 = tf.layers.conv2d(conv_block1_output1, filters=64, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block1_conv2_1')
                conv_block1_conv2_1 = tf.layers.batch_normalization(conv_block1_conv2_1, training=is_training, name='conv_block1_conv2_1_bn', reuse=reuse)
                conv_block1_conv2_1 = tf.nn.relu(conv_block1_conv2_1)

                conv_block1_conv2_2 = tf.layers.conv2d(conv_block1_conv2_1, filters=64, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block1_conv2_2')
                conv_block1_conv2_2 = tf.layers.batch_normalization(conv_block1_conv2_2, training=is_training, name='conv_block1_conv2_2_bn', reuse=reuse)
                conv_block1_conv2_2 = tf.nn.relu(conv_block1_conv2_2)

                conv_block1_conv2_3 = tf.layers.conv2d(conv_block1_conv2_2, filters=256, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block1_conv2_3')
                conv_block1_conv2_3 = tf.layers.batch_normalization(conv_block1_conv2_3, training=is_training, name='conv_block1_conv2_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block1_output2 = conv_block1_output1 + conv_block1_conv2_3
                conv_block1_output2 = tf.nn.relu(conv_block1_output2)

            with tf.variable_scope('block1_conv3') as scope:
                # conv_block1_conv2_shortcut = conv_block1_output1
                # conv_block1_conv2_shortcut = tf.layers.batch_normalization(conv_block1_conv1_shortcut, training=is_training, name='conv_block1_conv1_shortcut_bn', reuse=reuse)

                conv_block1_conv3_1 = tf.layers.conv2d(conv_block1_output2, filters=64, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block1_conv3_1')
                conv_block1_conv3_1 = tf.layers.batch_normalization(conv_block1_conv3_1, training=is_training, name='conv_block1_conv3_1_bn', reuse=reuse)
                conv_block1_conv3_1 = tf.nn.relu(conv_block1_conv3_1)

                conv_block1_conv3_2 = tf.layers.conv2d(conv_block1_conv3_1, filters=64, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block1_conv3_2')
                conv_block1_conv3_2 = tf.layers.batch_normalization(conv_block1_conv3_2, training=is_training, name='conv_block1_conv3_2_bn', reuse=reuse)
                conv_block1_conv3_2 = tf.nn.relu(conv_block1_conv3_2)

                conv_block1_conv3_3 = tf.layers.conv2d(conv_block1_conv3_2, filters=256, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block1_conv3_3')
                conv_block1_conv3_3 = tf.layers.batch_normalization(conv_block1_conv3_3, training=is_training, name='cconv_block1_conv3_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block1_output3 = conv_block1_output2 + conv_block1_conv3_3
                conv_block1_output3 = tf.nn.relu(conv_block1_output2)
            
        with tf.name_scope('block2'):
            with tf.variable_scope('block2_conv1') as scope:
                conv_block2_conv1_shortcut = tf.layers.conv2d(conv_block1_output3, filters=512, kernel_size=[1,1], strides=[2,2], padding='SAME', reuse=reuse, name='conv_block2_conv1_shortcut_conv')
                conv_block2_conv1_shortcut = tf.layers.batch_normalization(conv_block2_conv1_shortcut, training=is_training, name='conv_block2_conv1_shortcut_bn', reuse=reuse)
                print(conv_block2_conv1_shortcut)
                conv_block2_conv1_1 = tf.layers.conv2d(conv_block1_output3, filters=128, kernel_size=[1,1], strides=[2,2], padding='SAME', reuse=reuse, name='conv_block2_conv1_1')
                conv_block2_conv1_1 = tf.layers.batch_normalization(conv_block2_conv1_1, training=is_training, name='conv_block2_conv1_1_bn', reuse=reuse)
                conv_block2_conv1_1 = tf.nn.relu(conv_block2_conv1_1)

                conv_block2_conv1_2 = tf.layers.conv2d(conv_block2_conv1_1, filters=128, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv1_2')
                conv_block2_conv1_2 = tf.layers.batch_normalization(conv_block2_conv1_2, training=is_training, name='conv_block2_conv1_2_bn', reuse=reuse)
                conv_block2_conv1_2 = tf.nn.relu(conv_block2_conv1_2)

                conv_block2_conv1_3 = tf.layers.conv2d(conv_block2_conv1_2, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv1_3')
                conv_block2_conv1_3 = tf.layers.batch_normalization(conv_block2_conv1_3, training=is_training, name='conv_block2_conv1_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block2_output1 = conv_block2_conv1_shortcut + conv_block2_conv1_3
                conv_block2_output1 = tf.nn.relu(conv_block2_output1)

            with tf.variable_scope('block2_conv2') as scope:
                # conv_block2_conv1_shortcut = tf.layers.conv2d(conv_block1_output3, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv1_shortcut_conv')
                # conv_block2_conv1_shortcut = tf.layers.batch_normalization(conv_block2_conv1_shortcut, training=is_training, name='conv_block2_conv1_shortcut_bn', reuse=reuse)

                conv_block2_conv2_1 = tf.layers.conv2d(conv_block2_output1, filters=128, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv2_1')
                conv_block2_conv2_1 = tf.layers.batch_normalization(conv_block2_conv2_1, training=is_training, name='conv_block2_conv2_1_bn', reuse=reuse)
                conv_block2_conv2_1 = tf.nn.relu(conv_block2_conv2_1)

                conv_block2_conv2_2 = tf.layers.conv2d(conv_block2_conv2_1, filters=128, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv2_2')
                conv_block2_conv2_2 = tf.layers.batch_normalization(conv_block2_conv2_2, training=is_training, name='conv_block2_conv2_2_bn', reuse=reuse)
                conv_block2_conv2_2 = tf.nn.relu(conv_block2_conv2_2)

                conv_block2_conv2_3 = tf.layers.conv2d(conv_block2_conv2_2, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv2_3')
                conv_block2_conv2_3 = tf.layers.batch_normalization(conv_block2_conv2_3, training=is_training, name='conv_block2_conv2_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block2_output2 = conv_block2_output1 + conv_block2_conv2_3
                conv_block2_output2 = tf.nn.relu(conv_block2_output2)

            with tf.variable_scope('block2_conv3') as scope:
                # conv_block2_conv1_shortcut = tf.layers.conv2d(conv_block1_output3, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv1_shortcut_conv')
                # conv_block2_conv1_shortcut = tf.layers.batch_normalization(conv_block2_conv1_shortcut, training=is_training, name='conv_block2_conv1_shortcut_bn', reuse=reuse)

                conv_block2_conv3_1 = tf.layers.conv2d(conv_block2_output2, filters=128, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv3_1')
                conv_block2_conv3_1 = tf.layers.batch_normalization(conv_block2_conv3_1, training=is_training, name='conv_block2_conv3_1_bn', reuse=reuse)
                conv_block2_conv3_1 = tf.nn.relu(conv_block2_conv3_1)

                conv_block2_conv3_2 = tf.layers.conv2d(conv_block2_conv3_1, filters=128, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv3_2')
                conv_block2_conv3_2 = tf.layers.batch_normalization(conv_block2_conv3_2, training=is_training, name='conv_block2_conv3_2_bn', reuse=reuse)
                conv_block2_conv3_2 = tf.nn.relu(conv_block2_conv3_2)

                conv_block2_conv3_3 = tf.layers.conv2d(conv_block2_conv3_2, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv3_3')
                conv_block2_conv3_3 = tf.layers.batch_normalization(conv_block2_conv3_3, training=is_training, name='conv_block2_conv3_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block2_output3 = conv_block2_output2 + conv_block2_conv3_3
                conv_block2_output3 = tf.nn.relu(conv_block2_output3)

            with tf.variable_scope('block2_conv4') as scope:
                # conv_block2_conv1_shortcut = tf.layers.conv2d(conv_block1_output3, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv1_shortcut_conv')
                # conv_block2_conv1_shortcut = tf.layers.batch_normalization(conv_block2_conv1_shortcut, training=is_training, name='conv_block2_conv1_shortcut_bn', reuse=reuse)

                conv_block2_conv4_1 = tf.layers.conv2d(conv_block2_output3, filters=128, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv4_1')
                conv_block2_conv4_1 = tf.layers.batch_normalization(conv_block2_conv4_1, training=is_training, name='conv_block2_conv4_1_bn', reuse=reuse)
                conv_block2_conv4_1 = tf.nn.relu(conv_block2_conv4_1)

                conv_block2_conv4_2 = tf.layers.conv2d(conv_block2_conv4_1, filters=128, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv4_2')
                conv_block2_conv4_2 = tf.layers.batch_normalization(conv_block2_conv4_2, training=is_training, name='conv_block2_conv4_2_bn', reuse=reuse)
                conv_block2_conv4_2 = tf.nn.relu(conv_block2_conv4_2)

                conv_block2_conv4_3 = tf.layers.conv2d(conv_block2_conv4_2, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block2_conv4_3')
                conv_block2_conv4_3 = tf.layers.batch_normalization(conv_block2_conv4_3, training=is_training, name='conv_block2_conv4_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block2_output4 = conv_block2_output3 + conv_block2_conv4_3
                conv_block2_output4 = tf.nn.relu(conv_block2_output4)

        with tf.name_scope('block3'):
            with tf.variable_scope('block3_conv1') as scope:
                conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[2,2], padding='SAME', reuse=reuse, name='conv_block3_conv1_shortcut')
                conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=is_training, name='conv_block3_conv1_shortcut_bn', reuse=reuse)

                conv_block3_conv1_1 = tf.layers.conv2d(conv_block2_output4, filters=256, kernel_size=[1,1], strides=[2,2], padding='SAME', reuse=reuse, name='conv_block3_conv1_1')
                conv_block3_conv1_1 = tf.layers.batch_normalization(conv_block3_conv1_1, training=is_training, name='conv_block3_conv1_1_bn', reuse=reuse)
                conv_block3_conv1_1 = tf.nn.relu(conv_block3_conv1_1)

                conv_block3_conv1_2 = tf.layers.conv2d(conv_block3_conv1_1, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv1_2')
                conv_block3_conv1_2 = tf.layers.batch_normalization(conv_block3_conv1_2, training=is_training, name='conv_block3_conv1_2_bn', reuse=reuse)
                conv_block3_conv1_2 = tf.nn.relu(conv_block3_conv1_2)

                conv_block3_conv1_3 = tf.layers.conv2d(conv_block3_conv1_2, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv1_3')
                conv_block3_conv1_3 = tf.layers.batch_normalization(conv_block3_conv1_3, training=is_training, name='conv_block3_conv1_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block3_output1 = conv_block3_conv1_shortcut + conv_block3_conv1_3
                conv_block3_output1 = tf.nn.relu(conv_block3_output1)

            with tf.variable_scope('block3_conv2') as scope:
                # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv1_shortcut')
                # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=is_training, name='conv_block3_conv1_shortcut_bn', reuse=reuse)

                conv_block3_conv2_1 = tf.layers.conv2d(conv_block3_output1, filters=256, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv2_1')
                conv_block3_conv2_1 = tf.layers.batch_normalization(conv_block3_conv2_1, training=is_training, name='conv_block3_conv2_1_bn', reuse=reuse)
                conv_block3_conv2_1 = tf.nn.relu(conv_block3_conv2_1)

                conv_block3_conv2_2 = tf.layers.conv2d(conv_block3_conv2_1, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv2_2')
                conv_block3_conv2_2 = tf.layers.batch_normalization(conv_block3_conv2_2, training=is_training, name='conv_block3_conv2_2_bn', reuse=reuse)
                conv_block3_conv2_2 = tf.nn.relu(conv_block3_conv2_2)

                conv_block3_conv2_3 = tf.layers.conv2d(conv_block3_conv2_2, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv2_3')
                conv_block3_conv2_3 = tf.layers.batch_normalization(conv_block3_conv1_3, training=is_training, name='conv_block3_conv2_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block3_output2 = conv_block3_output1 + conv_block3_conv2_3
                conv_block3_output2 = tf.nn.relu(conv_block3_output2)

            with tf.variable_scope('block3_conv3') as scope:
                # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv1_shortcut')
                # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=is_training, name='conv_block3_conv1_shortcut_bn', reuse=reuse)

                conv_block3_conv3_1 = tf.layers.conv2d(conv_block3_output2, filters=256, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv3_1')
                conv_block3_conv3_1 = tf.layers.batch_normalization(conv_block3_conv3_1, training=is_training, name='conv_block3_conv3_1_1_bn', reuse=reuse)
                conv_block3_conv3_1 = tf.nn.relu(conv_block3_conv3_1)

                conv_block3_conv3_2 = tf.layers.conv2d(conv_block3_conv3_1, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv3_2')
                conv_block3_conv3_2 = tf.layers.batch_normalization(conv_block3_conv3_2, training=is_training, name='conv_block3_conv3_2_bn', reuse=reuse)
                conv_block3_conv3_2 = tf.nn.relu(conv_block3_conv3_2)

                conv_block3_conv3_3 = tf.layers.conv2d(conv_block3_conv3_2, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv3_3')
                conv_block3_conv3_3 = tf.layers.batch_normalization(conv_block3_conv3_3, training=is_training, name='conv_block3_conv3_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block3_output3 = conv_block3_output2 + conv_block3_conv3_3
                conv_block3_output3 = tf.nn.relu(conv_block3_output3)

            with tf.variable_scope('block3_conv4') as scope:
                # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv1_shortcut')
                # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=is_training, name='conv_block3_conv1_shortcut_bn', reuse=reuse)

                conv_block3_conv4_1 = tf.layers.conv2d(conv_block3_output3, filters=256, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv4_1')
                conv_block3_conv4_1 = tf.layers.batch_normalization(conv_block3_conv4_1, training=is_training, name='conv_block3_conv4_1_bn', reuse=reuse)
                conv_block3_conv4_1 = tf.nn.relu(conv_block3_conv4_1)

                conv_block3_conv4_2 = tf.layers.conv2d(conv_block3_conv4_1, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv4_2')
                conv_block3_conv4_2 = tf.layers.batch_normalization(conv_block3_conv4_2, training=is_training, name='conv_block3_conv4_2_bn', reuse=reuse)
                conv_block3_conv4_2 = tf.nn.relu(conv_block3_conv4_2)

                conv_block3_conv4_3 = tf.layers.conv2d(conv_block3_conv4_2, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv4_3')
                conv_block3_conv4_3 = tf.layers.batch_normalization(conv_block3_conv3_3, training=is_training, name='conv_block3_conv4_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block3_output4 = conv_block3_output3 + conv_block3_conv4_3
                conv_block3_output4 = tf.nn.relu(conv_block3_output4)

            with tf.variable_scope('block3_conv5') as scope:
                # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv1_shortcut')
                # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=is_training, name='conv_block3_conv1_shortcut_bn', reuse=reuse)

                conv_block3_conv5_1 = tf.layers.conv2d(conv_block3_output4, filters=256, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv5_1')
                conv_block3_conv5_1 = tf.layers.batch_normalization(conv_block3_conv5_1, training=is_training, name='conv_block3_conv5_1_bn', reuse=reuse)
                conv_block3_conv5_1 = tf.nn.relu(conv_block3_conv5_1)

                conv_block3_conv5_2 = tf.layers.conv2d(conv_block3_conv5_1, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv5_2')
                conv_block3_conv5_2 = tf.layers.batch_normalization(conv_block3_conv5_2, training=is_training, name='conv_block3_conv5_2_bn', reuse=reuse)
                conv_block3_conv5_2 = tf.nn.relu(conv_block3_conv5_2)

                conv_block3_conv5_3 = tf.layers.conv2d(conv_block3_conv5_2, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv5_3')
                conv_block3_conv5_3 = tf.layers.batch_normalization(conv_block3_conv5_3, training=is_training, name='conv_block3_conv5_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block3_output5 = conv_block3_output4 + conv_block3_conv5_3
                conv_block3_output5 = tf.nn.relu(conv_block3_output5)

            with tf.variable_scope('block3_conv6') as scope:
                # conv_block3_conv1_shortcut = tf.layers.conv2d(conv_block2_output4, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv1_shortcut')
                # conv_block3_conv1_shortcut = tf.layers.batch_normalization(conv_block3_conv1_shortcut, training=is_training, name='conv_block3_conv1_shortcut_bn', reuse=reuse)

                conv_block3_conv6_1 = tf.layers.conv2d(conv_block3_output5, filters=256, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv6_1')
                conv_block3_conv6_1 = tf.layers.batch_normalization(conv_block3_conv6_1, training=is_training, name='conv_block3_conv6_1_bn', reuse=reuse)
                conv_block3_conv6_1 = tf.nn.relu(conv_block3_conv6_1)

                conv_block3_conv6_2 = tf.layers.conv2d(conv_block3_conv5_1, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv6_2')
                conv_block3_conv6_2 = tf.layers.batch_normalization(conv_block3_conv6_2, training=is_training, name='conv_block3_conv6_2_bn', reuse=reuse)
                conv_block3_conv6_2 = tf.nn.relu(conv_block3_conv6_2)

                conv_block3_conv6_3 = tf.layers.conv2d(conv_block3_conv6_2, filters=1024, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block3_conv6_3')
                conv_block3_conv6_3 = tf.layers.batch_normalization(conv_block3_conv6_3, training=is_training, name='conv_block3_conv6_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block3_output6 = conv_block3_output5 + conv_block3_conv6_3
                conv_block3_output6 = tf.nn.relu(conv_block3_output6)

        with tf.name_scope('block4'):
            with tf.variable_scope('block4_conv1') as scope:
                conv_block4_conv1_shortcut = tf.layers.conv2d(conv_block3_output6, filters=2048, kernel_size=[1,1], strides=[2,2], padding='SAME', reuse=reuse, name='conv_block4_conv1_shortcut')
                conv_block4_conv1_shortcut = tf.layers.batch_normalization(conv_block4_conv1_shortcut, training=is_training, name='conv_block4_conv1_shortcut_bn', reuse=reuse)

                conv_block4_conv1_1 = tf.layers.conv2d(conv_block3_output6, filters=512, kernel_size=[1,1], strides=[2,2], padding='SAME', reuse=reuse, name='conv_block4_conv1_1')
                conv_block4_conv1_1 = tf.layers.batch_normalization(conv_block4_conv1_1, training=is_training, name='conv_block4_conv1_1_bn', reuse=reuse)
                conv_block4_conv1_1 = tf.nn.relu(conv_block4_conv1_1)

                conv_block4_conv1_2 = tf.layers.conv2d(conv_block4_conv1_1, filters=512, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv1_2')
                conv_block4_conv1_2 = tf.layers.batch_normalization(conv_block4_conv1_2, training=is_training, name='conv_block4_conv1_2_bn', reuse=reuse)
                conv_block4_conv1_2 = tf.nn.relu(conv_block4_conv1_2)

                conv_block4_conv1_3 = tf.layers.conv2d(conv_block4_conv1_2, filters=2048, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv1_3')
                conv_block4_conv1_3 = tf.layers.batch_normalization(conv_block4_conv1_3, training=is_training, name='conv_block4_conv1_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block4_output1 = conv_block4_conv1_shortcut + conv_block4_conv1_3
                conv_block4_output1 = tf.nn.relu(conv_block4_output1)

            with tf.variable_scope('block4_conv2') as scope:
                # conv_block4_conv1_shortcut = tf.layers.conv2d(conv_block3_output6, filters=2048, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv1_shortcut')
                # conv_block4_conv1_shortcut = tf.layers.batch_normalization(conv_block4_conv1_shortcut, training=is_training, name='conv_block4_conv1_shortcut_bn', reuse=reuse)

                conv_block4_conv2_1 = tf.layers.conv2d(conv_block4_output1, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv2_1')
                conv_block4_conv2_1 = tf.layers.batch_normalization(conv_block4_conv2_1, training=is_training, name='conv_block4_conv2_1_bn', reuse=reuse)
                conv_block4_conv2_1 = tf.nn.relu(conv_block4_conv2_1)

                conv_block4_conv2_2 = tf.layers.conv2d(conv_block4_conv2_1, filters=512, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv2_2')
                conv_block4_conv2_2 = tf.layers.batch_normalization(conv_block4_conv2_2, training=is_training, name='conv_block4_conv2_2_bn', reuse=reuse)
                conv_block4_conv2_2 = tf.nn.relu(conv_block4_conv2_2)

                conv_block4_conv2_3 = tf.layers.conv2d(conv_block4_conv2_2, filters=2048, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv2_3')
                conv_block4_conv2_3 = tf.layers.batch_normalization(conv_block4_conv2_3, training=is_training, name='conv_block4_conv2_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block4_output2 = conv_block4_output1 + conv_block4_conv2_3
                conv_block4_output2 = tf.nn.relu(conv_block4_output2)

            with tf.variable_scope('block4_conv3') as scope:
                # conv_block4_conv1_shortcut = tf.layers.conv2d(conv_block3_output6, filters=2048, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv1_shortcut')
                # conv_block4_conv1_shortcut = tf.layers.batch_normalization(conv_block4_conv1_shortcut, training=is_training, name='conv_block4_conv1_shortcut_bn', reuse=reuse)

                conv_block4_conv3_1 = tf.layers.conv2d(conv_block4_output2, filters=512, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv3_1')
                conv_block4_conv3_1 = tf.layers.batch_normalization(conv_block4_conv3_1, training=is_training, name='conv_block4_conv3_1_bn', reuse=reuse)
                conv_block4_conv3_1 = tf.nn.relu(conv_block4_conv3_1)

                conv_block4_conv3_2 = tf.layers.conv2d(conv_block4_conv3_1, filters=512, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv3_2')
                conv_block4_conv3_2 = tf.layers.batch_normalization(conv_block4_conv3_2, training=is_training, name='conv_block4_conv3_2_bn', reuse=reuse)
                conv_block4_conv3_2 = tf.nn.relu(conv_block4_conv3_2)

                conv_block4_conv3_3 = tf.layers.conv2d(conv_block4_conv3_2, filters=2048, kernel_size=[1,1], strides=[1,1], padding='SAME', reuse=reuse, name='conv_block4_conv3_3')
                conv_block4_conv3_3 = tf.layers.batch_normalization(conv_block4_conv3_3, training=is_training, name='conv_block4_conv3_3_bn', reuse=reuse)
                # conv_block1_conv1_3 = tf.nn.relu(conv_block1_conv1_3)
                conv_block4_output3 = conv_block4_output2 + conv_block4_conv3_3
                conv_block4_output3 = tf.nn.relu(conv_block4_output3)
            print(conv_block4_output3)
            fc1 = tf.layers.conv2d(conv_block4_output3, filters=2048, kernel_size=[16,1], strides=[1,1], padding='VALID', reuse=reuse, name='fc1')
            fc2 = tf.reduce_mean(fc1, axis=[1, 2], name='avgpool')
        return fc2

if __name__ == '__main__':
	example_data = [np.random.rand(512, 300, 1)]
	x = tf.placeholder(tf.float32, [None, 512, 300, 1])
	y = voicenet(x)
	# print(y)