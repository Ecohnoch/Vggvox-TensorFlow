from __future__ import unicode_literals

from functools import reduce
import tensorflow as tf
import numpy as np

import os
import random
import warnings
import argparse
import scipy.io.wavfile

from wav_reader import get_fft_spectrum
from utils.my_utils import audio_data_extract


average_pooling_dict = {
    300: 8,
    400: 11,
    500: 14,
    600: 18,
    700: 21,
    800: 24,
    900: 27,
    1000: 30
}

def voicenet(input_x, reuse=False, is_training=True):
    with tf.name_scope('audio_embedding_network'):
        # input_x = tf.layers.batch_normalization(input_x, training=is_training, name='bbn0', reuse=reuse)
        with tf.variable_scope('conv1') as scope:
            conv1_1 = tf.layers.conv2d(input_x, filters=96, kernel_size=[7,7], strides=[2,2], padding='SAME', reuse=reuse, name='cc1')
            conv1_1 = tf.layers.batch_normalization(conv1_1, training=is_training, name='bbn1', reuse=reuse)
            conv1_1 = tf.nn.relu(conv1_1)
            conv1_1 = tf.layers.max_pooling2d(conv1_1, pool_size=[3,3], strides=[2,2], name='mpool1')
            
        with tf.variable_scope('conv2') as scope:
            conv2_1 = tf.layers.conv2d(conv1_1, filters=256, kernel_size=[5,5], strides=[2,2], padding='SAME', reuse=reuse, name='cc2')
            conv2_1 = tf.layers.batch_normalization(conv2_1, training=is_training, name='bbn2', reuse=reuse)
            conv2_1 = tf.nn.relu(conv2_1)
            conv2_1 = tf.layers.max_pooling2d(conv2_1, pool_size=[3,3], strides=[2,2], name='mpool2')


        with tf.variable_scope('conv3') as scope:
            conv3_1 = tf.layers.conv2d(conv2_1, filters=384, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='cc3_1')
            conv3_1 = tf.layers.batch_normalization(conv3_1, training=is_training, name='bbn3', reuse=reuse)
            conv3_1 = tf.nn.relu(conv3_1)
            
            conv3_2 = tf.layers.conv2d(conv3_1, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='cc3_2')
            conv3_2 = tf.layers.batch_normalization(conv3_2, training=is_training, name='bbn4', reuse=reuse)
            conv3_2 = tf.nn.relu(conv3_2)

            conv3_3 = tf.layers.conv2d(conv3_2, filters=256, kernel_size=[3,3], strides=[1,1], padding='SAME', reuse=reuse, name='cc3_3')
            conv3_3 = tf.layers.batch_normalization(conv3_3, training=is_training, name='bbn5', reuse=reuse)
            conv3_3 = tf.nn.relu(conv3_3)
            conv3_3 = tf.layers.max_pooling2d(conv3_3, pool_size=[5,3], strides=[3,2], name='mpool3')

        with tf.variable_scope('conv4') as scope:
            conv4_3 = tf.layers.conv2d(conv3_3, filters=4096, kernel_size=[9,1], strides=[1,1], padding='VALID', reuse=reuse, name='cc4_1')
            conv4_3 = tf.layers.batch_normalization(conv4_3, training=is_training, name='bbn6', reuse=reuse)
            conv4_3 = tf.nn.relu(conv4_3)
            # conv4_3 = tf.layers.average_pooling2d(conv4_3, pool_size=[1, conv4_3.shape[2]], strides=[1,1], name='apool4')
            conv4_3 = tf.reduce_mean(conv4_3, axis=[1, 2], name='apool4')
    # if is_training:
    #     conv4_3 = tf.nn.dropout(conv4_3, 0.5)
    # flattened = tf.contrib.layers.flatten(conv4_3)
    flattened = tf.nn.l2_normalize(conv4_3)
    features = tf.layers.dense(flattened, 1024, reuse=reuse, name='fc_audio_vgg')

    # if is_training:
    #     features = tf.nn.dropout(features, 0.5)
    # features = tf.contrib.layers.fully_connected(flattened, 1024, reuse=reuse, activation_fn=None, scope='fc_vgg')

    return features


def build_buckets(max_sec, step_sec, frame_step):
    buckets = {}
    frames_per_sec = int(1/frame_step)
    end_frame = int(max_sec*frames_per_sec)
    step_frame = int(step_sec*frames_per_sec)
    for i in range(0, end_frame+1, step_frame):
        s = i
        s = np.floor((s-7+2)/2) + 1  # conv1
        s = np.floor((s-3)/2) + 1  # mpool1
        s = np.floor((s-5+2)/2) + 1  # conv2
        s = np.floor((s-3)/2) + 1  # mpool2
        s = np.floor((s-3+2)/1) + 1  # conv3
        s = np.floor((s-3+2)/1) + 1  # conv4
        s = np.floor((s-3+2)/1) + 1  # conv5
        s = np.floor((s-3)/2) + 1  # mpool5
        s = np.floor((s-1)/1) + 1  # fc6
        if s > 0:
            buckets[i] = int(s)
    return buckets

def load_wave(wave_file):
    buckets = build_buckets(10, 1, 0.01)
    data = get_fft_spectrum(wave_file, buckets)

    if data.shape[1] == 300:
        pass
    else:
        start = np.random.randint(0, data.shape[1] - 300)
        data = data[:, start:start+300]

    data = np.expand_dims(data, -1)
    return data

def load_wave_list(wave_file_list):
    ans = []
    for i in wave_file_list:
        data = load_wave(i)
        ans.append(data)
    return ans








def train(opt):
    lr            = opt.lr
    rs            = opt.random_seed
    batch_size    = opt.batch_size
    wav_dir       = opt.voxceleb_wav_dir
    split_file    = opt.vox_split_txt_file
    ckpt_save_dir = opt.ckpt_save_dir
    epoch_time    = opt.epoch

    print('-------------Training Args-----------------')
    print('--lr           :  ', lr)
    print('--rs           :  ', rs)
    print('--batch_size   :  ', batch_size)
    print('--wav_dir      :  ', wav_dir)
    print('--split_file   :  ', split_file)
    print('--ckpt_save_dir:  ', ckpt_save_dir)
    print('--epoch_time   :  ', epoch_time)
    # print('-------------------------------------------')
    
    all_people = os.listdir(wav_dir)
    all_people.sort()
    id2label = {}

    n_classes = len(all_people)
    start = 0
    for people in all_people:
        id2label[people] = start
        start += 1
    # print(id2label)

    train_audio, test_audio, veri_audio = audio_data_extract(split_file)
    train_audio = np.array(train_audio)

    test_audio = np.array(test_audio[:50])
    
    train_label = [ id2label[idx] for idx in train_audio[:, 1] ]
    test_label  = [ id2label[idxx] for idxx in test_audio[:, 1] ]
    test_audio = test_audio[:, 0]
    test_audio = [wav_dir + each_file for each_file in test_audio]
    test_audio = np.array(test_audio)

    train_label = np.array(train_label)
    train_audio = train_audio[:, 0]
    train_audio = [wav_dir + each_file for each_file in train_audio]
    train_audio = np.array(train_audio)
    np.random.seed(rs)
    np.random.shuffle(train_audio)
    np.random.seed(rs)
    np.random.shuffle(train_label)
    print('--dataset shape:  ', train_audio.shape, train_label.shape)
    print('-------------------------------------------')

    def get_batch(dataset, start, batch_size=32):
        end = (start + batch_size) % dataset.shape[0]
        if end > start:
            return load_wave_list(dataset[start:end]), end, False
        
        sub_dataset = np.hstack((dataset[start:], dataset[:end]))
        return load_wave_list(sub_dataset), end, True

    def get_label_batch(dataset, start, batch_size=32):
        end = (start + batch_size) % dataset.shape[0]
        if end > start:
            return dataset[start:end], end, False

        return np.hstack((dataset[start:], dataset[:end])), end, True


    x = tf.placeholder(tf.float32, [None, 512, 300, 1], name='audio_input')
    y_s = tf.placeholder(tf.int64, [None])
    y = tf.one_hot(y_s, n_classes, axis=-1)

    emb_ori = voicenet(x)
    emb = tf.layers.dense(emb_ori, n_classes)
    # emb = tf.contrib.layers.fully_connected(emb_ori, n_classes, activation_fn=None)
    emb_softmax = tf.nn.softmax(emb, axis=1)


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=emb, labels=y))


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(emb_softmax, 1), y_s), tf.float32))

    emb_test = voicenet(x, is_training=False, reuse=True)
    emb_softmax_test = tf.nn.softmax(emb, axis=1)
    accuracy_test = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(emb_softmax_test, 1), y_s), tf.float32))


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainer = tf.train.AdamOptimizer(lr).minimize(loss)

    saver = tf.train.Saver(max_to_keep=3, var_list=tf.global_variables())

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        # saver.restore(sess, '/data/ChuyuanXiong/backup/triplet_backup2/Speaker_vox_iter_51500.ckpt')

        counter = 0
        for i in range(epoch_time):
            idx_train = 0
            idx_train_label = 0

            while True:
                batch_train, idx_train, end_epoch = get_batch(train_audio, idx_train, batch_size=batch_size)
                batch_train_label, idx_train_label, end_epoch = get_label_batch(train_label, idx_train_label, batch_size=batch_size)

                batch_train = np.array(batch_train)

                _, loss_val, acc_val, emb_softmax_val = sess.run([trainer, loss, accuracy, emb], feed_dict={x: batch_train, y_s: batch_train_label})
                
                counter += 1

                if counter % 100 == 0:
                    print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)
                    filename = 'Speaker_vox_iter_{:d}'.format(counter) + '.ckpt'
                    filename = os.path.join(ckpt_save_dir, filename)
                    saver.save(sess, filename)

                    acc_val = sess.run(accuracy_test, feed_dict={x: load_wave_list(test_audio), y_s: test_label})
                    print('test acc:', acc_val)

                    # print(emb_softmax_val)

                if end_epoch:
                    print('end epoch: ', i)
                    break


def test(opt):
    lr                = opt.lr
    rs                = opt.random_seed
    batch_size        = opt.batch_size
    wav_dir           = opt.voxceleb_wav_dir
    split_file        = opt.vox_split_txt_file
    ckpt_restore_file = opt.ckpt_restore_file

    print('-------------Training Args-----------------')
    print('--lr               :  ', lr)
    print('--rs               :  ', rs)
    print('--batch_size       :  ', batch_size)
    print('--wav_dir          :  ', wav_dir)
    print('--split_file       :  ', split_file)
    print('--ckpt_restore_file:  ', ckpt_restore_file)
    # print('-------------------------------------------')
    
    wav_dir = wav_dir
    all_people = os.listdir(wav_dir)
    all_people.sort()
    id2label = {}

    n_classes = len(all_people)
    start = 0
    for people in all_people:
        id2label[people] = start
        start += 1
    # print(id2label)

    train_audio, test_audio, veri_audio = audio_data_extract(split_file)
    test_audio = np.array(test_audio)

    test_label = [ id2label[idx] for idx in test_audio[:, 1] ]
    test_label = np.array(test_label)
    test_audio = test_audio[:, 0]
    test_audio = [wav_dir + each_file for each_file in test_audio]
    test_audio = np.array(test_audio)
    np.random.seed(rs)
    np.random.shuffle(test_audio)
    np.random.seed(rs)
    np.random.shuffle(test_label)
    print(test_audio.shape, test_label.shape)
    print('--dataset shape:  ', train_audio.shape, train_label.shape)
    print('-------------------------------------------')

    def get_batch(dataset, start, batch_size=32):
        end = (start + batch_size) % len(dataset)
        if end > start:
            return load_wave_list_305(dataset[start:end]), end, False

        sub_dataset = np.hstack((dataset[start:], dataset[:end]))
        return load_wave_list_305(sub_dataset), end, True

    def get_label_batch(dataset, start, batch_size=32):
        end = (start + batch_size) % len(dataset)
        if end > start:
            return dataset[start:end], end, False
        return np.hstack((dataset[start:], dataset[:end])), end, True


    x = tf.placeholder(tf.float32, [None, 512, 300, 1], name='audio_input')
    y_s = tf.placeholder(tf.int64, [None])
    y = tf.one_hot(y_s, n_classes, axis=-1)

    emb_ori = voicenet(x, is_training=True)
    emb = tf.layers.dense(emb_ori, n_classes)
    # emb = tf.contrib.layers.fully_connected(emb_ori, n_classes, activation_fn=None)

    # emb = tf.nn.l2_normalize(emb, dim=1)
    # emb = tf.contrib.layers.fully_connected(emb_ori, n_classes)
    emb_softmax = tf.nn.softmax(emb, axis=1)

    # x_test = tf.placeholder(tf.float32, [None, 512, None, 1], name='audio_test')
    # y_test = tf.placeholder(tf.int64, [None])


    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=emb, labels=y))


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(emb_softmax, 1), y_s), tf.float32))

    emb_test = tf.placeholder(tf.float32, [None, n_classes])

    acc_num = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(emb_test, 1), y_s), tf.float32))


    opt = tf.train.AdamOptimizer(0.001)
    trainer = opt.minimize(loss)

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    
    saver = tf.train.Saver(max_to_keep=3, var_list=var_list)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True


    true_item = 0
    all_item = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # step = sess.run(global_step)
        saver.restore(sess, ckpt_restore_file)

        counter = 0
        for i in range(100000):
            idx_train = 0
            idx_train_label = 0

            true_item = 0
            all_item  = 0
            while True:
                batch_test, idx_train, end_epoch = get_batch(test_audio, idx_train)
                batch_test_label, idx_train_label, end_epoch = get_label_batch(test_label, idx_train_label)

                # loss_val, acc_val, acc_num_val, emb_softmax_val = sess.run([loss, accuracy, acc_num, emb], feed_dict={x: batch_test, y_s: batch_test_label})

                # # Chunk test---------------------
                # emb_vals = []
                # each_chunk = 50
                # for each_audio in batch_test:
                #     start_chunk = 0
                #     testing_chunk = []
                #     while(start_chunk + 400 <= each_audio.shape[1]):
                #         testing_chunk.append(each_audio[:, start_chunk:start_chunk+400])
                #         start_chunk += each_chunk
                #     emb_val = sess.run(emb_softmax, feed_dict={x: testing_chunk})
                #     # pred = tf.argmax(tf.reduce_sum(emb_val, 0), 1)
                #     emb_val = np.sum(emb_val, 0)
                #     # print(emb_val)
                #     emb_vals.append(emb_val)

                # emb_vals = np.array(emb_vals)
                # acc_num_val = sess.run(acc_num, feed_dict={emb_test: emb_vals, y_s: batch_test_label})
                # true_item += acc_num_val
                # all_item += 32
                # print(true_item/all_item)

                # 3s Test-------------
                emb_softmax_val = sess.run(emb_softmax, feed_dict={x: batch_test})
                # print(emb_softmax_val)


                acc_num_val     = sess.run(acc_num,     feed_dict={emb_test: emb_softmax_val, y_s: batch_test_label})

                true_item += acc_num_val
                all_item  += batch_size
                print('Test batch acc: ', true_item / all_item) 

                counter += 1

                # # if counter % 100 == 0:
                # print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val, 'a_acc:', true_item/all_item)

                # x_val, emb_val = sess.run([x, emb_ori], feed_dict={x: batch_test})
                # print(emb_val)

                if end_epoch:
                    print('end epoch: ', i)
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--voxceleb_wav_dir', default='/data/ChuyuanXiong/up/wav/', required=True)
    parser_train.add_argument('--vox_split_txt_file', default='utils/vox1_split_backup.txt', required=True)
    parser_train.add_argument('--batch_size', default=32, type=int, required=True)
    parser_train.add_argument('--lr', default=0.001, type=float, required=True)
    parser_train.add_argument('--ckpt_save_dir', default='/data/ChuyuanXiong/backup/speaker_real308_ckpt', required=True)
    parser_train.add_argument('--random_seed', default=100, type=int, required=False)
    parser_train.add_argument('--epoch', default=10000, type=int, required=False)
    parser_train.set_defaults(func=train)

    # Test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--voxceleb_wav_dir', default='/data/ChuyuanXiong/up/wav/', required=True)
    parser_test.add_argument('--vox_split_txt_file', default='utils/vox1_split_backup.txt', required=True)
    parser_test.add_argument('--batch_size', default=32, type=int, required=True)
    parser_test.add_argument('--lr', default=0.001, type=float, required=True)
    parser_test.add_argument('--ckpt_restore_file', default='/data/ChuyuanXiong/backup/triplet_backup2/Speaker_vox_iter_51500.ckpt', required=True)
    parser_test.add_argument('--random_seed', default=100, type=int, required=False)
    parser_test.set_defaults(func=test)
    opt = parser.parse_args()
    opt.func(opt)

