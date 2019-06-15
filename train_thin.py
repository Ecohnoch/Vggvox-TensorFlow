from __future__ import unicode_literals

from functools import reduce
import tensorflow as tf
import numpy as np

import os
import time
import random
import warnings
import argparse
import librosa
import threading
import scipy.io.wavfile

from utils.my_utils import audio_data_extracted_5994
from utils.my_utils import set_mp
from thin_resnet import resnet34
from netVLAD import VLAD_pooling

regularizer = tf.contrib.layers.l2_regularizer(5e-4)




def load_wav(vid_path, sr, mode='train'):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr
    if mode == 'train':
        extended_wav = np.append(wav, wav)
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        extended_wav = np.append(wav, wav[::-1])
        return extended_wav

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


def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T


def load_data(path, win_length=400, sr=16000, hop_length=160, n_fft=512, spec_len=250, mode='train'):
    wav = load_wav(path, sr=sr, mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    if mode == 'train':
        randtime = np.random.randint(0, time-spec_len)
        spec_mag = mag_T[:, randtime:randtime+spec_len]
    else:
        spec_mag = mag_T
    # preprocessing, subtract mean, divided by time-wise var
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    res = (spec_mag - mu) / (std + 1e-5)
    res = np.array(res)
    res = np.expand_dims(res, -1)
    return res



def load_wave_list(wave_file_list, mp_pooler=None):
    try:
        ans = [mp_pooler.apply_async(load_data, args=(ID, 400, 16000, 160, 512, 250)) for ID in  wave_file_list]
        ans = np.array([p.get() for p in ans])
        return ans
    except:
        print('****** Error')
        print(wave_file_list)
        return ans

def train_ori(opt):
    # lr            = opt.lr
    rs            = opt.random_seed
    batch_size    = opt.batch_size
    wav_dir       = opt.voxceleb_wav_dir
    split_file    = opt.vox_split_txt_file
    ckpt_save_dir = opt.ckpt_save_dir
    epoch_time    = opt.epoch
    avg           = 'netvlad'

    print('-------------Training Args-----------------')
    # print('--lr           :  ', lr)
    print('--rs           :  ', rs)
    print('--batch_size   :  ', batch_size)
    print('--wav_dir      :  ', wav_dir)
    print('--split_file   :  ', split_file)
    print('--ckpt_save_dir:  ', ckpt_save_dir)
    print('--epoch_time   :  ', epoch_time)
    print('--avg          :  ', avg)
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

    train_audio, train_label = audio_data_extracted_5994(split_file, wav_dir)
    np.random.seed(rs)
    np.random.shuffle(train_audio)
    np.random.seed(rs)
    np.random.shuffle(train_label)
    print('--dataset shape:  ', train_audio.shape, train_label.shape)
    print('-------------------------------------------')

    def step_decay(epoch, epoch_time):
        '''
        The learning rate begins at 10^initial_power,
        and decreases by a factor of 10 every step epochs.
        '''
        half_epoch = epoch_time // 2
        stage1, stage2, stage3 = int(half_epoch * 0.5), int(half_epoch * 0.8), half_epoch
        stage4 = stage3 + stage1
        stage5 = stage4 + (stage2 - stage1)
        stage6 = epoch_time

        milestone = [stage1, stage2, stage3, stage4, stage5, stage6]
        gamma = [1.0, 0.1, 0.01, 1.0, 0.1, 0.01]

        lr = 0.005
        init_lr = 0.001
        stage = len(milestone)
        for s in range(stage):
            if epoch < milestone[s]:
                lr = init_lr * gamma[s]
                break
        # print('Learning rate for epoch {} is {}.'.format(epoch + 1, lr))
        return np.float(lr)

    mp_pooler = set_mp(processes=12)
    print('mp_pooler: ', 12)
    def get_batch(dataset, start, batch_size=32):
        end = (start + batch_size) % dataset.shape[0]
        if end > start:
            return load_wave_list(dataset[start:end], mp_pooler), end, False
        
        sub_dataset = np.hstack((dataset[start:], dataset[:end]))
        return load_wave_list(sub_dataset, mp_pooler), end, True

    def get_label_batch(dataset, start, batch_size=32):
        end = (start + batch_size) % dataset.shape[0]
        if end > start:
            return dataset[start:end], end, False

        return np.hstack((dataset[start:], dataset[:end])), end, True


    x = tf.placeholder(tf.float32, [None, 257, 250, 1], name='audio_input')
    y_s = tf.placeholder(tf.int64, [None])
    lr  = tf.placeholder(tf.float32, [], name='learning_rate')
    
    y = tf.one_hot(y_s, n_classes, axis=-1)
    emb_ori = resnet34(x, is_training=True, kernel_initializer=tf.orthogonal_initializer())
    fc1 = tf.layers.conv2d(emb_ori, filters=512, kernel_size=[7,1], strides=[1,1], padding='SAME', activation=tf.nn.relu, name='fc_block1_conv')    
    if avg == 'netvlad':
        x_center = tf.layers.conv2d(emb_ori, filters=10, kernel_size=[7,1], strides=[1,1], use_bias=True, padding='SAME', name='x_center_conv')
        pooling_output = VLAD_pooling(fc1, x_center, k_centers=10)
    else:
        pooling_output = tf.reduce_mean(fc1, [1, 2], name='gap')

    fc2 = tf.layers.dense(pooling_output, 512, activation=tf.nn.relu , name='fc_block2_conv')
    emb = tf.layers.dense(fc2, n_classes, name='fc_prob_conv')

    # emb = tf.contrib.layers.fully_connected(emb_ori, n_classes, activation_fn=None)
    emb_softmax = tf.nn.softmax(emb, axis=1)


    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=emb, labels=y))
    conv_var = [var for var in tf.trainable_variables() if 'conv' in var.name]
    l2_loss = tf.add_n([tf.nn.l2_loss(var) for var in conv_var])
    loss = loss + 1e-4 * l2_loss


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(emb_softmax, 1), y_s), tf.float32))

    saver = tf.train.Saver(max_to_keep=3) 


    total_params = 0
    for variable in tf.global_variables():
        shape = variable.get_shape()
        variable_params = 1
        for dim in shape:
            variable_params *= dim.value
        total_params += variable_params
    print('total params: ', total_params)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainer = tf.train.AdamOptimizer(lr).minimize(loss)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    now_lr = step_decay(0, epoch_time)
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, '/home/ruc_tliu_1/Vggvox-TensorFlow/ckpt/Speaker_vox_iter_6500.ckpt')

        counter = 0
        cur_ckpt_file = ''
        start_time = time.time()

        for i in range(epoch_time):
            idx_train = 0
            idx_train_label = 0
            while True:
                
                batch_train, idx_train, end_epoch = get_batch(train_audio, idx_train, batch_size=batch_size)
                batch_train_label, idx_train_label, end_epoch = get_label_batch(train_label, idx_train_label, batch_size=batch_size)
                batch_train = np.array(batch_train)

                _, loss_val, acc_val, emb_softmax_val = sess.run([trainer, loss, accuracy, emb], feed_dict={x: batch_train, y_s: batch_train_label, lr: now_lr})

                counter += 1

                if counter % 100 == 0:
                    print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val, 'now_lr: ', now_lr, 'time: %.4f'% (time.time() - start_time))
                    start_time = time.time()
                if counter % 500 == 0:
                    filename = 'Speaker_vox_iter_{:d}'.format(counter) + '.ckpt'
                    filename = os.path.join(ckpt_save_dir, filename)
                    cur_ckpt_file = filename
                    saver.save(sess, filename)
                if counter % 1000 == 0:
                    # veri(cur_ckpt_file)
                    pass
                if end_epoch:
                    now_lr = step_decay(i + 1, epoch_time)
                    print('end epoch ', i, 'now_lr: ', now_lr)
                    break

def parse_function(example_proto):
    features = {
        'audio_raw': tf.FixedLenFeature([], tf.string),
        'label'    : tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(example_proto, features)
    audio_file = tf.decode_raw(features['audio_raw'], tf.float32)
    audio_file = tf.reshape(audio_file, [257, 250, 1])
    label      = tf.cast(features['label'], tf.int64)
    return audio_file, label

def train(opt):
    lr            = opt.lr
    # rs            = opt.random_seed
    batch_size    = opt.batch_size
    # wav_dir       = opt.voxceleb_wav_dir
    # split_file    = opt.vox_split_txt_file
    ckpt_save_dir = opt.ckpt_save_dir
    epoch_time    = opt.epoch
    avg           = 'netvlad'
    # avg           = 'avgpool'
    trans_file    = '/data/ChuyuanXiong/up/voxceleb2_tfrecord/tran.tfrecords'

    print('-------------Training Args-----------------')
    print('--lr           :  ', lr)
    # print('--rs           :  ', rs)
    print('--batch_size   :  ', batch_size)
    # print('--wav_dir      :  ', wav_dir)
    # print('--split_file   :  ', split_file)
    print('--ckpt_save_dir:  ', ckpt_save_dir)
    print('--epoch_time   :  ', epoch_time)
    print('--trans_file   :  ', trans_file)
    # print('-------------------------------------------')

    n_classes = 5994

    # train_audio, train_label = audio_data_extracted_5994(split_file, wav_dir)
    # test_audio, test_label = audio_data_extracted_5994('utils/voxlb2_val.txt', wav_dir)
    # test_audio = test_audio[:64]
    # test_label = test_label[:64]

    # np.random.seed(rs)
    # np.random.shuffle(train_audio)
    # np.random.seed(rs)
    # np.random.shuffle(train_label)

    # print('--dataset shape:  ', train_audio.shape, train_label.shape)
    dataset = tf.data.TFRecordDataset(trans_file)
    dataset = dataset.map(parse_function)
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()
    next_element = iterator.get_next()
    print('-------------------------------------------')



    x = tf.placeholder(tf.float32, [None, 257, 250, 1], name='audio_input')
    y_s = tf.placeholder(tf.int64, [None])
    y = tf.one_hot(y_s, n_classes, axis=-1)

    emb_ori = resnet34(x, is_training=True)
    fc1 = tf.layers.conv2d(emb_ori, filters=512, kernel_size=[7,1], strides=[1,1], padding='SAME', activation=tf.nn.relu, name='fc_block1')
    # print(fc1)
    # fc1 = tf.contrib.layers.flatten(fc1)
    # print(fc1)

    # AVGPooling
    # TODO
    # NetVLAD
    if avg == 'netvlad':
        x_center = tf.layers.conv2d(emb_ori, filters=10, kernel_size=[7,1], strides=[1,1], use_bias=True, padding='SAME', name='x_center')
        pooling_output = VLAD_pooling(fc1, x_center, k_centers=10)
    else:
        pooling_output = tf.reduce_mean(fc1, [1, 2])

    print('pooling output shape: ', pooling_output)
    fc2 = tf.layers.dense(pooling_output, 512, activation=tf.nn.relu , name='fc_block2')

    emb = tf.layers.dense(fc2, n_classes, name='fc_prob', use_bias=False)

    # emb = tf.contrib.layers.fully_connected(emb_ori, n_classes, activation_fn=None)
    emb_softmax = tf.nn.softmax(emb, axis=1)


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=emb, labels=y_s))


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(emb_softmax, 1), y_s), tf.float32))

    saver = tf.train.Saver(max_to_keep=3, var_list=tf.global_variables()) 

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        trainer = tf.train.AdamOptimizer(lr).minimize(loss)

    

    # coord = tf.train.Coordinator()
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        # saver.restore(sess, '/data/ChuyuanXiong/backup/speaker_real327_ckpt/Speaker_vox_iter_6600.ckpt')
        # saver.restore(sess, '/data/ChuyuanXiong/backup/speaker_real413_ckpt/Speaker_vox_iter_38500.ckpt')
        # saver.restore(sess, '/data/ChuyuanXiong/backup/speaker_real413_ckpt/Speaker_vox_iter_4200.ckpt')
        saver.restore(sess, '/data/ChuyuanXiong/backup/Speaker_vox_iter_91000.ckpt')
        counter = 0

        for i in range(epoch_time):
            sess.run(iterator.initializer)
            while True:
                try:
                    batch_train, label_train = sess.run(next_element)
                    _, loss_val, acc_val, emb_softmax_val = sess.run([trainer, loss, accuracy, emb], feed_dict={x: batch_train, y_s: label_train})

                    counter += 1

                    if counter % 100 == 0:
                        print('counter: ', counter, 'loss_val', loss_val, 'acc: ', acc_val)
                        filename = 'Speaker_vox_iter_{:d}'.format(counter) + '.ckpt'
                        filename = os.path.join(ckpt_save_dir, filename)
                        saver.save(sess, filename)
                except tf.errors.OutOfRangeError:
                    print('end epoch ', i)
                    break


def test(opt):
    # rs                = opt.random_seed
    batch_size        = opt.batch_size
    wav_dir           = opt.voxceleb_wav_dir
    split_file        = opt.vox_split_txt_file
    ckpt_restore_file = opt.ckpt_restore_file

    print('-------------Training Args-----------------')
    # print('--rs               :  ', rs)
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
    print('--dataset shape:  ', test_audio.shape, test_label.shape)
    print('-------------------------------------------')

    def get_batch(dataset, start, batch_size=32):
        end = (start + batch_size) % len(dataset)
        if end > start:
            sub_dataset1 = load_wave_list(dataset[start:end])
            sub_dataset2 = load_wave_list(dataset[start:end])
            sub_dataset3 = load_wave_list(dataset[start:end])
            sub_dataset4 = load_wave_list(dataset[start:end])
            sub_dataset5 = load_wave_list(dataset[start:end])
            sub_dataset6 = load_wave_list(dataset[start:end])
            sub_dataset7 = load_wave_list(dataset[start:end])
            sub_dataset8 = load_wave_list(dataset[start:end])
            sub_dataset9 = load_wave_list(dataset[start:end])
            sub_dataset10 = load_wave_list(dataset[start:end])
            return [sub_dataset1, sub_dataset2, sub_dataset3, sub_dataset4, sub_dataset5, sub_dataset6, sub_dataset7, sub_dataset8, sub_dataset9, sub_dataset10], end, False

        sub_dataset = np.hstack((dataset[start:], dataset[:end]))
        sub_dataset1 = load_wave_list(sub_dataset)
        sub_dataset2 = load_wave_list(sub_dataset)
        sub_dataset3 = load_wave_list(sub_dataset)
        sub_dataset4 = load_wave_list(sub_dataset)
        sub_dataset5 = load_wave_list(sub_dataset)
        sub_dataset6 = load_wave_list(sub_dataset)
        sub_dataset7 = load_wave_list(sub_dataset)
        sub_dataset8 = load_wave_list(sub_dataset)
        sub_dataset9 = load_wave_list(sub_dataset)
        sub_dataset10 = load_wave_list(sub_dataset)
        return [sub_dataset1, sub_dataset2, sub_dataset3, sub_dataset4, sub_dataset5, sub_dataset6, sub_dataset7, sub_dataset8, sub_dataset9, sub_dataset10], end, True

    def get_label_batch(dataset, start, batch_size=32):
        end = (start + batch_size) % len(dataset)
        if end > start:
            return dataset[start:end], end, False
        return np.hstack((dataset[start:], dataset[:end])), end, True


    x = tf.placeholder(tf.float32, [None, 257, 250, 1], name='audio_input')
    y_s = tf.placeholder(tf.int64, [None])
    

    # with tf.device('/cpu:0'):
    #     q = tf.FIFOQueue(batch_size*3, [tf.float32, tf.int64], shapes=[[512, 300, 1], []])
    #     enqueue_op = q.enqueue_many([x, y_s])
    #     x_b, y_b = q.dequeue_many(batch_size)

    y = tf.one_hot(y_s, n_classes, axis=-1)
    emb_ori = voicenet(x, is_training=False)
    emb = tf.layers.dense(emb_ori, n_classes, name='fc_prob')
    # emb = tf.contrib.layers.fully_connected(emb_ori, n_classes, activation_fn=None)

    # emb = tf.nn.l2_normalize(emb, dim=1)
    # emb = tf.contrib.layers.fully_connected(emb_ori, n_classes)
    emb_softmax = tf.nn.softmax(emb, axis=1)

    # x_test = tf.placeholder(tf.float32, [None, 512, None, 1], name='audio_test')
    # y_test = tf.placeholder(tf.int64, [None])


    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(logits=emb, labels=y))


    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(emb_softmax, 1), y_s), tf.float32))

    test_322 = tf.placeholder(tf.float32, [None, n_classes])

    test_322_softmax = tf.nn.softmax(test_322, axis=1)
    accuracy_322 = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(test_322_softmax, 1), y_s), tf.float32))

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

    # coord = tf.train.Coordinator()

    true_item = 0
    all_item = 0
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # step = sess.run(global_step)
        saver.restore(sess, ckpt_restore_file)
        print(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

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

                # 3s Test-------------
                acc_test = sess.run(accuracy, feed_dict={x: batch_test[0], y_s: batch_test_label})
                # print(emb_softmax_val)

                print(acc_test)

                counter += 1

                if end_epoch:
                    print('end epoch: ', i)
                    break

def calculate_eer(y, y_score):
    # y denotes groundtruth scores,
    # y_score denotes the prediction scores.
    from scipy.optimize import brentq
    from sklearn.metrics import roc_curve
    from scipy.interpolate import interp1d

    fpr, tpr, thresholds = roc_curve(y, y_score, pos_label=1)
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer, thresh

def veri(ckpt_file='/data/ChuyuanXiong/backup/speaker_real413_ckpt/Speaker_vox_iter_27700.ckpt', reuse=True):
    wav_dir = '/data/ChuyuanXiong/up/voxceleb1/wav/'
    verify_list = np.loadtxt('utils/voxceleb1_veri_test.txt', str)
    verify_lb = np.array([int(i[0]) for i in verify_list])
    list1 = np.array([os.path.join(wav_dir, i[1]) for i in verify_list])
    list2 = np.array([os.path.join(wav_dir, i[2]) for i in verify_list])
    total_list = np.concatenate((list1, list2))
    unique_list= np.unique(total_list)
    avg = 'netvlad'


    x = tf.placeholder(tf.float32, [None, 257, None, 1], name='audio_input')
    y_s = tf.placeholder(tf.int64, [None])

    y = tf.one_hot(y_s, 5994, axis=-1)
    emb_ori = resnet34(x, is_training=False, reuse=reuse)
    fc1 = tf.layers.conv2d(emb_ori, filters=512, kernel_size=[7,1], strides=[1,1], padding='SAME', activation=tf.nn.relu, reuse=reuse, name='fc_block1')
    # print(fc1)
    # fc1 = tf.contrib.layers.flatten(fc1)
    # print(fc1)

    # AVGPooling
    # TODO
    # NetVLAD

    if avg == 'netvlad':
        x_center = tf.layers.conv2d(emb_ori, filters=10, kernel_size=[7,1], strides=[1,1], use_bias=True, padding='SAME', name='x_center')
        pooling_output = VLAD_pooling(fc1, x_center, k_centers=10)
    else:
        pooling_output = tf.reduce_mean(fc1, [1, 2], name='gap')


    print('pooling output shape: ', pooling_output)
    fc2 = tf.layers.dense(pooling_output, 512, activation=tf.nn.relu, reuse=reuse , name='fc_block2')
    fc2 = tf.nn.l2_normalize(fc2, 1)

    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(max_to_keep=3, var_list=var_list)

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        # step = sess.run(global_step)
        saver.restore(sess, ckpt_file)
        # saver.restore(sess, '/data/ChuyuanXiong/backup/speaker_real413_ckpt/Speaker_vox_iter_59700.ckpt')
        total_length = len(unique_list)
        feats, scores, labels = [], [], []
        for c, ID in enumerate(unique_list):
            if c % 50 == 0: 
                print('Finish extracting features for {}/{}th wav.'.format(c, total_length))
            specs = load_data(ID, mode='eval')
            emb_val = sess.run(fc2, feed_dict={x:[specs]})
            avg_emb = np.mean(emb_val, 0)
            # print(avg_emb.shape)
            feats.append([avg_emb])
        feats = np.array(feats)
        for c, (p1, p2) in enumerate(zip(list1, list2)):
            ind1 = np.where(unique_list == p1)[0][0]
            ind2 = np.where(unique_list == p2)[0][0]
            v1 = feats[ind1, 0]
            v2 = feats[ind2, 0]

            scores += [np.sum(v1*v2)]
            labels += [verify_lb[c]]
        scores = np.array(scores)
        labels = np.array(labels)

        eer, thresh = calculate_eer(labels, scores)
        print(eer)

if __name__ == '__main__':
    # veri(ckpt_file='/data/ChuyuanXiong/params/speaker_515_ckpt/Speaker_vox_iter_79500.ckpt', reuse=False)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--voxceleb_wav_dir', default='/data/ChuyuanXiong/up/wav/', required=True)
    parser_train.add_argument('--vox_split_txt_file', default='utils/voxlb2_train.txt', required=True)
    parser_train.add_argument('--batch_size', default=32, type=int, required=True)
    parser_train.add_argument('--lr', default=0.001, type=float, required=True)
    parser_train.add_argument('--ckpt_save_dir', default='/data/ChuyuanXiong/backup/speaker_real419_ckpt', required=True)
    parser_train.add_argument('--random_seed', default=100, type=int, required=False)
    parser_train.add_argument('--epoch', default=10000, type=int, required=False)
    parser_train.set_defaults(func=train_ori)

    # Test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--voxceleb_wav_dir', default='/data/ChuyuanXiong/up/wav/', required=True)
    parser_test.add_argument('--vox_split_txt_file', default='utils/vox1_split_backup.txt', required=True)
    parser_test.add_argument('--batch_size', default=32, type=int, required=True)
    parser_test.add_argument('--ckpt_restore_file', default='/data/ChuyuanXiong/backup/triplet_backup2/Speaker_vox_iter_51500.ckpt', required=True)
    parser_test.add_argument('--random_seed', default=100, type=int, required=False)
    parser_test.set_defaults(func=test)

    # Veri
    # parser_veri = subparsers.add_parser('veri')
    # parser_veri.set_defaults(func=veri)

    opt = parser.parse_args()
    opt.func(opt)
    # veri()

