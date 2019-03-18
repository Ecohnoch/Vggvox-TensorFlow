# -- coding: utf-8 --
import os
import cv2
import csv
import sys
import random
import numpy as np 

import utils.read as read

from scipy.io import wavfile
from python_speech_features import fbank, delta

import skimage
import skimage.io
import skimage.transform


def read_data_from_csv(filePath):
	with open(filePath) as f:
		reader = csv.reader(f)
		ans = list(reader)
		return ans

def get_meta_info(csvFilePath):
	info = {}
	csv_data = read_data_from_csv(csvFilePath)[1:]
	name2id = {}
	for i in csv_data:
		j = i[0].split('\t')
		info[j[0]] = {}
		info[j[0]]['name'] = j[1]
		name2id[j[1]] = j[0]
		info[j[0]]['gender'] = j[2]
		info[j[0]]['nationality'] = j[3]
	# length 1251, 1251
	return info, name2id

def id2name(info, idx):
	return info[idx]['name']


def audio_data_extract(fileName):
	reader = read.reader(fileName, fileType='txt', dataType='str', dataGap=' ', headerGap=' ')
	reader.read_file()
	reader.data.insert(0, reader.header)
	all_data = reader.data
	

	#  path, label1(id), label2(hash)
	audio_train = []
	audio_veri  = []
	audio_test  = []
	for i in all_data:
		path   = i[1]
		labels = i[1].split('/')
		label1 = labels[0]
		label2 = labels[1]
		if i[0] == '1':
			audio_train.append([path, label1, label2])
		elif i[0] == '2':
			audio_veri.append([path, label1, label2])
		elif i[0] == '3':
			audio_test.append([path, label1, label2])

	return audio_train, audio_test, audio_veri
  
def positive_batch_pair(info, audio_dir, face_dir, audio_batch):
	# audio_dir = '/data/ChuyuanXiong/up/wav/'
	# face_dir  = '/data/ChuyuanXiong/up/unzippedFaces/'
	#

	batch_size = len(audio_batch)
	batch_pair = []
	for i in audio_batch:
		# i: [path, label1(id), label2(hash)]
		
		audio_file    = audio_dir + i[0]
		face_file_dir = face_dir  + id2name(info, i[1]) + '/1.6/' + i[2]
		face_file_list= os.listdir(face_file_dir)
		face_random   = random.sample(face_file_list, 1)[0]
		face_random_path = face_file_dir + '/' + face_random

		# [audio, face, label_id, label_ge, id, hash]
		batch_pair.append([audio_file, face_random_path, 1, 1,  i[1], i[2]])
	return np.array(batch_pair)

def positive_batch_gender_pair(info, name2id, audio_dir, face_dir, audio_batch):
	# audio_dir = '/data/ChuyuanXiong/up/wav/'
	# face_dir  = '/data/ChuyuanXiong/up/unzippedFaces/'
	#
	batch_size = len(audio_batch)
	batch_pair = []
	for i in audio_batch:
		# i: [path, label1(id), label2(hash)]
		audio_file = audio_dir + i[0]
		audio_gender = info[i[1]]['gender']

		face_file_id = random.sample(info.keys(), 1)[0]
		while info[face_file_id]['gender'] != audio_gender:
			face_file_id = random.sample(info.keys(), 1)[0]

		face_file_dir = face_dir + info[face_file_id]['name'] + '/1.6/'
		face_file_random_hash = random.sample(os.listdir(face_file_dir), 1)[0]
		face_file_dir = face_file_dir + face_file_random_hash + '/'
		face_file_random_name = random.sample(os.listdir(face_file_dir), 1)[0]
		face_file_path = face_file_dir + face_file_random_name

		# [audio, face, label, audio_gender, face_gender, id, hash]
		batch_pair.append([audio_file, face_file_path, 1, audio_gender, info[face_file_id]['gender'], i[1], i[2]])
	return np.array(batch_pair)

def negtive_batch_gender_pair(info, name2id, audio_dir, face_dir, audio_batch):
	# audio_dir = '/data/ChuyuanXiong/up/wav/'
	# face_dir  = '/data/ChuyuanXiong/up/unzippedFaces/'
	#
	batch_size = len(audio_batch)
	negtive_pair = []
	for i in audio_batch:
		# i: [path, label1(id), label2(hash)]
		audio_file = audio_dir + i[0]
		audio_gender = info[i[1]]['gender']

		face_file_id = random.sample(info.keys(), 1)[0]
		while info[face_file_id]['gender'] == audio_gender:
			face_file_id = random.sample(info.keys(), 1)[0]

		face_file_dir = face_dir + info[face_file_id]['name'] + '/1.6/'
		face_file_random_hash = random.sample(os.listdir(face_file_dir), 1)[0]
		face_file_dir = face_file_dir + face_file_random_hash + '/'
		face_file_random_name = random.sample(os.listdir(face_file_dir), 1)[0]
		face_file_path = face_file_dir + face_file_random_name

		# [audio, face, label, audio_gender, face_gender id, hash]
		negtive_pair.append([audio_file, face_file_path, 0, audio_gender,  info[face_file_id]['gender'], i[1], i[2]])
	return np.array(negtive_pair)

def negtive_batch_pair(info, name2id, audio_dir, face_dir, audio_batch):
	# audio_dir = '/data/ChuyuanXiong/up/wav/'
	# face_dir  = '/data/ChuyuanXiong/up/unzippedFaces/'
	#

	batch_size = len(audio_batch)
	batch_pair = []
	for i in audio_batch:
		# i: [path, label1(id), label2(hash)]
		
		audio_file    = audio_dir + i[0]
		audio_gender = info[i[1]]['gender']
		name = id2name(info, i[1])
		face_file_names = os.listdir(face_dir)
		face_file_names.remove(name)
		face_random_name = random.sample(face_file_names, 1)[0]
		face_label1 = name2id[face_random_name]
		face_gender = info[face_label1]['gender']
		face_random_name = face_dir + face_random_name + '/1.6/'
		face_random_hash = os.listdir(face_random_name)
		face_random_hash = random.sample(face_random_hash, 1)[0]
		face_label2 = face_random_hash
		face_random_name = face_random_name + face_random_hash
		face_random_file = os.listdir(face_random_name)
		face_random_file = random.sample(face_random_file, 1)[0]
		face_random_file = face_random_name + '/' + face_random_file

		# check
		if i[1] == face_label1 or i[2] == face_label2:
			print("*** Error when negtive sampling.")

		label_ge = 0
		if audio_gender == face_gender:
			label_ge = 1

		# [audio, face, label_id, label_ge, id, hash]
		batch_pair.append([audio_file, face_random_file, 0, label_ge, i[1], i[2]])
	return np.array(batch_pair)


def face_path_to_array(facePath, save_size):
	img = cv2.imread(facePath)
	img = cv2.resize(img, (save_size, save_size))
	img = img - 127.5
	img = img / 127.5
	return img.astype(np.float32)

def face_path_to_array_312(facePath, save_size):
    img = skimage.io.imread(facePath)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (save_size, save_size))
    return resized_img

def face_path_list_to_array(facePathList, save_size):
	ans = []
	for i in facePathList:
		img = face_path_to_array_312(i, save_size)
		ans.append(img)

	return np.array(ans)


# synset = [l.strip() for l in open('synset.txt').readlines()]


# returns image of shape [224, 224, 3]
# [height, width, depth]
def load_image(path):
    # load image
    img = skimage.io.imread(path)
    img = img / 255.0
    assert (0 <= img).all() and (img <= 1.0).all()
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = skimage.transform.resize(crop_img, (224, 224))
    return resized_img


def normalize_frames(m,epsilon=1e-12):
    return [(v - np.mean(v)) / max(np.std(v),epsilon) for v in m]


def get_fbank(signal, target_sample_rate):    
    filter_banks, energies = fbank(signal, samplerate=target_sample_rate, nfilt=64,nfft=int(target_sample_rate*0.025))
    filter_banks = normalize_frames(filter_banks)
    return np.array(filter_banks)


def read_wav(fname):
    fs, signal = wavfile.read(fname)
    return fs, signal


def audio_path_to_array(audioPath, seconds):
	fs, signal = read_wav(audioPath)
	if len(signal) < seconds*fs:
		signal = np.hstack((signal,[0]*(seconds*fs-len(signal))))
	start_sample = random.randint(0,len(signal)-seconds*fs)
	signal = signal[start_sample:start_sample+seconds*fs]
	return get_fbank(signal, fs)


def audio_path_list_to_array(audioPathList, seconds):
	ans = []
	for i in audioPathList:
		fbank = audio_path_to_array(i, seconds)
		ans.append(fbank)

	ans = np.array(ans)
	ans = ans.astype(np.float32)
	return ans