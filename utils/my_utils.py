# -- coding: utf-8 --
import os
# import cv2
import csv
import sys
import random
import numpy as np 

import utils.read as read

from scipy.io import wavfile
# from python_speech_features import fbank, delta

def set_mp(processes=8):
    import multiprocessing as mp

    def init_worker():
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    global pool
    try:
        pool.terminate()
    except:
        pass

    if processes:
        pool = mp.Pool(processes=processes, initializer=init_worker)
    else:
        pool = None
    return pool

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
	
def audio_data_extracted_5994(fileName, rootdir):
    '''
    filename:  voxlb2_train.txt,   voxlb2_val.txt
    rootdir :  /data/ChuyuanXiong/backup/dev
    '''
    train_list = np.loadtxt(fileName, str)
    train_file_list = np.array([os.path.join(rootdir, i[0]) for i in train_list])
    train_label_list= np.array([int(i[1]) for i in train_list])
    return train_file_list, train_label_list

