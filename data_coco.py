from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import cv2
import numpy as np
from Queue import Queue
from threading import Thread as Process
import time

from utils import *


class DataSet(object):
    """TextDataSet
    process text input file dataset 
    text file format:
      image_path
    """

    def __init__(self, common_params=None, dataset_params=None, train=True, with_ab=False, shuffle=True):
        """
        Args:
          common_params: A dict
          dataset_params: A dict
        """
        if dataset_params:
            self.data_path = str(dataset_params['path'])
            self.thread_num = int(dataset_params['thread_num'])
            self.prior_path = str(dataset_params['prior_path']) if 'prior_path' in dataset_params else './resources/prior_probs_smoothed.npy'

        if common_params:
            self.batch_size = int(common_params['batch_size'])
            self.with_caption = True if common_params['with_caption'] == '1' else False

        self.training = train
        self.with_ab = with_ab

        # record and image_label queue
        self.record_queue = Queue(maxsize=15000)
        self.batch_queue = Queue(maxsize=300)

        hf = h5py.File(self.data_path, 'r')

        if self.training:
            self.train_origs = hf['train_ims']            
            self.train_words = hf['train_words']                                         
            self.train_lengths = hf['train_length'] 
        else:
            self.train_origs = hf['val_ims']            
            self.train_words = hf['val_words']                                         
            self.train_lengths = hf['val_length'] 
            self.thread_num = 4

        self.record_point = 0
        self.record_number = len(self.train_origs)
        self.shuffle = shuffle
        self.idx = [i for i in xrange(self.record_number)]
        self.idx = np.asarray(self.idx)
        # self.idx = np.random.permutation(self.record_number)

        t_record_producer = Process(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(self.thread_num):
            t = Process(target=self.image_customer)
            t.daemon = True
            t.start()

    def record_producer(self):
        """record_queue's processor
        """
        while True:
            if self.record_point % self.record_number == 0:
                if shuffle:
                    self.idx = np.random.permutation(self.record_number)
                self.record_point = 0
            i = self.idx[self.record_point]
            self.record_queue.put(i)
            self.record_point += 1

    def image_customer(self):
        while True:
            images = []
            captions = []
            lens = []
            count = 0
            while count < self.batch_size:
                idx = self.record_queue.get()
                image = self.train_origs[idx]
                caption = self.train_words[idx]
                length = self.train_lengths[idx]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 1:]
                if is_grayscale(ab):
                    continue
                # Augmentation.
                mirror = np.random.randint(0, 2)
                if mirror:
                    image = np.fliplr(image)
                images.append(image)
                captions.append(caption)
                lens.append(length)
                count += 1
            images = np.asarray(images, dtype=np.uint8)
            captions = np.asarray(captions, dtype=np.int32)
            lens = np.asarray(lens, dtype=np.int32)
            l, gt, prior, ab = preprocess(images, c313=True, prior_path=self.prior_path, mask_gray=(not self.with_caption))
            if self.with_ab:
                self.batch_queue.put((l, gt, prior, captions, lens, ab))
            else:    
                self.batch_queue.put((l, gt, prior, captions, lens))

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
        """
        # print(self.record_queue.qsize(), self.batch_queue.qsize())
        return self.batch_queue.get()
