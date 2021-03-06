from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import cv2
import numpy as np
from Queue import Queue
from threading import Thread as Process
# import time

from utils import *


class DataSet(object):
    """TextDataSet
    process text input file dataset 
    text file format:
      image_path
    """

    def __init__(self, common_params=None, dataset_params=None, training=True, shuffle=True):
        """
        Args:
          common_params: A dict
          dataset_params: A dict
        """
        if common_params:
            self.image_size = int(common_params['image_size'])
            self.batch_size = int(common_params['batch_size'])
            self.is_gan = True if common_params['is_gan'] == '1' else False
            self.is_rgb = True if common_params['is_rgb'] == '1' else False
          
        if dataset_params:
            self.data_path = str(dataset_params['path'])
            self.thread_num = int(int(dataset_params['thread_num']) / 2)
            self.thread_num2 = int(int(dataset_params['thread_num']) / 2)
            self.c313 = True if dataset_params['c313'] == '1' else False
            self.cond_l = True if dataset_params['cond_l'] == '1' else False
            self.gamma = float(dataset_params['gamma'])
            print('Gamma in prior smoothing: {}.'.format(self.gamma))
            self.augment = dataset_params['augment'] == '1'
            if self.augment:
                print('Random lighting augmentation used.')

        self.training = training

        if not self.training:
            self.data_path = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val.txt'
            self.thread_num = 2
            self.thread_num2 = 2
            
        # record and image_label queue
        self.record_queue = Queue(maxsize=30000)
        self.image_queue = Queue(maxsize=15000)

        self.batch_queue = Queue(maxsize=300)

        self.record_list = []  
        self.prior_path = '/srv/glusterfs/xieya/prior/313_ab_1.npy' if self.cond_l else './resources/prior_probs_smoothed.npy'

        # filling the record_list
        input_file = open(self.data_path, 'r')

        for line in input_file:
            line = line.strip()
            self.record_list.append(line)

        self.record_point = 0
        self.record_number = len(self.record_list)
        self.shuffle = shuffle

        self.num_batch_per_epoch = int(self.record_number / self.batch_size)

        t_record_producer = Process(target=self.record_producer)
        t_record_producer.daemon = True
        t_record_producer.start()

        for i in range(self.thread_num):
            t = Process(target=self.record_customer)
            t.daemon = True
            t.start()

        for i in range(self.thread_num2):
            t = Process(target=self.image_customer)
            t.daemon = True
            t.start()

    def record_producer(self):
        """record_queue's processor
        """
        while True:
            if self.record_point % self.record_number == 0:
                if self.shuffle:
                    random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def image_process(self, image):
        """record process 
        Args: record 
        Returns:
          image: 3-D ndarray
        """
        h = image.shape[0]
        w = image.shape[1]

        # Random flip
        mirror = np.random.randint(0, 2)
        if mirror and self.training:
            image = np.fliplr(image)

        if w > h:
            image = cv2.resize(image, (int(self.image_size * w / h), self.image_size))
            crop_start = np.random.randint(0, int(self.image_size * w / h) - self.image_size + 1)
            image = image[:, crop_start:crop_start + self.image_size, :]
        else:
            image = cv2.resize(image, (self.image_size, int(self.image_size * h / w)))
            crop_start = np.random.randint(0, int(self.image_size * h / w) - self.image_size + 1)
            image = image[crop_start:crop_start + self.image_size, :, :]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def record_customer(self):
        """record queue's customer 
        """
        while True:
            item = self.record_queue.get()
            out = cv2.imread(item)
            if out is None:
                print(item, os.path.isfile(item))
                continue
            if len(out.shape) == 3 and out.shape[2] == 3:
                self.image_queue.put(out)

    def image_customer(self):
        while True:
            images = []
            for i in range(self.batch_size):
                image = self.image_queue.get()
                image = self.image_process(image)
                images.append(image)
            images = np.asarray(images, dtype=np.uint8)

            self.batch_queue.put(preprocess(images, c313=self.c313, is_gan=self.is_gan, 
                                 is_rgb=self.is_rgb, cond_l=self.cond_l, prior_path=self.prior_path, gamma=self.gamma, augment=self.augment))

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
        """
        # print(self.record_queue.qsize(), self.image_queue.qsize(), self.batch_queue.qsize())
        return self.batch_queue.get()
