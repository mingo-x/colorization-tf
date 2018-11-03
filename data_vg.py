from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import random
import cv2
import numpy as np
from Queue import Queue
from threading import Thread as Process
#from multiprocessing import Process,Queue
import time

from utils import *

from skimage.io import imread
from skimage.transform import resize


class DataSet(object):
    """TextDataSet
    process text input file dataset 
    text file format:
      image_path
    """

    def __init__(self, common_params=None, dataset_params=None, train=True, shuffle=True, with_ab=False):
        """
        Args:
          common_params: A dict
          dataset_params: A dict
        """
        if common_params:
            self.image_size = int(common_params['image_size'])
            self.batch_size = int(common_params['batch_size'])
          
        if dataset_params:
            self.thread_num = int(int(dataset_params['thread_num']) / 2)
            self.thread_num2 = int(int(dataset_params['thread_num']) / 2)
            self.c313 = True if dataset_params['c313'] == '1' else False

        if train:
            self.split_file = '/srv/glusterfs/xieya/data/visual_genome/train.txt'
        else:
            self.split_file = '/srv/glusterfs/xieya/data/visual_genome/val.txt'
            self.thread_num = 2
            self.thread_num2 = 2
        
        self.data_dir = '/srv/glusterfs/xieya/data/visual_genome/VG_100K_224'
        self.regions = json.load(open(os.path.join('/srv/glusterfs/xieya/data/visual_genome/224_filtered_region_descriptions.json'), 'r'))
        # record and image_label queue
        self.record_queue = Queue(maxsize=30000)
        self.image_queue = Queue(maxsize=15000)
        self.batch_queue = Queue(maxsize=300)

        self.record_list = []  
        self.prior_path = './resources/prior_probs_smoothed.npy'

        # filling the record_list
        with open(self.split_file, 'r') as fin:
            for line in fin:
                img_id, img_idx, reg_num = line.strip().split(' ')
                img_idx = int(img_idx)
                img_path = os.path.join(self.data_dir, '{}.jpg'.format(img_id))
                regs = self.regions[img_idx]['regions']
                for reg_idx in xrange(int(reg_num)):
                    self.record_list.append((img_path, regs[reg_idx]))

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

    def image_process(self, image, reg):
        """record process 
        Args: record 
        Returns:
          image: 3-D ndarray
          bbox
        """
        h = image.shape[0]
        w = image.shape[1]
        bbox = np.ones((h,w, 1))
        reg_x = reg['x']
        reg_y = reg['y']
        reg_w = reg['width']
        reg_h = reg['height']

        mirror = np.random.randint(0, 2)
        if mirror and self.train:
            image = np.fliplr(image)
            # Flip bbox.
            reg_x = w - reg_x - reg_w

        if w > h:
            # Assume img_size == 224
            # image = cv2.resize(image, (int(self.image_size * w / h), self.image_size))
            crop_start = np.random.randint(0, min(int(self.image_size * w / h) - self.image_size + 1, reg_x + 1))
            image = image[:, crop_start: crop_start + self.image_size, :]
            reg_x -= crop_start
        else:
            # image = cv2.resize(image, (self.image_size, int(self.image_size * h / w)))
            crop_start = np.random.randint(0, min(int(self.image_size * h / w) - self.image_size + 1, reg_y + 1))
            image = image[crop_start:crop_start + self.image_size, :, :]
            reg_y -= crop_start
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bbox[reg_y: reg_y + reg_h, reg_x: reg_x + reg_w] = 3.  # Weight 3 for in-box pixels.

        return image, bbox

    def record_customer(self):
        """record queue's customer 
        """
        while True:
            img_path, reg = self.record_queue.get()
            img = cv2.imread(img_path)
            if img is None:
                print(img_path, os.path.isfile(img_path))
                continue
            if len(out.shape) == 3 and out.shape[2] == 3:
                self.image_queue.put((img, reg))

    def image_customer(self):
        while True:
            bboxes = []
            captions = []
            images = []
            lens = []
            for i in range(self.batch_size):
                image, reg = self.image_queue.get()
                image, bbox = self.image_process(image, reg)
                images.append(image)
                bboxes.append(bbox)
                captions.append(reg['phrase'])
                lens.append(reg['phrase_len'])
            images = np.asarray(images, dtype=np.uint8)
            bboxes = np.asarray(bboxes, dtype=np.float32)
            captions = np.asarray(captions, dtype=np.int32)
            lens = np.asarray(lens, dtype=np.int32)

            l, gt, prior, ab = preprocess(images, c313=True, prior_path=self.prior_path)
            if self.with_ab:
                self.batch_queue.put((l, gt, prior, captions, lens, ab, bboxes))
            else:    
                self.batch_queue.put((l, gt, prior, captions, lens, bboxes))

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
        """
        # print(self.record_queue.qsize(), self.image_queue.qsize(), self.batch_queue.qsize())
        return self.batch_queue.get()
