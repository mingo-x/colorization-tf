from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import h5py
import cv2
from glob import glob
import numpy as np
import os
import pickle
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

    def __init__(self, common_params=None, dataset_params=None, train=True, with_ab=False, shuffle=True, with_cocoseg=False):
        """
        Args:
          common_params: A dict
          dataset_params: A dict
        """
        self.cap_prior = False
        self.with_cocoseg = with_cocoseg
        if self.with_cocoseg:
            self.im2cap = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/im2cap_comb.p', 'rb'))
            self.caps = np.load('/srv/glusterfs/xieya/data/coco_seg/annotations/captions_comb.npy')
            self.lens = np.load('/srv/glusterfs/xieya/data/coco_seg/annotations/caption_lengths_comb.npy')
            file_list = sorted(glob(os.path.join("/srv/glusterfs/xieya/data/coco_seg/images_224/val2017", "*.jpg")))
            file_list = [f.split("/")[-1].replace(".jpg", "") for f in file_list]
            file_list = list(filter(lambda x: x in self.im2cap, file_list))
            self.files = file_list
            cocoseg_dict = pickle.load(open('/srv/glusterfs/xieya/data/language/vocabulary.p', 'r'))
            self.cocoseg_vrev = dict((v, k) for (k, v) in cocoseg_dict.iteritems()) 
            self.this_dict = pickle.load(open('/home/xieya/colorfromlanguage/priors/coco_colors_vocab.p', 'r'))
            self.img_id = 0
            self.gray_list = pickle.load(open('/srv/glusterfs/xieya/data/coco_seg/val_filtered_gray.p', 'rb'))

        if dataset_params:
            self.data_path = str(dataset_params['path'])
            self.thread_num = int(dataset_params['thread_num'])
            self.prior_path = str(dataset_params['prior_path']) if 'prior_path' in dataset_params else './resources/prior_probs_smoothed.npy'

        if common_params:
            self.batch_size = int(common_params['batch_size'])
            self.with_caption = True if common_params['with_caption'] == '1' else False
            self.sampler = True if common_params['sampler'] == '1' else False
            if 'with_cap_prior' in common_params:
                self.cap_prior = common_params['with_cap_prior'] == '1'
                self.cap_prior_gamma = float(common_params['cap_prior_gamma'])

        self.training = train
        self.with_ab = with_ab
        if self.cap_prior:
            self.cap_prior_encoder = CaptionPrior(self.cap_prior_gamma)

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
            # self.thread_num = 4

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
        if self.with_cocoseg:
            while True:
                for i in self.files:
                    self.record_queue.put(i)
        else:
            while True:
                if self.record_point % self.record_number == 0:
                    if self.shuffle:
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
                if self.with_cocoseg:
                    self.img_id += 1
                    if self.img_id in self.gray_list:
                        continue
                    cidx = self.im2cap[idx][0]
                    image_path = os.path.join("/srv/glusterfs/xieya/data/coco_seg/images_224/val2017", idx + ".jpg")
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    h, w, _ = image.shape
                    thr_h = h - 224
                    thr_w = w - 224
                    start_h = int((thr_h + 1) / 2)
                    start_w = int((thr_w + 1) / 2)
                    image = image[start_h: start_h + 224, start_w: start_w + 224, :]
                    caption = self.caps[cidx]
                    # Conver caption from cocoseg dict to this dict
                    caption = [self.cocoseg_vrev.get(c, 'unk') for c in caption]
                    caption = [self.this_dict.get(c, 0) for c in caption]
                    length = self.lens[cidx]
                else:
                    image = self.train_origs[idx]
                    caption = self.train_words[idx]
                    length = self.train_lengths[idx]
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[:, :, 1:]
                # if is_grayscale(ab):
                #     continue
                # Augmentation.
                mirror = np.random.randint(0, 2)
                if self.training and mirror:
                    image = np.fliplr(image)
                images.append(image)
                captions.append(caption)
                lens.append(length)
                count += 1
            images = np.asarray(images, dtype=np.uint8)
            captions = np.asarray(captions, dtype=np.int32)
            if self.cap_prior:
                cap_priors = self.cap_prior_encoder.get_weight(captions)
                # print(np.sum(cap_priors))
            lens = np.asarray(lens, dtype=np.int32)
            l, gt, prior, ab = preprocess(images, c313=True, prior_path=self.prior_path, mask_gray=(not self.with_caption), sampler=self.sampler)
            if self.with_ab:
                self.batch_queue.put((l, gt, prior, captions, lens, ab))
            elif self.cap_prior:
                self.batch_queue.put((l, gt, prior, captions, lens, cap_priors))
            else:    
                self.batch_queue.put((l, gt, prior, captions, lens))

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
        """
        # print(self.record_queue.qsize(), self.batch_queue.qsize())
        return self.batch_queue.get()
