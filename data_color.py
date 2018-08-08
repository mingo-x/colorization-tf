from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np
from Queue import Queue
from threading import Thread as Process
#from multiprocessing import Process,Queue

from utils import *

class DataSet(object):
  """TextDataSet
  process text input file dataset 
  text file format:
    image_path
  """

  def __init__(self):
    """
    Args:
      common_params: A dict
      dataset_params: A dict
    """
    self.input_size = 224
    self.batch_size = 32
    
    self.data_path = 'data/train.txt'
    self.gray_dir = '/srv/glusterfs/xieya/data/imagenet_gray/train'
    self.thread_num = 8
    self.thread_num2 = 8
    #record and image_label queue
    self.record_queue = Queue(maxsize=16000)
    self.image_queue = Queue(maxsize=8000)

    self.batch_queue = Queue(maxsize=200)

    self.record_list = []  
    input_file = open(self.data_path, 'r')

    for line in input_file:
      line = line.strip()
      name = os.path.split(line)[1]
      self.record_list.append(name)
    # self.record_list.sort()

    self.record_point = 0
    self.record_number = len(self.record_list)
    print('Total: {}'.format(self.record_number))

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
        self.record_point = 0
      self.record_queue.put(self.record_list[self.record_point])
      self.record_point += 1

  def image_process(self, img_batch):
    img_l_batch = []
    img_l_rs_batch = []
    img_name_batch = []
    for img, name in img_batch:
        img_rs = cv2.resize(img, (self.input_size, self.input_size))

        img_l = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_l = img_l[None, :, :, None]
        img_l_rs = cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY)
        img_l_rs = img_l_rs[:, :, None]

        img_l = (img_l.astype(dtype=np.float32)) / 255. * 100 - 50
        img_l_rs = (img_l_rs.astype(dtype=np.float32)) / 255.0 * 100 - 50

        img_l_batch.append(img_l)
        img_l_rs_batch.append(img_l_rs)
        img_name_batch.append(name)

    img_l_rs_batch = np.asarray(img_l_rs_batch)

    return img_l_batch, img_l_rs_batch, img_name_batch

  def record_customer(self):
    """record queue's customer 
    """
    while True:
      item = self.record_queue.get()
      out = cv2.imread(os.path.join(self.gray_dir, item))
      if len(out.shape)==3 and out.shape[2]==3:
        self.image_queue.put((out, item))


  def image_customer(self):
    while True:
      images = []
      for i in range(self.batch_size):
        image = self.image_queue.get()
        # image = self.image_process(image)
        images.append(image)
      # images = np.asarray(images, dtype=np.uint8)

      self.batch_queue.put(image_process(images))


  def batch(self):
    """get batch
    Returns:
      images: 4-D ndarray [batch_size, height, width, 3]
    """
    # print(self.record_queue.qsize(), self.image_queue.qsize(), self.batch_queue.qsize())
    return self.batch_queue.get()