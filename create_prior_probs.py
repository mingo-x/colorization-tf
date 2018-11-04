#!/srv/glusterfs/xieya/anaconda2/bin/python

# ----- Parameters passed to the cluster -------
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue

#$ -t 1:50

#$ -S /srv/glusterfs/xieya/anaconda2/bin/python

#$ -l h_rt=5:59:59

#$ -l h_vmem=8G

#$ -o /srv/glusterfs/xieya/log

#$ -e /srv/glusterfs/xieya/log

#$ -j y

#$ -cwd

import h5py
import numpy as np
from skimage.io import imread
from skimage import color
from skimage.transform import resize
import sklearn.neighbors as nn

import os
import sys

_GRID_PATH = ''
_LOG_FREQ = 100
_N_CLASSES = 313
_TASK_NUM = 50
_TASK_ID = os.environ.get('SGE_TASK_ID')
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1
else:
    _TASK_ID = 0


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        self.cc = np.load(km_filepath)
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='ball_tree').fit(self.cc)
        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1, sameBlock=True):
        pts_flt = pts_nd
        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:, np.newaxis]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:, np.newaxis]

        self.pts_enc_flt[self.p_inds, inds] = wts
        pts_enc_nd = self.pts_enc_flt

        return pts_enc_nd


class LookupEncode():
    '''Encode points using lookups'''
    def __init__(self, grid_path=''):
        self.cc = np.load(grid_path)
        self.grid_width = 10
        self.offset = np.abs(np.amin(self.cc)) + 17  # add to get rid of negative numbers
        self.x_mult = 300  # differentiate x from y
        self.labels = {}
        for idx, (x, y) in enumerate(self.cc):
            x += self.offset
            x *= self.x_mult
            y += self.offset
            if x + y in self.labels:
                print('Id collision!!!')
            self.labels[x + y] = idx

    def encode_points(self, pts_nd):
        '''Return 3d prior.'''
        pts_flt = pts_nd.reshape((-1, 2))

        # round AB coordinates to nearest grid tick
        pgrid = np.round(pts_flt / self.grid_width) * self.grid_width

        # get single number by applying offsets
        pvals = pgrid + self.offset
        pvals = pvals[:, 0] * self.x_mult + pvals[:, 1]

        labels = np.zeros(pvals.shape, dtype='int32')
        labels.fill(-1)

        # lookup in label index and assign values
        for k in self.labels:
            labels[pvals == k] = self.labels[k]

        if len(labels[labels == -1]) > 0:
            print("Point outside of grid!!!")
            labels[labels == -1] = 0

        return labels.reshape(pts_nd.shape[:-1])


def get_index(ab):
    ab = ab[:, np.newaxis, :]
    distance = np.sum(np.square(ab - points), axis=2)
    index = np.argmin(distance, axis=1)
    return index


def get_file_list():
    filename_list = []
    img_idx = 0
    for img_f in lists_f:
        if img_idx % _TASK_NUM == _TASK_ID:
            img_f = img_f.strip()
            filename_list.append(img_f)
        img_idx += 1
        # if img_idx >= 19200:
        #     break
    return filename_list


def cal_prob():
    out_path = '/srv/glusterfs/xieya/prior/{0}_onehot_{1}.npy'.format(_N_CLASSES, _TASK_ID)
    # if os.path.isfile(out_path):
        # print('Done.')
        # return

    filename_lists = get_file_list()
    counter = 0
    probs = np.zeros((_N_CLASSES), dtype=np.float64)
    # random.shuffle(filename_lists)

    # construct graph
    # in_data = tf.placeholder(tf.float64, [None, 2])
    # expand_in_data = tf.expand_dims(in_data, axis=1)

    # distance = tf.reduce_sum(tf.square(expand_in_data - points), axis=2)
    # index = tf.argmin(distance, axis=1)
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # sess = tf.Session(config=config)

    for img_f in filename_lists:
        img_f = img_f.strip()
        if not os.path.isfile(img_f):
            print(img_f)
            continue
        img = imread(img_f)
        img = resize(img, (224, 224))
        if len(img.shape) != 3 or img.shape[2] != 3:
            continue
        img_lab = color.rgb2lab(img)
        img_lab = img_lab.reshape((-1, 3))
        img_ab = img_lab[:, 1:]
        # nd_index = sess.run(index, feed_dict={in_data: img_ab})
        nd_index = get_index(img_ab)
        for i in nd_index:
            i = int(i)
            probs[i] += 1

        if counter % _LOG_FREQ == 0:
            print(counter)
            sys.stdout.flush()
        counter += 1

    # sess.close()
    # probs = probs / np.sum(probs)
    np.save(out_path, probs)


def cal_prob_soft(cond_l=False, is_vg=False):
    if cond_l:
        print('Conditioning on luma.')

    out_path = '/srv/glusterfs/xieya/prior/vg_{0}_{2}soft_{1}.npy'.format(_N_CLASSES, _TASK_ID, 'abl_' if cond_l else '')
    if os.path.isfile(out_path):
        print('Done.')
        return

    if is_vg:
        filename_lists = []
        dir_path = '/srv/glusterfs/xieya/data/visual_genome/VG_100K_224'
        fs = os.listdir(dir_path)
        fs.sort()
        print('Sorted.')
        for i in xrange(len(fs)):
            if i % _TASK_NUM == 0:
                f = fs[i]
                filename_lists.append(os.path.join(dir_path, f))
    else:
        filename_lists = get_file_list()
    counter = 0
    nnenc = NNEncode(10, 5.0, km_filepath='./resources/pts_in_hull.npy')
    if cond_l:
        probs = np.zeros((101, _N_CLASSES), dtype=np.float64)    
    else:
        probs = np.zeros((_N_CLASSES), dtype=np.float64)

    for img_f in filename_lists:
        # img_f = img_f.strip()
        if not os.path.isfile(img_f):
            print(img_f)
            continue
        img = imread(img_f)
        if len(img.shape) != 3 or img.shape[2] != 3:
            continue
        img = resize(img, (224, 224))
        img_lab = color.rgb2lab(img)
        img_lab = img_lab.reshape((-1, 3))
        img_ab = img_lab[:, 1:]
        img_313 = nnenc.encode_points_mtx_nd(img_ab, axis=1)  # [H*W, 313]
        if cond_l:
            img_l = img_lab[:, 0]
            l_idx = np.round(img_l).astype(np.int32)
            for l in xrange(101):
                probs[l] += np.sum(img_313[l_idx == l], axis=0)
        else:
            probs += np.sum(img_313, axis=0)

        if counter % _LOG_FREQ == 0:
            print(counter)
            sys.stdout.flush()
        counter += 1

    np.save(out_path, probs)


def cal_prob_coco():
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    train_origs = hf['train_ims']  # BGR format
    counter = 0
    probs = np.zeros((_N_CLASSES), dtype=np.float64)

    for i in xrange(len(train_origs)):
        if i % _TASK_NUM != _TASK_ID:
            continue
        img_bgr = train_origs[i]
        img_rgb = img_bgr[:, :, ::-1]
        img_lab = color.rgb2lab(img_rgb)
        img_lab = img_lab.reshape((-1, 3))
        img_ab = img_lab[:, 1:]
        nd_index = get_index(img_ab)
        for i in nd_index:
            i = int(i)
            probs[i] += 1

        if counter % _LOG_FREQ == 0:
            print(counter)
            sys.stdout.flush()
        counter += 1

    np.save('/srv/glusterfs/xieya/prior/coco_{0}_onehot_{1}'.format(_N_CLASSES, _TASK_ID), probs)


def cal_prob_coco_soft(cond_l=False):
    hf = h5py.File('/srv/glusterfs/xieya/data/coco_colors.h5', 'r')
    train_origs = hf['train_ims']  # BGR format
    counter = 0
    nnenc = NNEncode(10, 5.0, km_filepath='./resources/pts_in_hull.npy')
    if cond_l:
        probs = np.zeros((101, _N_CLASSES), dtype=np.float64)
    else:
        probs = np.zeros((_N_CLASSES), dtype=np.float64)

    for i in xrange(len(train_origs)):
        if i % _TASK_NUM != _TASK_ID:
            continue
        img_bgr = train_origs[i]
        img_rgb = img_bgr[:, :, ::-1]
        img_lab = color.rgb2lab(img_rgb)
        img_lab = img_lab.reshape((-1, 3))
        img_ab = img_lab[:, 1:]
        img_313 = nnenc.encode_points_mtx_nd(img_ab, axis=1)  # [H*W, 313]
        if cond_l:
            img_l = img_lab[:, 0]
            l_idx = np.round(img_l).astype(np.int32)
            for l in xrange(101):
                probs[l] += np.sum(img_313[l_idx == l], axis=0)
        else:
            probs += np.sum(img_313, axis=0)

        if counter % _LOG_FREQ == 0:
            print(counter)
            sys.stdout.flush()
        counter += 1

    np.save('/srv/glusterfs/xieya/prior/coco_{0}_{2}_soft_{1}'.format(_N_CLASSES, _TASK_ID, 'abl' if cond_l else ''), probs)


def merge():
    print("Merging...")
    probs = np.zeros((_N_CLASSES), dtype=np.float64)
    path_pattern = '/srv/glusterfs/xieya/prior/{0}_onehot_{1}_soft.npy'
    for i in xrange(_TASK_NUM):
        file_path = path_pattern.format(_N_CLASSES, i)
        p = np.load(file_path)
        probs += p
        print(i)
    probs = probs / np.sum(probs)
    np.save('/srv/glusterfs/xieya/prior/{}_soft'.format(_N_CLASSES), probs)


def cal_ab_hist_given_l():
    out_path = '/srv/glusterfs/xieya/prior/val_{0}_abl_{1}.npy'.format(_N_CLASSES, _TASK_ID)
    if os.path.isfile(out_path):
        print('Done.')
        return

    filename_lists = get_file_list()
    counter = 0
    probs = np.zeros((101, _N_CLASSES), dtype=np.float64)
    lookup = LookupEncode('resources/pts_in_hull.npy')

    for img_f in filename_lists:
        img_f = img_f.strip()
        if not os.path.isfile(img_f):
            print(img_f)
            continue
        img = imread(img_f)
        img = resize(img, (224, 224))
        if len(img.shape) != 3 or img.shape[2] != 3:
            continue
        img_lab = color.rgb2lab(img).reshape((-1, 3))
        img_l = img_lab[:, 0]
        img_ab = img_lab[:, 1:]
        ab_idx = lookup.encode_points(img_ab)
        l_idx = np.round(img_l).astype(np.int32)
        for ab in xrange(313):
            for l in xrange(101):
                probs[l, ab] += np.sum(np.logical_and(ab_idx == ab, l_idx == l))

        if counter % _LOG_FREQ == 0:
            print(counter)
            sys.stdout.flush()
        counter += 1

    np.save(out_path, probs)


def merge_abl():
    print("Merging...")
    probs = np.zeros((101, _N_CLASSES), dtype=np.float64)
    path_pattern = '/srv/glusterfs/xieya/prior/val_{0}_abl_soft_{1}.npy'
    for i in xrange(_TASK_NUM):
        file_path = path_pattern.format(_N_CLASSES, i)
        if not os.path.exists(file_path):
            print("{} missing, skipped.".format(file_path))
            continue
        p = np.load(file_path)
        probs += p
        print(i)
    probs_nonzero = probs[probs > 0]
    print(np.mean(probs_nonzero), np.min(probs_nonzero), np.max(probs_nonzero), np.median(probs_nonzero), np.std(probs_nonzero))
    probs = probs / np.sum(probs)
    np.save('/srv/glusterfs/xieya/prior/val_{}_abl_soft_bin1'.format(_N_CLASSES), probs)


if __name__ == "__main__":
    lists_f = open('/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val.txt')
    if _N_CLASSES == 313:
        _GRID_PATH = '/home/xieya/colorization-tf/resources/pts_in_hull.npy'
    else:
        _GRID_PATH = '/home/xieya/colorfromlanguage/priors/full_lab_grid_10.npy'
    points = np.load(_GRID_PATH)
    points = points.astype(np.float64)
    points = points[None, :, :]
    print("Number of classes: {}.".format(_N_CLASSES))
    # print("Imagenet.")
    # cal_prob()
    cal_prob_soft(False, is_vg=True)
    # cal_ab_hist_given_l()
    # print("Coco.")
    # cal_prob_coco()
    # cal_prob_coco_soft(True)
    # merge()
    # merge_abl()
