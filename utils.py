import cv2
from skimage import color
from skimage.transform import downscale_local_mean, resize
from skimage.io import imread, imsave
import math
import numpy as np
import os
import sklearn.neighbors as nn
import warnings
import configparser

import pickle
import random

# *****************************
# ***** Utility functions *****
# *****************************

def check_value(inds, val):
    ''' Check to see if an array is a single element equaling a particular value
    for pre-processing inputs in a function '''
    if(np.array(inds).size==1):
        if(inds==val):
            return True
    return False

def na(): # shorthand for new axis
    return np.newaxis

def flatten_nd_array(pts_nd,axis=1):
    ''' Flatten an nd array into a 2d array with a certain axis
    INPUTS
        pts_nd       N0xN1x...xNd array
        axis         integer
    OUTPUTS
        pts_flt     prod(N \ N_axis) x N_axis array     '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
    pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS,SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt,pts_nd,axis=1,squeeze=False):
    ''' Unflatten a 2d array with a certain axis
    INPUTS
        pts_flt     prod(N \ N_axis) x M array
        pts_nd      N0xN1x...xNd array
        axis        integer
        squeeze     bool     if true, M=1, squeeze it out
    OUTPUTS
        pts_out     N0xN1x...xNd array        '''
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0,NDIM),np.array((axis))) # non axis indices
    NPTS = np.prod(SHP[nax])

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        # print NEW_SHP
        # print pts_flt.shape
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax,np.array(axis).flatten()),axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out


class NNEncode():
    ''' Encode points using NN search and Gaussian kernel '''
    def __init__(self, NN, sigma, km_filepath='', cc=-1):
        if(check_value(cc, -1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self, pts_nd, axis=1, sameBlock=True, flatten=False):
        if not flatten:
            pts_flt = flatten_nd_array(pts_nd, axis=axis)
        else:
            pts_flt = pts_nd

        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0  # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P, self.K))
            self.p_inds = np.arange(0, P, dtype='int')[:, na()]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2 / (2 * self.sigma**2))
        wts = wts / np.sum(wts, axis=1)[:, na()]

        self.pts_enc_flt[self.p_inds, inds] = wts
        if not flatten:
            pts_enc_nd = unflatten_2d_array(self.pts_enc_flt, pts_nd, axis=axis)
        else:
            pts_enc_nd = self.pts_enc_flt

        return pts_enc_nd

    def decode_points_mtx_nd(self,pts_enc_nd,axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd,axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt,self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt,pts_enc_nd,axis=axis)
        return pts_dec_nd

    def decode_1hot_mtx_nd(self,pts_enc_nd,axis=1,returnEncode=False):
        pts_1hot_nd = nd_argmax_1hot(pts_enc_nd,axis=axis)
        pts_dec_nd = self.decode_points_mtx_nd(pts_1hot_nd,axis=axis)
        if(returnEncode):
            return (pts_dec_nd,pts_1hot_nd)
        else:
            return pts_dec_nd

def _nnencode(data_ab_ss, n=10):
  '''Encode to 313bin
  Args:
    data_ab_ss: [N, H, W, 2]
  Returns:
    gt_ab_313 : [N, H, W, 313]
  '''
  NN = n
  sigma = 5.0
  enc_dir = './resources/'

  data_ab_ss = np.transpose(data_ab_ss, (0, 3, 1, 2))
  nnenc = NNEncode(NN, sigma, km_filepath=os.path.join(enc_dir, 'pts_in_hull.npy'))
  gt_ab_313 = nnenc.encode_points_mtx_nd(data_ab_ss, axis=1)

  gt_ab_313 = np.transpose(gt_ab_313, (0, 2, 3, 1))
  return gt_ab_313


class LookupEncode():
    '''Encode points using lookups'''
    def __init__(self, grid_path=''):
        self.cc = np.load(grid_path)
        self.grid_width = 10
        self.offset = np.abs(np.amin(self.cc)) + 17 # add to get rid of negative numbers
        self.x_mult = 300 # differentiate x from y
        self.labels = {}
        for idx, (x,y) in enumerate(self.cc):
            x += self.offset
            x *= self.x_mult
            y += self.offset
            if x+y in self.labels:
                print('Id collision!!!')
            self.labels[x+y] = idx

    def encode_points(self, pts_nd):
        '''Return 3d prior.'''
        pts_flt = pts_nd.reshape((-1, 2))

        # round AB coordinates to nearest grid tick
        pgrid = np.round(pts_flt / self.grid_width) * self.grid_width

        # get single number by applying offsets
        pvals = pgrid + self.offset
        pvals = pvals[:, 0] * self.x_mult + pvals[:, 1]

        labels = np.zeros(pvals.shape,dtype='int32')
        labels.fill(-1)

        # lookup in label index and assign values
        for k in self.labels:
            labels[pvals == k] = self.labels[k]

        if len(labels[labels == -1]) > 0:
            print("Point outside of grid!!!")
            labels[labels == -1] = 0

        return labels.reshape(pts_nd.shape[:-1])


# ***************************
# ***** SUPPORT CLASSES *****
# ***************************
class PriorFactor():
    ''' Class handles prior factor '''
    def __init__(self, alpha=1.0, gamma=0, verbose=False, priorFile=''):
        # INPUTS
        #   alpha           integer     prior correction factor, 0 to ignore prior, 1 to divide by prior, alpha to divide by prior**alpha
        #   gamma           integer     percentage to mix in uniform prior with empirical prior
        #   priorFile       file        file which contains prior probabilities across classes

        # settings
        self.alpha = alpha
        self.gamma = gamma
        self.verbose = verbose

        # empirical prior probability
        self.prior_probs = np.load(priorFile)

        # define uniform probability
        self.uni_probs = np.zeros_like(self.prior_probs)
        self.uni_probs[self.prior_probs != 0] = 1.
        self.uni_probs = self.uni_probs / np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1 - self.gamma) * self.prior_probs + self.gamma * self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor[self.prior_factor == np.inf] = 0.  # mask out unused classes
        self.prior_factor = self.prior_factor / np.sum(self.prior_probs * self.prior_factor)  # re-normalize

        # implied empirical prior
        # self.implied_prior = self.prior_probs*self.prior_factor
        # self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

        if(self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print 'Prior factor correction:'
        print '  (alpha,gamma) = (%.2f, %.2f)' % (self.alpha, self.gamma)
        non_zero = self.prior_factor[self.prior_factor > 0]
        print '(min,max,mean,med,exp,std) = (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)' % (
            np.min(non_zero),
            np.max(non_zero),
            np.mean(non_zero),
            np.median(non_zero),
            np.sum(self.prior_factor * self.prior_probs),
            np.std(non_zero))    

    def forward_condl(self, gt_313, l):
        shape = l.shape
        ab_idx = np.argmax(gt_313, axis=3).flatten()
        l_idx = np.round(l).flatten().astype(np.int32)
        corr_factor = self.prior_factor[l_idx, ab_idx]
        corr_factor = corr_factor.reshape(shape)
        return corr_factor      

    def forward(self, data_ab_quant, axis=1):
        data_ab_maxind = np.argmax(data_ab_quant, axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis == 0):
            return corr_factor[na(), :]
        elif(axis == 1):
            return corr_factor[:, na(), :]
        elif(axis == 2):
            return corr_factor[:, :, na(), :]
        elif(axis == 3):
            return corr_factor[:, :, :, na()]

    def get_weights(self, ab_idx):
        return self.prior_factor[ab_idx]


def _prior_boost(gt_ab_313, gamma=0.5, alpha=1.0, prior_path='./resources/prior_probs_smoothed.npy', cond_l=False, luma=None):
    '''
    Args:
      gt_ab_313: (N, H, W, 313)
    Returns:
      prior_boost: (N, H, W, 1)
    '''
    gamma = gamma
    alpha = alpha

    pc = PriorFactor(alpha, gamma, priorFile=prior_path)

    if cond_l:
        prior_boost = pc.forward_condl(gt_ab_313, luma)
    else:
        gt_ab_313 = np.transpose(gt_ab_313, (0, 3, 1, 2))
        prior_boost = pc.forward(gt_ab_313, axis=1)

        prior_boost = np.transpose(prior_boost, (0, 2, 3, 1))

    return prior_boost


def get_prior(data_ab):
    gt_ab_313 = _nnencode(data_ab)
    prior = _prior_boost(gt_ab_313, gamma=0.)
    # Non-gray mask?
    # thresh = 5
    # nongray_mask = (np.sum(np.sum(np.sum(np.abs(data_ab) > thresh, axis=1), axis=1), axis=1) > 0)[:, np.newaxis, np.newaxis, np.newaxis]
    # Subsampling?
    return prior


def preprocess(data, training=True, c313=False, is_gan=False, is_rgb=True, prior_path='./resources/prior_probs_smoothed.npy', mask_gray=True, cond_l=False, gamma=0.5, sampler=False, augment=False):
    '''Preprocess
    Args: 
      data: RGB batch (N * H * W * 3)
    Return:
      data_l: L channel batch (N * H * W * 1)
      gt_ab_313: ab discrete channel batch (N * H/4 * W/4 * 313)
      prior_boost_nongray: (N * H/4 * W/4 * 1) 
    '''
    warnings.filterwarnings("ignore")
    N = data.shape[0]
    H = data.shape[1]
    W = data.shape[2]

    # rgb2lab
    img_lab = color.rgb2lab(data)

    # slice
    # l: [0, 100]
    img_l = img_lab[:, :, :, 0:1]
    # ab: [-110, 110]
    data_ab = img_lab[:, :, :, 1:]

    if is_gan:
        if is_rgb:
            data = data.astype(np.float32)
            data /= 255.
            data -= 0.5
            data *= 2.
            # print(np.min(data), np.max(data))
            return data
        else:
            data_ab /= 110.
            # print(np.min(data), np.max(data))
            return data_ab

    data_l_ss = downscale_local_mean(img_l, (1, 4, 4, 1))
    # scale img_l to [-1, 1]
    data_l = (img_l - 50.) / 50.
    if augment:
        data_l = random_lighting(data_l)
    
    # subsample 1/4  (N * H/4 * W/4 * 2)
    # data_ab_ss = data_ab[:, ::4, ::4, :]

    data_ab_ss = downscale_local_mean(data_ab, (1, 4, 4, 1))

    # NonGrayMask {N, 1, 1, 1}
    thresh = 5
    if mask_gray:
        nongray_mask = (np.sum(np.sum(np.sum(np.abs(data_ab_ss) > thresh, axis=1), axis=1), axis=1) > 0)[:, np.newaxis, np.newaxis, np.newaxis]

    # NNEncoder
    # gt_ab_313: [N, H/4, W/4, 313]
    gt_ab_313 = _nnencode(data_ab_ss)

    # Prior_Boost 
    # prior_boost: [N, 1, H/4, W/4]
    prior_boost = _prior_boost(gt_ab_313, gamma=gamma, prior_path=prior_path, cond_l=cond_l, luma=data_l_ss)

    # Eltwise
    # prior_boost_nongray: [N, 1, H/4, W/4]
    if mask_gray:
        prior_boost_nongray = prior_boost * nongray_mask
    else:
        prior_boost_nongray = prior_boost

    if training:
        if sampler:
            data_l_ss = (data_l_ss - 50.) / 50.
            return data_l, data_l_ss, prior_boost_nongray, data_ab_ss
        else:
            return data_l, gt_ab_313, prior_boost_nongray, data_ab_ss
    else:
        return data_l, data_ab


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1)  # only difference


def JBU(ab_ss, l, k=3, scale=4):
    h, w, _ = l.shape
    ab_ic = cv2.resize(ab_ss, (w, h), interpolation=cv2.INTER_NEAREST)
    a = cv2.ximgproc.jointBilateralFilter(l, ab_ic[:, :, 0: 1], d=-1, sigmaColor=10., sigmaSpace=8.)
    b = cv2.ximgproc.jointBilateralFilter(l, ab_ic[:, :, 1:], d=-1, sigmaColor=10., sigmaSpace=8.)
    ab = np.dstack((a, b))
    # h, w, c = ab_ss.shape
    # h *= scale
    # w *= scale
    # ab = np.zeros((h, w, c))
    # for i in xrange(h):
    #     for j in xrange(w):
    #         ab[i, j] = _JBU_pix(ab_ss, l, i, j, k=k, scale=scale)
            
    return ab


def _JBU_pix(ab_ss, l, pi, pj, k=3, scale=4):
    h, w, _ = l.shape
    r = k / 2
    a = 0.
    b = 0.
    f = 0.
    for i in xrange(-r, r):
        for j in xrange(-r, r):
            qi = pi + i
            qj = pj + j
            if qi < 0 or qi >= h or qj < 0 or qj >= w:
                continue
            w = _bilateral_weight(pi, pj, qi, qj, l[pi, pj, 0], l[qi, qj, 0], scale=scale)
            a_s, b_s = ab_ss[qi / 4, qj / 4]
            a += a_s * w
            b += b_s * w
            f += w

    a /= f
    b /= f

    return (a, b)


def _bilateral_weight(pi, pj, qi, qj, lp, lq, sig_d=3., sig_r=15., scale=4.0):
    domain_term = ((pi / scale - qi / scale) ** 2 + (pj / scale - qj / scale) ** 2) / (2 * sig_d ** 2)
    range_term = (lp - lq) ** 2 / (2 * sig_r ** 2)
    return math.exp(-(domain_term + range_term))


def decode(data_l, conv8_313, rebalance=1, return_313=False, sfm=True, jbu=False, jbu_k=3):
    """
    Args:
      data_l   : [1, height, width, 1]
      conv8_313: [1, height/4, width/4, 313]
    Returns:
      img_rgb  : [height, width, 3]
    """
    # [-1, 1] to [0, 100]
    data_l = (data_l + 1) * 50
    _, height, width, _ = data_l.shape
    data_l = data_l[0, :, :, :]
    conv8_313 = conv8_313[0, :, :, :]
    enc_dir = './resources'
    if sfm:
        conv8_313_rh = conv8_313 * rebalance
        class8_313_rh = softmax(conv8_313_rh)
    else:
        class8_313_rh = conv8_313

    cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
    
    data_ab = np.dot(class8_313_rh, cc)
    # data_ab = resize(data_ab, (height, width))
    if jbu:
        data_ab_us = JBU(data_ab.astype('float32'), data_l.astype('float32'), k=jbu_k)
    else:
        data_ab_us = cv2.resize(data_ab, (width, height), interpolation=cv2.INTER_CUBIC)

    img_lab = np.concatenate((data_l, data_ab_us), axis=-1).astype('float64')
    img_rgb = color.lab2rgb(img_lab)
    if return_313:
        return img_rgb, data_ab, class8_313_rh, softmax(conv8_313)
    else:
        return img_rgb, data_ab


def get_data_l(image_path):
  """
  Args:
    image_path  
  Returns:
    data_l 
  """
  data = imread(image_path)
  data = data[None, :, :, :]
  img_lab = color.rgb2lab(data)
  img_l = img_lab[:, :, :, 0:1]
  data_l = img_l - 50
  data_l = data_l.astype(dtype=np.float32)
  return data, data_l

def process_config(conf_file):
  """process configure file to generate CommonParams, DataSetParams, NetParams 
  Args:
    conf_file: configure file path 
  Returns:
    CommonParams, DataSetParams, NetParams, SolverParams
  """
  common_params = {}
  dataset_params = {}
  net_params = {}
  solver_params = {}

  #configure_parser
  config = configparser.ConfigParser()
  config.read(conf_file)

  #sections and options
  for section in config.sections():
    #construct common_params
    if section == 'Common':
      for option in config.options(section):
        common_params[option] = config.get(section, option)
    #construct dataset_params
    if section == 'DataSet':
      for option in config.options(section):
        dataset_params[option] = config.get(section, option)
    #construct net_params
    if section == 'Net':
      for option in config.options(section):
        net_params[option] = config.get(section, option)
    #construct solver_params
    if section == 'Solver':
      for option in config.options(section):
        solver_params[option] = config.get(section, option)

  return common_params, dataset_params, net_params, solver_params


def save_images(X, save_path):
    # [0, 1] -> [0,255]
    if isinstance(X.flatten()[0], np.floating):
        X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        # X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3), dtype=X.dtype)
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw))

    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x

    imsave(save_path, img)
  

def is_grayscale(gt_ab):
    thresh = 5
    if len(gt_ab.shape) == 4:
        is_gray = np.sum(np.sum(np.sum(np.abs(gt_ab) > thresh, axis=1), axis=1), axis=1) == 0
    elif len(gt_ab.shape) == 3:
        is_gray = np.sum(np.abs(gt_ab) > thresh) == 0
    return is_gray


class CaptionPrior():
    def __init__(self, gamma=0.5):
        color_probs = pickle.load(open('/home/xieya/colorfromlanguage/priors/color_probs.p', 'r'))
        color_num = len(color_probs)
        smoothed_color_probs = {}
        uniform = 1. / color_num
        for c in color_probs:
            smoothed_color_probs[c] = gamma * uniform + (1 - gamma) * color_probs[c]
        self.color_weights = {}
        for c in color_probs:
            self.color_weights[c] = 1. / smoothed_color_probs[c] / color_num
            print(c, self.color_weights[c])

    def _get_weight(self, caption):
        weight = 0.
        for w in caption:
            if w in self.color_weights:
                weight += self.color_weights[w]
        if weight == 0.:
            weight = 1
        return weight

    def get_weight(self, captions):
        n_dims = len(captions.shape)
        if n_dims == 2:
            weights = []
            for caption in captions:
                weights.append(self._get_weight(caption))
            weights = np.asarray(weights)
            return weights
        elif n_dims == 1:
            return self._get_weight(captions)


def rand0(x):
    # [-x, x)
    return (random.random() * 2 - 1) * x


def random_lighting(img_l):
    # input: [-1, 1]
    # Random lighting by FastAI
    b = rand0(0.2)
    c = rand0(0.1)
    c = -1 / (c - 1) if c < 0 else c + 1
    mu = np.mean(img_l)

    return np.clip((img_l - mu) * c + mu + b, -1, 1).astype('float32')
