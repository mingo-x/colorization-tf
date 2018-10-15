import cv2
from skimage import color
from skimage.transform import downscale_local_mean, resize
from skimage.io import imread, imsave
import numpy as np
import os
import sklearn.neighbors as nn
import warnings
import configparser
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
    def __init__(self,NN,sigma,km_filepath='',cc=-1):
        if(check_value(cc,-1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='ball_tree').fit(self.cc)

        self.alreadyUsed = False

    def encode_points_mtx_nd(self,pts_nd,axis=1, sameBlock=True, flatten=False):
        if not flatten:
            pts_flt = flatten_nd_array(pts_nd,axis=axis)
        else:
            pts_flt = pts_nd

        P = pts_flt.shape[0]
        if(sameBlock and self.alreadyUsed):
            self.pts_enc_flt[...] = 0 # already pre-allocated
        else:
            self.alreadyUsed = True
            self.pts_enc_flt = np.zeros((P,self.K))
            self.p_inds = np.arange(0,P,dtype='int')[:,na()]

        (dists, inds) = self.nbrs.kneighbors(pts_flt)

        wts = np.exp(-dists**2/(2*self.sigma**2))
        wts = wts/np.sum(wts,axis=1)[:,na()]

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
    def __init__(self, prior_path=''):
        self.cc = np.load(prior_path)
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
    def __init__(self,alpha=1.0,gamma=0,verbose=False,priorFile=''):
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
        self.uni_probs[self.prior_probs!=0] = 1.
        self.uni_probs = self.uni_probs/np.sum(self.uni_probs)

        # convex combination of empirical prior and uniform distribution       
        self.prior_mix = (1-self.gamma)*self.prior_probs + self.gamma*self.uni_probs

        # set prior factor
        self.prior_factor = self.prior_mix**-self.alpha
        self.prior_factor = self.prior_factor/np.sum(self.prior_probs*self.prior_factor) # re-normalize

        # implied empirical prior
        # self.implied_prior = self.prior_probs*self.prior_factor
        # self.implied_prior = self.implied_prior/np.sum(self.implied_prior) # re-normalize

        if(self.verbose):
            self.print_correction_stats()

    def print_correction_stats(self):
        print 'Prior factor correction:'
        print '  (alpha,gamma) = (%.2f, %.2f)'%(self.alpha,self.gamma)
        print '  (min,max,mean,med,exp,std) = (%.3f, %.3f, %.3f, %.3f, %.3f, %.3f)'%(
          np.min(self.prior_factor),np.max(self.prior_factor),np.mean(self.prior_factor),np.median(self.prior_factor),np.sum(self.prior_factor*self.prior_probs), np.std(self.prior_factor))

    def forward(self, data_ab_quant,axis=1):
        data_ab_maxind = np.argmax(data_ab_quant,axis=axis)
        corr_factor = self.prior_factor[data_ab_maxind]
        if(axis==0):
            return corr_factor[na(),:]
        elif(axis==1):
            return corr_factor[:,na(),:]
        elif(axis==2):
            return corr_factor[:,:,na(),:]
        elif(axis==3):
            return corr_factor[:,:,:,na()]

    def get_weights(self, ab_idx):
        return self.prior_factor[ab_idx]

def _prior_boost(gt_ab_313, gamma=0.5, alpha=1.0, prior_path='./resources/prior_probs_smoothed.npy'):
  '''
  Args:
    gt_ab_313: (N, H, W, 313)
  Returns:
    prior_boost: (N, H, W, 1)
  '''
  gamma = gamma
  alpha = alpha

  pc = PriorFactor(alpha, gamma, priorFile=prior_path)

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


def preprocess(data, training=True, c313=False, is_gan=False, is_rgb=True, prior_path='./resources/prior_probs_smoothed.npy', mask_gray=True):
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

  #rgb2lab
  img_lab = color.rgb2lab(data)

  #slice
  #l: [0, 100]
  img_l = img_lab[:, :, :, 0:1]
  #ab: [-110, 110]
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

  #scale img_l to [-1, 1]
  data_l = (img_l - 50.) / 50.
  data_l_ss = downscale_local_mean(data_l, (1, 4, 4, 1))
  #subsample 1/4  (N * H/4 * W/4 * 2)
  # data_ab_ss = data_ab[:, ::4, ::4, :]

  data_ab_ss = downscale_local_mean(data_ab, (1, 4, 4, 1))

  #NonGrayMask {N, 1, 1, 1}
  thresh = 5
  if mask_gray:
    nongray_mask = (np.sum(np.sum(np.sum(np.abs(data_ab_ss) > thresh, axis=1), axis=1), axis=1) > 0)[:, np.newaxis, np.newaxis, np.newaxis]

  #NNEncoder
  #gt_ab_313: [N, H/4, W/4, 313]
  gt_ab_313 = _nnencode(data_ab_ss)

  if c313:
    data_313_ss = np.concatenate((data_l_ss, gt_ab_313), axis=-1)
  else:
    data_real = np.concatenate((data_l_ss, data_ab_ss / 110.), axis=-1)

  #Prior_Boost 
  #prior_boost: [N, 1, H/4, W/4]
  prior_boost = _prior_boost(gt_ab_313, prior_path=prior_path)

  #Eltwise
  #prior_boost_nongray: [N, 1, H/4, W/4]
  if mask_gray:
    prior_boost_nongray = prior_boost * nongray_mask
  else:
    prior_boost_nongray = prior_boost

  if training:
    if c313:
    # Upscale.
      return data_l, gt_ab_313, prior_boost_nongray, data_313_ss
    # return data_l, gt_ab_313, prior_boost_nongray, img_lab
    else:
      return data_l, gt_ab_313, prior_boost_nongray, data_real
  else:
    return data_l, data_ab


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference


def decode(data_l, conv8_313, rebalance=1):
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
  conv8_313_rh = conv8_313 * rebalance
  class8_313_rh = softmax(conv8_313_rh)

  cc = np.load(os.path.join(enc_dir, 'pts_in_hull.npy'))
  
  data_ab = np.dot(class8_313_rh, cc)
  # data_ab = resize(data_ab, (height, width))
  data_ab = cv2.resize(data_ab, (height, width), interpolation=cv2.INTER_CUBIC)

  img_lab = np.concatenate((data_l, data_ab), axis=-1)
  img_rgb = color.lab2rgb(img_lab)
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
    