import os
from utils import PriorFactor

enc_dir = './resources'
gamma = 0.5
alpha = 1.0

pc = PriorFactor(alpha, gamma, priorFile=os.path.join(enc_dir, 'prior_probs.npy'), verbose=True)