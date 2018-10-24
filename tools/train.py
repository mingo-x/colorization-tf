import sys
sys.path.append('./')
from optparse import OptionParser
from solver import Solver
from solver_language import Solver_Language
from solver_gan import Solver_GAN
from solver_multigpu import SolverMultigpu
from solver_sampler import Solver_Sampler
from utils import process_config

parser = OptionParser()
parser.add_option("-c", "--conf", dest="configure",  
                  help="configure filename")
(options, args) = parser.parse_args() 
if options.configure:
    conf_file = str(options.configure)
else:
    print('please specify --conf configure filename')
    exit(0)

common_params, dataset_params, net_params, solver_params = process_config(conf_file)

# Log configuration.
for key in common_params:
    print(key, common_params[key])
for key in net_params:
    print(key, net_params[key])

if common_params['is_gan'] == '1':
    solver = Solver_GAN(True, common_params, solver_params, net_params, dataset_params)
    print('GAN solver.')
elif common_params['is_coco'] == '1':
    solver = Solver_Language(True, common_params, solver_params, net_params, dataset_params)
    print('Language solver.')
elif common_params['sampler'] == '1':
    solver = Solver_Sampler(True, common_params, solver_params, net_params, dataset_params)
    print('Sampler solver.')
elif len(str(common_params['gpus']).split(',')) == 1:
    solver = Solver(True, common_params, solver_params, net_params, dataset_params)
    print('Normal solver.')
else:
    solver = SolverMultigpu(True, common_params, solver_params, net_params, dataset_params)
    print('Multigpu solver.')
solver.train_model()
