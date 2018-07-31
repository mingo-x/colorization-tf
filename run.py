#!/srv/glusterfs/xieya/anaconda2/bin/python

# ----- Parameters passed to the cluster -------
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue

#$ -t 1:2

#$ -S /srv/glusterfs/xieya/anaconda2/bin/python

#$ -l h_rt=0:59:59

#$ -l h_vmem=4G

#$ -o /srv/glusterfs/xieya/log

#$ -e /srv/glusterfs/xieya/log

#$ -j y


import os
import sys

task_id = os.environ.get('SGE_TASK_ID')

print task_id

