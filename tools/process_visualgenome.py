#!/srv/glusterfs/xieya/anaconda2/bin/python

# ----- Parameters passed to the cluster -------
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue

#$ -t 1:100

#$ -S /srv/glusterfs/xieya/anaconda2/bin/python

#$ -l h_rt=5:59:59

#$ -l h_vmem=8G

#$ -o /srv/glusterfs/xieya/log

#$ -e /srv/glusterfs/xieya/log

#$ -j y


import cv2
import json
import numpy as np
import os
from skimage import io
import spacy

_LOG_FREQ = 10
_RESCALE_SIZE = 224
_TASK_ID = os.environ.get('SGE_TASK_ID')
_TASK_NUM = 100
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1
else:
    _TASK_ID = 0


def _scale_to_int(x, scale):
    return int(np.round(x / scale))


def build_vocabulary():
    nlp = spacy.load('en_core_web_md')
    regions = json.load(open('/srv/glusterfs/xieya/data/visual_genome/region_descriptions.json', 'r'))
    voc = {}
    embedding = []
    for img in regions:
        for reg in img['regions']:
            print(reg['phrase'].encode('utf-8'))
            phrase = reg['phrase'].encode('utf-8')
            emb = nlp(phrase)
            for w in emb:
                print(w.text, w.vector)
                if not w.has_vector:
                    print('***********', w)
            raw_input('')


def scale_images(img_path_file, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    print('Task id: ', _TASK_ID)
    count = 0
    i = 0

    with open(img_path_file, 'r') as fin:
        for line in fin:
            if i % _TASK_NUM == _TASK_ID:
                img_path = line.strip()
                img_name = os.path.split(img_path)[1]
                img = cv2.imread(img_path)
                h = img.shape[0]
                w = img.shape[1]
                scale = 1. * _RESCALE_SIZE / min(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1. else cv2.INTER_AREA)
                cv2.imwrite(os.path.join(out_dir, img_name), img)
                count += 1

                if count % _LOG_FREQ == 0:
                    print(count)
            i +=1 


def scale_regions(region_file_name):
    regions = json.load(open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', region_file_name), 'r'))
    img_metas = json.load(open('/srv/glusterfs/xieya/data/visual_genome/image_data.json', 'r'))
    new_data = []
    for img in regions:
        img_id = img['id']
        img_meta = img_metas[img_id]
        img_w = img_meta.width
        img_h = img_meta.height
        scale = min(img_w, img_h) / 224.
        original_img = io.imread('/srv/glusterfs/xieya/data/visual_genome/VG_100K/{}.jpg'.format(img_id))
        new_regions = []
        for reg in img['regions']:
            x = reg['x']
            y = reg['y']
            w = reg['width']
            h = reg['height']

            print(reg['phrase'])
            region_img = original_img[x: x + h, y: y + w]
            io.imshow(region_img)
            io.show()

            nx = _scale_to_int(x, scale)
            ny = _scale_to_int(y, scale)
            nw = _scale_to_int(w, scale)
            nh = _scale_to_int(h, scale)
            new_reg = {'region_id': reg['region_id'], 'x': nx, 'y': ny, 'width': nw, 'height': nh, 'phrase': reg['phrase']}
            new_regions.append(new_reg)

        new_data.append({'id': img_id, 'regions': new_regions})

    json.dump(new_data, open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', 'scaled_' + region_file_name), 'w'))


if __name__ == "__main__":
    # build_vocabulary()
    scale_images('/srv/glusterfs/xieya/data/visual_genome/100k_1.txt', '/srv/glusterfs/xieya/data/visual_genome/VG_100K_224')
    # scale_regions('region_descriptions.json')
