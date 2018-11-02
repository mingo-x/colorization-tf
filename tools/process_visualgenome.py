#!/srv/glusterfs/xieya/anaconda2/bin/python

# ----- Parameters passed to the cluster -------
## <= 1h is short queue, <= 6h is middle queue, <= 48 h is long queue

#$ -t 1:1

#$ -S /srv/glusterfs/xieya/anaconda2/bin/python

#$ -l h_rt=5:59:59

#$ -l h_vmem=40G

#$ -o /srv/glusterfs/xieya/log

#$ -e /srv/glusterfs/xieya/log

#$ -j y


import cv2
import json
import numpy as np
import os
import pickle
from skimage import io

_LOG_FREQ = 10
_RESCALE_SIZE = 224
_TASK_ID = os.environ.get('SGE_TASK_ID')
_TASK_NUM = 50
if _TASK_ID is not None:
    print("Task id: {}".format(_TASK_ID))
    _TASK_ID = int(_TASK_ID) - 1
else:
    _TASK_ID = 0


def _scale_to_int(x, scale):
    return int(np.round(x / scale))


def build_vocabulary_by_spacy():
    import spacy
    nlp = spacy.load('en_core_web_md')
    print('Word embedding loaded.')
    regions = json.load(open('/srv/glusterfs/xieya/data/visual_genome/region_descriptions.json', 'r'))
    print('Region json loaded.')
    voc = {'unk': 0}
    unk_phrase = unicode('bckground')
    unk_token = nlp(unk_phrase)[0]
    embedding = [unk_token.vector]
    img_count = 0
    for img in regions:
        for reg in img['regions']:
            phrase = reg['phrase']
            emb = nlp(phrase)
            for w in emb:
                word = w.text.encode('utf-8').lower()
                if word not in voc:
                    if w.has_vector:
                        idx = len(voc)
                        embedding.append(w.vector)
                        voc[word] = idx
                        print(word, len(voc))
        img_count += 1
        if img_count % 100 == 0:
            print("Image count: {}".format(img_count))

    print("Vocabulary size: {}".format(len(voc)))
    pickle.dump(voc, open('/srv/glusterfs/xieya/data/visual_genome/spacy_voc.p', 'wb'))
    pickle.dump(embedding, open('/srv/glusterfs/xieya/data/visual_genome/spacy_emb.p', 'wb'))


def load_glove(filename):
    emb_dict = {}
    dir_path = '/srv/glusterfs/xieya/data/language'
    emb_name = os.path.splitext(filename)[0]
    with open(os.path.join(dir_path, filename)) as fin:
        for i, line in enumerate(fin):
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:]
            emb = [float(x) for x in entries]
            emb_dict[word] = emb
            if i % 100 == 0:
                print(i, word, emb[0: 5])
    pickle.dump(emb_dict, open(os.path.join(dir_path, emb_name + '.p'), 'wb'))


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
    print('Region json loaded.')
    # img_metas = json.load(open('/srv/glusterfs/xieya/data/visual_genome/image_data.json', 'r'))
    # img_map = {}
    # for i in xrange(len(img_metas)):
    #     img_map[img_metas[i]['image_id']] = i
    # print('Image json loaded.')
    new_data = []
    for img in regions:
        img_id = img['id']
        scale = min(img_w, img_h) / 224.
        new_regions = []
        for reg in img['regions']:
            reg_id = reg['region_id']
            x = reg['x']
            y = reg['y']
            w = reg['width']
            h = reg['height']
            phrase = reg['phrase']

            nx = _scale_to_int(x, scale)
            ny = _scale_to_int(y, scale)
            nw = _scale_to_int(w, scale)
            nh = _scale_to_int(h, scale)
            new_reg = {'region_id': reg_id, 'x': nx, 'y': ny, 'width': nw, 'height': nh, 'phrase': phrase}
            new_regions.append(new_reg)

        new_data.append({'id': img_id, 'regions': new_regions})

    json.dump(new_data, open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', '224_' + region_file_name), 'w'))


if __name__ == "__main__":
    # build_vocabulary_by_spacy()
    load_glove('glove.6B.300d.txt')
    # scale_images('/srv/glusterfs/xieya/data/visual_genome/100k_2.txt', '/srv/glusterfs/xieya/data/visual_genome/VG_100K_224_2')
    # scale_regions('region_descriptions.json')
