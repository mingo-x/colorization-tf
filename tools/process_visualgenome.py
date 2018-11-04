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


import cv2
import json
import numpy as np
import os
import pickle
from skimage import io
import string

_LANGUAGE_DIR = '/srv/glusterfs/xieya/data/language'
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


def build_vocabulary_by_glove(emb_file):
    emb_name = os.path.splitext(emb_file)[0]
    emb_dict = pickle.load(open(os.path.join(_LANGUAGE_DIR, emb_file), 'rb'))
    regions = json.load(open('/srv/glusterfs/xieya/data/visual_genome/region_descriptions.json', 'r'))
    print('Region json loaded.')
    voc_dict = {'unk': 0}
    embeddings = [emb_dict['unk']]
    unknowns = set()
    for img in regions:
        for reg in img['regions']:
            phrase = reg['phrase'].encode('utf-8').lower()
            words = phrase.strip().split()
            for w in words:
                if w not in voc_dict:
                    if w in emb_dict:
                        idx = len(voc_dict)
                        embeddings.append(emb_dict[w])
                        voc_dict[w] = idx
                        print(w, idx)
                    else:
                        nw = w.translate(None, string.punctuation)
                        if nw in emb_dict:
                            idx = len(voc_dict)
                            embeddings.append(emb_dict[nw])
                            voc_dict[w] = idx
                            print(nw, idx)
                        else:
                            unknowns.add(w)
    # Manual amendament.
    if 'tshirt' not in voc_dict:
        idx = len(voc_dict)
        voc_dict['tshirt'] = idx
        embeddings.append(emb_dict['t-shirt'])

    print("Vocabulary size: {}".format(len(voc_dict)))
    pickle.dump(voc_dict, open('/srv/glusterfs/xieya/data/visual_genome/{}_voc.p'.format(emb_name), 'wb'))
    pickle.dump(embeddings, open('/srv/glusterfs/xieya/data/visual_genome/{}_emb.p'.format(emb_name), 'wb'))
    print("Unknown size: {}".format(len(unknowns)))
    with open('/srv/glusterfs/xieya/data/visual_genome/{}_unknown.txt'.format(emb_name), 'w') as fout:
        for w in unknowns:
            fout.write(w + '\n')


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


def filter_regions(emb_name, region_json_file):
    voc_dict = pickle.load(open('/srv/glusterfs/xieya/data/visual_genome/{}_voc.p'.format(emb_name), 'rb'))
    regions = json.load(open(os.path.join('/srv/glusterfs/xieya/data/visual_genome/', region_json_file), 'r'))
    print('Region json loaded.')
    color_voc = pickle.load(open('/srv/glusterfs/xieya/data/color/vocabulary.p', 'rb'))
    color_set = set()
    for c in color_voc:
        if c in voc_dict:
            color_set.add(voc_dict[c])
        else:
            print('{} not in vocabulary'.format(c))
    
    new_data = []
    region_count = 0
    has_color_count = 0
    less_20_count = 0

    for img in regions:
        img_id = img['id']
        new_regs = []
        for reg in img['regions']:
            region_count += 1
            phrase = reg['phrase'].encode('utf-8').lower()
            words = phrase.strip().split()

            # Turn text into vector.
            phrase_len = len(words)
            if phrase_len > 20:
                continue
            else:
                less_20_count += 1
            vector = [0] * 20
            for i in xrange(phrase_len):
                vector[i] = voc_dict.get(words[i], 0)

            # Filter out phrase without color.
            has_color = False
            for i in xrange(phrase_len):
                w_id = vector[i]
                if w_id in color_set:
                    has_color = True
                    break
            if has_color:
                has_color_count += 1
            else:
                continue

            new_reg = {'region_id': reg['region_id'], 'x': reg['x'], 'y': reg['y'], 'width': reg['width'], 'height': reg['height'], 'phrase': vector, 'phrase_len': phrase_len}
            new_regs.append(new_reg)

        new_data.append({'id': img_id, 'regions': new_regs})
        print(has_color_count)

    print('Total regions: {}'.format(region_count))
    print('Total images: {}'.format(len(new_data)))
    print('Regions with color: {}'.format(has_color_count))
    print('Regions with valid length: {}'.format(less_20_count))
    json.dump(new_data, open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', 'filtered_' + region_json_file), 'w'))


def load_glove(filename):
    emb_dict = {}
    emb_name = os.path.splitext(filename)[0]
    with open(os.path.join(_LANGUAGE_DIR, filename)) as fin:
        for i, line in enumerate(fin):
            tokens = line.split(' ')
            word = tokens[0]
            entries = tokens[1:]
            emb = [float(x) for x in entries]
            emb_dict[word] = emb
            if i % 100 == 0:
                print(i, word, emb[0: 5])
    pickle.dump(emb_dict, open(os.path.join(_LANGUAGE_DIR, emb_name + '.p'), 'wb'))


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
                out_path = os.path.join(out_dir, img_name)
                if os.path.exists(out_path):
                    continue
                img = cv2.imread(img_path)
                h = img.shape[0]
                w = img.shape[1]
                scale = 1. * _RESCALE_SIZE / min(h, w)
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC if scale > 1. else cv2.INTER_AREA)
                cv2.imwrite(out_path, img)
                count += 1

                if count % _LOG_FREQ == 0:
                    print(count)
            i +=1 


def scale_regions(region_file_name):
    regions = json.load(open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', region_file_name), 'r'))
    print('Region json loaded.')
    img_metas = json.load(open('/srv/glusterfs/xieya/data/visual_genome/image_data.json', 'r'))
    img_map = {}
    for i in xrange(len(img_metas)):
        img_map[img_metas[i]['image_id']] = i
    print('Image json loaded.')
    reg_count = 0
    for img in regions:
        img_id = img['id']
        img_meta = img_metas[img_map[img_id]]
        img_w = img_meta['width']
        img_h = img_meta['height']
        scale = min(img_w, img_h) / 224.
        for reg in img['regions']:
            x = reg['x']
            y = reg['y']
            w = reg['width']
            h = reg['height']

            nx = _scale_to_int(x, scale)
            ny = _scale_to_int(y, scale)
            nw = _scale_to_int(w, scale)
            nh = _scale_to_int(h, scale)
            reg['x'] = nx
            reg['y'] = ny
            reg['width'] = nw
            reg['height'] = nh
        reg_count += len(img['regions'])

    json.dump(regions, open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', '224_' + region_file_name), 'w'))
    print('Total images: {}'.format(len(regions)))
    print('Total regions: {}'.format(reg_count))


def split_train_val_test(region_file_name):
    regions = json.load(open(os.path.join('/srv/glusterfs/xieya/data/visual_genome', region_file_name), 'r'))
    print('Region json loaded.')
    train_img_count = 0
    train_reg_count = 0
    val_img_count = 0
    val_reg_count = 0
    test_img_count = 0
    test_reg_count = 0
    with open('/srv/glusterfs/xieya/data/visual_genome/train.txt', 'w') as f_train, open('/srv/glusterfs/xieya/data/visual_genome/val.txt', 'w') as f_val, open('/srv/glusterfs/xieya/data/visual_genome/test.txt', 'w') as f_test:
        for img_idx in xrange(len(regions)):
            img = regions[img_idx]
            img_id = img['id']
            reg_num = len(img['regions'])
            # 85-5-10
            r = np.random.randint(100)
            if r < 85:
                fout = f_train
                train_img_count += 1
                train_reg_count += reg_num
            elif r < 90:
                fout = f_val
                val_img_count += 1
                val_reg_count += reg_num
            else:
                fout = f_test
                test_img_count += 1
                test_reg_count += reg_num
            fout.write('{0} {1} {2}\n'.format(img_id, img_idx, reg_num))
    print('Train set: image {0} region {1}'.format(train_img_count, train_reg_count))
    print('Val set: image {0} region {1}'.format(val_img_count, val_reg_count))
    print('Test set: image {0} region {1}'.format(test_img_count, test_reg_count))


if __name__ == "__main__":
    # build_vocabulary_by_spacy()
    # build_vocabulary_by_glove('glove.6B.50d.p')
    # filter_regions('glove.6B.50d', 'region_descriptions.json')
    # load_glove('glove.6B.300d.txt')
    scale_images('/srv/glusterfs/xieya/data/visual_genome/100k_2.txt', '/srv/glusterfs/xieya/data/visual_genome/VG_100K_224')
    # scale_regions('region_descriptions.json')
    # split_train_val_test('224_filtered_region_descriptions.json')
