import os

import tensorflow as tf


IMG_SIZE = 224
IMG_DIR = '/srv/glusterfs/xieya/data/imagenet1k_uncompressed/val'
NUM_IMGS = 100


img_list = os.listdir(IMG_DIR)
img_list.sort()

img_count = 0
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    model = tf.keras.applications.vgg16.VGG16()
    for img_name in img_list:
        if not img_name.endswith('.JPEG'):
            continue
        print(img_name)
        img_path = os.path.join(IMG_DIR, img_name)
        image = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
        image = tf.keras.preprocessing.image.img_to_array(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        image = tf.keras.applications.vgg16.preprocess_input(image)
        yhat = model.predict(image)
        label = tf.keras.applications.vgg16.decode_predictions(yhat, top=1)
        label = label[0][0]
        print('%s %s (%.2f%%)' % (label[0], label[1], label[2]*100))

        if img_count == NUM_IMGS:
            break
