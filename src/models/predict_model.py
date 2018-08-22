from unet import UNet
import tensorflow as tf
import numpy as np
import os
import sys

from tqdm import tqdm
from skimage.io import imread, imshow, imread_collection, concatenate_images


IMG_SIZE = 101
IMG_CHANNELS = 1
path_train = '../../data/train'
path_test = '../../data/test'

train_ids = next(os.walk(path_train + "/images"))[2]
test_ids = next(os.walk(path_test + "/images"))[2]

train_images = np.zeros((len(train_ids)*2, IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.uint8)
train_labels = np.zeros((len(train_ids)*2, IMG_SIZE, IMG_SIZE, 1), dtype=np.bool)

print('Getting and resizing train images and masks without padding ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    img = imread(path_train + "/images/" + id_)[:,:,:IMG_CHANNELS]
    train_images[n*2] = img
    train_images[2*n+1] = np.fliplr(img)

    mask = imread(path_train + "/masks/" + id_)
    mask = np.expand_dims(mask, axis = -1)

    train_labels[2*n] = mask
    train_labels[2*n+1] = np.fliplr(mask)

test_images = np.zeros((len(test_ids), IMG_SIZE, IMG_SIZE, IMG_CHANNELS), dtype=np.uint8)
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    img = imread(path_test + "/images/" + id_)[:,:,:IMG_CHANNELS]
    test_images[n] = img

def shuffle():
    global images, labels

    p = np.random.permutation(len(train_ids))
    images = train_images[p]
    labels = train_labels[p]

def next_train_batch(batch_s, batch_count, is_first_iter):
    if(batch_count == 0):
        shuffle()
    count = batch_s * batch_count
    return images[count:(count + batch_s)], labels[count:(count + batch_s)]

X = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, IMG_CHANNELS])
Y_ = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])

with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('../../models/loss-0.0024245440383765526.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('../../models/'))
    op = sess.graph.get_operations()

    logits = sess.run(['logits/BiasAdd:0'], feed_dict={X: train_images[:10]})
    for m in op:
        print(m.values())