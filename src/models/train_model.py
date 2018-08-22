import tensorflow as tf
import numpy as np
import warnings
import random
import os
import sys
from unet import UNet

from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imsave, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore', category=UserWarning, module='skimage')

seed = 42
random.seed = seed
np.random.seed = seed

# train_filename_dataset = tf.data.Dataset.list_files('data/train/images/*.png')
# test_filename_dataset = tf.data.Dataset.list_files('data/test/images/*.png')
#
# train_images = train_filename_dataset.map(lambda x: tf.image.decode_png(tf.read_file(x)))
# test_images = test_filename_dataset.map(lambda x: tf.image.decode_png(tf.read_file(x)))
#
# iterator = train_images.make_one_shot_iterator()
# next_image = iterator.get_next()

IMG_SIZE = 101
IMG_CHANNELS = 1
path_train = '../../data/train'
path_test = '../../data/test'

train_ids = next(os.walk(path_train + "/images"))[2]
test_ids = next(os.walk(path_test + "/images"))[2]

X_train, X_val, _, _ = train_test_split(train_ids, train_ids, test_size=.2)
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

tf.reset_default_graph()
X = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, IMG_CHANNELS])
Y_ = tf.placeholder(tf.float32, [None, IMG_SIZE, IMG_SIZE, 1])
lr = tf.placeholder(tf.float32)
#
model = UNet()
model.build(X)
cross_entropy = tf.losses.sigmoid_cross_entropy(Y_, model.output)
train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy)

# init
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_count = 0
display_count = 1
epoch_loss = 0
best_loss = np.inf
saver = tf.train.Saver()
writer = tf.summary.FileWriter('../../logs')
writer.add_graph(tf.get_default_graph())

for i in range(50000):
    # training on batches of 50 images with 50 mask images
    if (batch_count > 79):
        epoch_loss = 0
        batch_count = 0

    batch_X, batch_Y = next_train_batch(50, batch_count, i == 0)

    batch_count += 1

    train_loss, _ = sess.run([cross_entropy, train_step], {X: batch_X, Y_: batch_Y, lr: 0.0005})
    epoch_loss += train_loss

    if (i % 200 == 0):
        print(str(display_count) + " training loss:" + str(train_loss))
        display_count += 1

# saver.save(sess, f"models/loss-{epoch_loss}")
# tf.saved_model.simple_save(sess,
#                            f"models/loss-{epoch_loss}",
#                            inputs={"x": X},
#                            outputs={"y_": Y_})
finalimg = tf.nn.sigmoid(model.output)
tests = list()
truthmask = list()
for i in range(40):
    batch_X, batch_Y = next_train_batch(50, i, i == 0)
    output = sess.run([finalimg], feed_dict={X:batch_X})
    for pred, gt in zip(output[0], batch_Y):
        tests.append(pred.reshape(101, 101))
        truthmask.append((gt*255).reshape(101, 101))

for i, img in enumerate(tests):
    imsave(f"../../reports/figures/{i}.png", img)
for i, gt in enumerate(truthmask):
    imsave(f"../../reports/figures/{i}_gt.png", gt)
print("Done!")

# epochs = 3

#
# with tf.Session() as sess:
#     for i in range(10):
#         image_array = sess.run([next_image])
#         print(image_array)