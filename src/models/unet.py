import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, conv2d_transpose


class UNet:
    def __init__(self):
        pass

    def conv_block(self, input, filters):
        conv1 = conv2d(input, filters, kernel_size=(3,3), strides=(1, 1), activation=tf.nn.relu)
        conv2 = conv2d(conv1, filters, kernel_size=(3,3), strides=(1, 1), activation=tf.nn.relu)
        pool1 = max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2))

        return pool1

    def deconv_block(self, input, filters, conv):
        deconv1 = conv2d_transpose(input, filters, kernel_size=(3, 3), strides=(2, 2))
        deconv_shape = tf.shape(deconv1)
        conv_shape = tf.shape(conv)
        offsets = [0, (conv_shape[1]-deconv_shape[1])//2, (conv_shape[2]-deconv_shape[2])//2, 0]
        size = [-1, deconv_shape[1], deconv_shape[2], filters*2]
        conv_crop = tf.slice(conv, offsets, size)
        conv1 = tf.concat([deconv1, conv_crop], 3)

        conv2 = conv2d(conv1, filters, kernel_size=(3,3), activation=tf.nn.relu)
        conv3 = conv2d(conv2, filters, kernel_size=(3,3), activation=tf.nn.relu)

        return conv3

    def build(self, input):
        conv1 = self.conv_block(input, 64)
        conv2 = self.conv_block(conv1, 128)
        conv3 = self.conv_block(conv2, 256)
        conv4 = self.conv_block(conv3, 512)

        conv5 = conv2d(conv4, 1024, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu)
        conv5 = conv2d(conv5, 1024, kernel_size=(3, 3), strides=(1, 1), activation=tf.nn.relu)

        deconv4 = self.deconv_block(conv5, 512, conv4)
        deconv3 = self.deconv_block(deconv4, 256, conv3)
        deconv2 = self.deconv_block(deconv3, 128, conv2)
        deconv1 = self.deconv_block(deconv2, 64, conv1)

        self.output = conv2d(deconv1, filters=2, kernel_size=1)
