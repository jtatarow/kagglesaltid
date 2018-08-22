import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, conv2d_transpose, dropout, batch_normalization
from tensorflow.python.ops import array_ops


class UNet:
    def __init__(self):
        pass

    def conv_block(self, input, filters, scope):
        with tf.variable_scope(scope):
            conv1 = conv2d(input, filters, kernel_size=(3,3), activation=tf.nn.relu, padding='SAME')
            conv2 = conv2d(conv1, filters, kernel_size=(3,3), activation=tf.nn.relu, padding='SAME')
            bn = batch_normalization(conv2)

        return bn

    def deconv_block(self, input, filters, conv, padding, scope):
        with tf.variable_scope(scope):
            deconv1 = conv2d_transpose(input, filters, kernel_size=(3, 3), strides=[2, 2], padding=padding)
            deconv_shape = tf.shape(deconv1)
            conv_shape = tf.shape(conv)
            offsets = [0, (conv_shape[1]-deconv_shape[1])//2, (conv_shape[2]-deconv_shape[2])//2, 0]
            size = [-1, deconv_shape[1], deconv_shape[2], filters]
            conv_crop = tf.slice(conv, offsets, size)
            conv1 = tf.concat([deconv1, conv_crop], 3)
            bn = batch_normalization(conv1)
            drop = dropout(bn, .25)
            conv2 = conv2d(drop, filters, kernel_size=(3,3), activation=tf.nn.relu, name='middle1', padding="SAME")
            conv3 = conv2d(conv2, filters, kernel_size=(3,3), activation=tf.nn.relu, name='middle2', padding="SAME")

        return conv3

    def build(self, input):
        self.conv1 = self.conv_block(input, 64, "conv1")
        pool1 = max_pooling2d(self.conv1, pool_size=(2, 2), strides=2)
        drop1 = dropout(pool1, .25)
        self.conv2 = self.conv_block(drop1, 128, 'conv2')
        pool2 = max_pooling2d(self.conv2, pool_size=(2, 2), strides=2)
        drop2 = dropout(pool2, .25)
        self.conv3 = self.conv_block(drop2, 256, 'conv3')
        pool3 = max_pooling2d(self.conv3, pool_size=(2, 2), strides=2)
        drop3 = dropout(pool3, .25)
        self.conv4 = self.conv_block(drop3, 512, 'conv4')
        pool4 = max_pooling2d(self.conv4, pool_size=(2, 2), strides=2)
        drop4 = dropout(pool4, .25)

        self.conv5 = conv2d(drop4, 1024, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME')
        self.conv5_2 = conv2d(self.conv5, 1024, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME')

        self.deconv4 = self.deconv_block(self.conv5_2, 512, self.conv4, "SAME", 'deconv4')
        self.deconv3 = self.deconv_block(self.deconv4, 256, self.conv3, "VALID", 'deconv3')
        self.deconv2 = self.deconv_block(self.deconv3, 128, self.conv2, "SAME", 'deconv2')
        self.deconv1 = self.deconv_block(self.deconv2, 64, self.conv1, "VALID", 'deconv1')
        self.output = conv2d(self.deconv1, filters=1, kernel_size=1, name='logits')
