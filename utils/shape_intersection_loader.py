import numpy as np
import time
import random
import math
import os
import sys
from multiprocessing import Process, Queue
import _pickle as cPickle
import io
import pdb
import gizeh


def draw(shape, pixel, filename=None):
    resize = pixel / 10
    surface = gizeh.Surface(width=pixel, height=pixel, bg_color=(0, 0, 0))  # in pixels
    if shape[0] == 1:
        line = gizeh.polyline(
            points=[(resize * shape[1], resize * shape[2]), (resize * shape[3], resize * shape[4])],
            stroke_width=1,
            stroke=(1, 1, 1))
        line.draw(surface)
    if shape[0] == 2:
        circle = gizeh.circle(r=resize * shape[3], xy=[resize * shape[1], resize * shape[2]], stroke=(1, 1, 1),
                              stroke_width=1)
        circle.draw(surface)
    if shape[0] == 3:
        rect = gizeh.rectangle(lx=resize * shape[3], ly=resize * shape[4], xy=(
            resize * shape[1] + resize * shape[3] / 2, resize * shape[2] + resize * shape[4] / 2),
                               stroke=(1, 1, 1),
                               stroke_width=1, angle=0)
        rect.draw(surface)
    img = surface.get_npimage()[:, :, 0]  # returns a (width x height) numpy array
    if filename is not None:
        surface.write_to_png(filename + ".png")
    return img


def int2onehot(lst, dim):
    array = np.zeros((len(lst), dim), dtype=np.double)
    for i in range(len(lst)):
        array[i][lst(i)] = 1
    return array


class Loader(object):
    def __init__(self, path, batch_size, mask_range, pixel, ques_len, maxsize=20, embedding_size=256, shuffle=True):
        self.path = path
        self.shuffle = shuffle
        self.ques_len = ques_len
        self.batch_size = batch_size
        self.epoch = 1
        self.batch_idx = -1  # conj, stmt pairs supply in current epoch.
        self.total_iter = -1  # total iteration
        self.maxsize = maxsize
        self.queue = Queue(self.maxsize)
        self.reader = Process(target=self.read)
        self.reader.daemon = True
        self.total_size = 12800
        self.total_batch = int(self.total_size / self.batch_size)
        self.pixel = pixel
        self.mask_range = mask_range
        self.embedding_size = embedding_size
        self.count = 0
        self.draw = True
        self.reader.start()

    def next_batch(self):
        data = self.queue.get()
        if data is None:
            self.epoch += 1
            self.batch_idx = 0
        else:
            self.batch_idx += 1
            self.total_iter += 1
        return data

    def read(self):
        with open(self.path, 'rb') as f:
            a = cPickle.load(f)
        while True:
            if self.shuffle:
                random.shuffle(a)
            bs = self.batch_size
            l = len(a)
            # print('file size:', l)
            for i in range(l // bs):
                self.queue.put(self.prepare(a[i * bs: (i + 1) * bs]))

    def prepare(self, data):
        X = np.zeros((self.batch_size, 12, 5))
        X_length = np.zeros((self.batch_size, 12))
        Y = np.zeros((self.batch_size, ))
        num_sen = np.zeros((self.batch_size, ))
        mask = np.zeros((self.batch_size, 12))
        Ques = np.zeros((self.batch_size, self.ques_len, self.embedding_size))  # no information at all
        Ques_length = np.ones((self.batch_size, )) * self.ques_len
        img = np.zeros((self.batch_size, self.pixel, self.pixel, 12))
        loss_mask = np.zeros((self.batch_size, self.pixel, self.pixel, 12))

        l = np.array(list(map(lambda x: len(x[0]), data)))

        for i in range(self.batch_size):
            single_X = np.array(data[i][0])
            X[i, :single_X.shape[0], :single_X.shape[1]] = single_X
            X_length[i,:single_X.shape[0]] = 5
            Y[i] = data[i][1]
            num_sen[i] = single_X.shape[0]
            mask[i,:single_X.shape[0]] = 1
            # image channels for middle supervision
            for j in range(l[i]):
                img[i, :, :, j] = draw(data[i][0][j], self.pixel) / 255
                loss_mask[i, :, :, j] = (self.mask_range - 1) * img[i, :, :, j] + 1

        return X, X_length, Y, num_sen, mask, Ques, Ques_length, img, loss_mask

    def destruct(self):
        self.reader.terminate()
