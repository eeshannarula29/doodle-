import os
import numpy as np

CAT = 0
TRAIN = 1
RAINBOW = 2

CATIGORIES = [
'cat',
'train',
'rainbow'
]

ext = '.npy'

classes = 3

prepath = os.getcwd()


def getpaths():

    PATHS = []

    for CATIGORY in CATIGORIES:
        postpath = CATIGORY + ext
        PATHS.append(os.path.join(prepath,postpath))

    return PATHS

DIM1 = 28
DIM2 = 28
streams = 1

shape_sigle = (DIM1,DIM2)
input_shape = (DIM1,DIM2,1)

for_one = (1,DIM1,DIM2,1)

def reshape(x):
    reshaped = []
    nums = []
    for i in x:
        shape = i.shape
        nums.append(shape[0])
        reshaped.append(np.reshape(i,(i.shape[0],DIM1,DIM2,streams)))
    return nums,reshaped

def get_doodle_prediction(prediction):
     return CATIGORIES[(list(prediction[0]).index(max(list(prediction[0]))))]

lr = 0.01
loss = 'categorical_crossentropy'
activation = 'softmax'
epochs = 1
