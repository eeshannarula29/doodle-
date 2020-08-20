import numpy as np
import consts
import cv2
import os

class Data:

    @staticmethod
    def getData():
        data = []
        for path in consts.getpaths():
            data.append(np.load(path))
        num,data = consts.reshape(data)
        Y = Data.getLabels(num)
        X = Data.concat(data)
        return X/255,Y

    @staticmethod
    def concat(array):
        final_array = []
        for CATIGORY in array:
            for img in CATIGORY:
                final_array.append(img)
        return np.array(final_array)

    @staticmethod
    def getLabels(num):
        labels = []
        for i in num:
            label = num.index(i)
            target = [0] * consts.classes
            target[label] = 1
            for j in range(i):
                labels.append(target)
        return np.array(labels)

    @staticmethod
    def FromOneHot(OneHotEncodedArray):

        normal_array = []
        for target in OneHotEncodedArray:
            normal_array.append(list(target).index(max(list(target))))
        return np.array(normal_array)
