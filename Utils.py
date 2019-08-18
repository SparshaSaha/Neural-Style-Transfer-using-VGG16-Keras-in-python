import numpy as np
import os
import time

os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

import keras
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img, img_to_array
from skimage.transform import resize
class Utils(object):

    def __init__(self, imageH, imageW):
        self.imageH = imageH
        self.imageW = imageW
    
    def preprocessImage(self, imagePath):
        image = load_img(imagePath)
        imageArray = img_to_array(image)
        resizedImage = resize(imageArray, (self.imageH, self.imageW, 3))
        resizedImage = resizedImage.astype('float64')
        resizedImage4D = np.expand_dims(resizedImage, axis=0)
        imageForVGG = preprocess_input(resizedImage4D)
        return imageForVGG
    
    def unpreprocess(self, image):
        image[..., 0] += 103.939
        image[..., 1] += 116.779
        image[..., 2] += 126.68
        image = image[..., ::-1]
        return image
    
    def scaleImage(self, image):
        image = image - image.min()
        image = image / image.max()
        return image
