

from __future__ import print_function, division
from builtins import range, input

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend"

from keras.layers import Input, Lambda, Dense, Flatten
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from skimage.transform import resize
from keras.preprocessing.image import save_img
from datetime import datetime
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
from Utils import Utils

utils = Utils(420, 420)

def VGGAveragePool(shape):
  vgg = VGG16(input_shape=shape, weights='imagenet', include_top=False)
  new_model = Sequential()
  for layer in vgg.layers:
    if layer.__class__ == MaxPooling2D:
      new_model.add(AveragePooling2D())
    else:
      new_model.add(layer)

  return new_model

def getVGGModifiedModel(shape, num_convs):
  model = VGGAveragePool(shape)
  new_model = Sequential()
  n = 0
  for layer in model.layers:
    if layer.__class__ == Conv2D:
      n += 1
    new_model.add(layer)
    if n >= num_convs:
      break

  return new_model

def gram_matrix(img):
  X = K.batch_flatten(K.permute_dimensions(img, (2, 0, 1)))
  G = K.dot(X, K.transpose(X))/ (img.shape.dims[0]*img.shape.dims[1]*img.shape.dims[2])
  return G


def style_loss(y, t):
  return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


def minimize(fn, epochs, batch_shape):
  t0 = datetime.now()
  losses = []
  x = np.random.randn(np.prod(batch_shape))
  for i in range(epochs):
    x, l, _ = fmin_l_bfgs_b(
      func=fn,
      x0=x,
      maxfun=20
    )
    x = np.clip(x, -127, 127)
    print("iter=%s, loss=%s" % (i, l))
    losses.append(l)

  print("duration:", datetime.now() - t0)
  plt.plot(losses)
  plt.show()

  newimg = x.reshape(*batch_shape)
  final_img = utils.unpreprocess(newimg)
  return final_img[0]

def getStyleTransferredImage(contentImagePath, styleImagePath, savePath):

  content_img = utils.preprocessImage(contentImagePath)
  h, w = content_img.shape[1:3]
  style_img = utils.preprocessImage(styleImagePath)

  batch_shape = content_img.shape
  shape = content_img.shape[1:]
  vgg = VGGAveragePool(shape)

  content_model = Model(vgg.input, vgg.layers[13].get_output_at(0))
  content_model.summary()
  content_target = K.variable(content_model.predict(content_img))


  symbolic_conv_outputs = [
    layer.get_output_at(1) for layer in vgg.layers \
    if layer.name.endswith('conv1')
  ]

  style_model = Model(vgg.input, symbolic_conv_outputs)

  style_layers_outputs = [K.variable(y) for y in style_model.predict(style_img)]

  style_weights = [0.02,0.07,0.4,0.3,0.2]

  loss =  K.mean(K.square(content_model.output - content_target))

  for w, symbolic, actual in zip(style_weights, symbolic_conv_outputs, style_layers_outputs):
    loss += w * style_loss(symbolic[0], actual[0])

  grads = K.gradients(loss, vgg.input)
  get_loss_and_grads = K.function(
    inputs=[vgg.input],
    outputs=[loss] + grads
  )


  def get_loss_and_grads_wrapper(x_vec):
    l, g = get_loss_and_grads([x_vec.reshape(*batch_shape)])
    return l.astype(np.float64), g.flatten().astype(np.float64)


  final_img = minimize(get_loss_and_grads_wrapper, 10, batch_shape)
  plt.imshow(utils.scaleImage(final_img))
  plt.show()
  save_img(savePath, final_img)

getStyleTransferredImage("./ContentImages/India.png", "./StyleImages/abstractHumans.jpg", "StylizedImages/IndiaAbstractHumans.jpg")
