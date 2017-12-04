# Copyright 2016 Google Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""
The core code for applying Canonical Correlation Analysis to deep networks.

This module contains the core functions to apply canonical correlation analysis
to deep neural networks. The main function is get_cca_similarity, which takes in
two sets of activations, typically the neurons in two layers and their outputs
on all of the datapoints D = [d_1,...,d_m] that have been passed through.

Inputs have shape (num_neurons1, m), (num_neurons2, m). This can be directly
applied used on fully connected networks. For convolutional layers, the 3d block
of neurons can either be flattened entirely, along channels, or alternatively,
the dft_ccas (Discrete Fourier Transform) module can be used.

See https://arxiv.org/abs/1706.05806 for full details.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
from keras import backend as K
import numpy as np
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils

num_cca_trials = 5
epsilon = 1e-6


def positivedef_matrix_sqrt(array):
  """Stable method for computing matrix square roots, supports complex matrices.

  Args:
            array: A numpy 2d array, can be complex valued that is a positive
                   definite symmetric (or hermitian) matrix

  Returns:
            sqrtarray: The matrix square root of array
  """
  w, v = np.linalg.eigh(array)
  #  A - np.dot(v, np.dot(np.diag(w), v.T))
  wsqrt = np.sqrt(w)
  sqrtarray = np.dot(v, np.dot(np.diag(wsqrt), np.conj(v).T))
  return sqrtarray


def remove_small(sigma_xx, sigma_xy, sigma_yx, sigma_yy, threshold=1e-6):
  """Takes covariance between X, Y, and removes values of small magnitude.

  Args:
            sigma_xx: 2d numpy array, variance matrix for x
            sigma_xy: 2d numpy array, crossvariance matrix for x,y
            sigma_yx: 2d numpy array, crossvariance matrixy for x,y,
                      (conjugate) transpose of sigma_xy
            sigma_yy: 2d numpy array, variance matrix for y
            threshold: cutoff value for norm below which directions are thrown
                       away

  Returns:
            sigma_xx_crop: 2d array with low x norm directions removed
            sigma_xy_crop: 2d array with low x and y norm directions removed
            sigma_yx_crop: 2d array with low x and y norm directiosn removed
            sigma_yy_crop: 2d array with low y norm directions removed
            x_idxs: indexes of sigma_xx that were removed
            y_idxs: indexes of sigma_yy that were removed
  """

  x_diag = np.abs(np.diagonal(sigma_xx))
  y_diag = np.abs(np.diagonal(sigma_yy))
  x_idxs = (x_diag >= threshold)
  y_idxs = (y_diag >= threshold)

  sigma_xx_crop = sigma_xx[x_idxs][:, x_idxs]
  sigma_xy_crop = sigma_xy[x_idxs][:, y_idxs]
  sigma_yx_crop = sigma_yx[y_idxs][:, x_idxs]
  sigma_yy_crop = sigma_yy[y_idxs][:, y_idxs]

  return (sigma_xx_crop, sigma_xy_crop, sigma_yx_crop, sigma_yy_crop, x_idxs,
          y_idxs)


def compute_ccas(sigma_xx, sigma_xy, sigma_yx, sigma_yy, verbose=True):
  """Main cca computation function, takes in variances and crossvariances.

  This function takes in the covariances and cross covariances of X, Y,
  preprocesses them (removing small magnitudes) and outputs the raw results of
  the cca computation, including cca directions in a rotated space, and the
  cca correlation coefficient values.

  Args:
            sigma_xx: 2d numpy array, (num_neurons_x, num_neurons_x)
                      variance matrix for x
            sigma_xy: 2d numpy array, (num_neurons_x, num_neurons_y)
                      crossvariance matrix for x,y
            sigma_yx: 2d numpy array, (num_neurons_y, num_neurons_x)
                      crossvariance matrix for x,y (conj) transpose of sigma_xy
            sigma_yy: 2d numpy array, (num_neurons_y, num_neurons_y)
                      variance matrix for y
            verbose:  boolean on whether to print intermediate outputs

  Returns:
            [ux, sx, vx]: [numpy 2d array, numpy 1d array, numpy 2d array]
                          ux and vx are (conj) transposes of each other, being
                          the canonical directions in the X subspace.
                          sx is the set of canonical correlation coefficients-
                          how well corresponding directions in vx, Vy correlate
                          with each other.
            [uy, sy, vy]: Same as above, but for Y space
            invsqrt_xx:   Inverse square root of sigma_xx to transform canonical
                          directions back to original space
            invsqrt_yy:   Same as above but for sigma_yy
            x_idxs:       The indexes of the input sigma_xx that were pruned
                          by remove_small
            y_idxs:       Same as above but for sigma_yy
  """

  (sigma_xx, sigma_xy, sigma_yx, sigma_yy, x_idxs, y_idxs) = remove_small(
      sigma_xx, sigma_xy, sigma_yx, sigma_yy)

  numx = sigma_xx.shape[0]
  numy = sigma_yy.shape[0]

  if numx == 0 or numy == 0:
    return ([0, 0, 0], [0, 0, 0], np.zeros_like(sigma_xx),
            np.zeros_like(sigma_yy), x_idxs, y_idxs)

  if verbose:
    print("adding eps to diagonal and taking inverse")
  sigma_xx +=epsilon * np.eye(numx)
  sigma_yy +=epsilon * np.eye(numy)
  inv_xx = np.linalg.pinv(sigma_xx)
  inv_yy = np.linalg.pinv(sigma_yy)

  if verbose:
    print("taking square root")
  invsqrt_xx = positivedef_matrix_sqrt(inv_xx)
  invsqrt_yy = positivedef_matrix_sqrt(inv_yy)

  if verbose:
    print("dot products...")
  arr_x = np.dot(sigma_yx, invsqrt_xx)
  arr_x = np.dot(inv_yy, arr_x)
  arr_x = np.dot(invsqrt_xx, np.dot(sigma_xy, arr_x))
  arr_y = np.dot(sigma_xy, invsqrt_yy)
  arr_y = np.dot(inv_xx, arr_y)
  arr_y = np.dot(invsqrt_yy, np.dot(sigma_yx, arr_y))

  if verbose:
    print("trying to take final svd")
  arr_x_stable = arr_x + epsilon * np.eye(arr_x.shape[0])
  arr_y_stable = arr_y + epsilon * np.eye(arr_y.shape[0])
  ux, sx, vx = np.linalg.svd(arr_x_stable)
  uy, sy, vy = np.linalg.svd(arr_y_stable)

  sx = np.sqrt(np.abs(sx))
  sy = np.sqrt(np.abs(sy))
  if verbose:
    print("computed everything!")

  return [ux, sx, vx], [uy, sy, vy], invsqrt_xx, invsqrt_yy, x_idxs, y_idxs


def sum_threshold(array, threshold):
  """Computes threshold index of decreasing nonnegative array by summing.

  This function takes in a decreasing array nonnegative floats, and a
  threshold between 0 and 1. It returns the index i at which the sum of the
  array up to i is threshold*total mass of the array.

  Args:
            array: a 1d numpy array of decreasing, nonnegative floats
            threshold: a number between 0 and 1

  Returns:
            i: index at which np.sum(array[:i]) >= threshold
  """
  assert (threshold >= 0) and (threshold <= 1), "print incorrect threshold"

  for i in range(len(array)):
    if np.sum(array[:i]) / np.sum(array) >= threshold:
      return i


def create_zero_dict(compute_dirns, dimension):
  """Outputs a zero dict when neuron activation norms too small.

  This function creates a return_dict with appropriately shaped zero entries
  when all neuron activations are very small.

  Args:
            compute_dirns: boolean, whether to have zero vectors for directions
            dimension: int, defines shape of directions

  Returns:
            return_dict: a dict of appropriately shaped zero entries
  """
  return_dict = {}
  return_dict["mean"] = (np.asarray(0), np.asarray(0))
  return_dict["sum"] = (np.asarray(0), np.asarray(0))
  return_dict["cca_coef1"] = np.asarray(0)
  return_dict["cca_coef2"] = np.asarray(0)
  return_dict["idx1"] = 0
  return_dict["idx2"] = 0

  if compute_dirns:
    return_dict["cca_dirns1"] = np.zeros((1, dimension))
    return_dict["cca_dirns2"] = np.zeros((1, dimension))

  return return_dict


def get_cca_similarity(acts1, acts2, threshold=0.98, compute_dirns=True,
                       verbose=True):
  """The main function for computing cca similarities.

  This function computes the cca similarity between two sets of activations,
  returning a dict with the cca coefficients, a few statistics of the cca
  coefficients, and (optionally) the actual directions.

  Args:
            acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                   datapoints where entry (i,j) is the output of neuron i on
                   datapoint j.
            acts2: (num_neurons2, data_points) same as above, but (potentially)
                   for a different set of neurons. Note that acts1 and acts2
                   can have different numbers of neurons, but must agree on the
                   number of datapoints

            threshold: float between 0, 1 used to get rid of trailing zeros in
                       the cca correlation coefficients to output more accurate
                       summary statistics of correlations.

            compute_dirns: boolean value determining whether actual cca
                           directions are computed. (For very large neurons and
                           datasets, may be better to compute these on the fly
                           instead of store in memory.)

            verbose: Boolean, whether info about intermediate outputs printed

  Returns:
            return_dict: A dictionary with outputs from the cca computations.
                         Contains neuron coefficients (combinations of neurons
                         that correspond to cca directions), the cca correlation
                         coefficients (how well aligned directions correlate),
                         x and y idxs (for computing cca directions on the fly
                         if compute_dirns=False), and summary statistics. If
                         compute_dirns=True, the cca directions are also
                         computed.
  """

  # assert dimensionality equal
  assert acts1.shape[1] == acts2.shape[1], "dimensions don't match"
  # check that acts1, acts2 are transposition
  #assert acts1.shape[0] < acts1.shape[1], ("input must be number of neurons by datapoints")
  return_dict = {}

  # compute covariance with numpy function for extra stability
  numx = acts1.shape[0]

  covariance = np.cov(acts1, acts2)
  sigmaxx = covariance[:numx, :numx]
  sigmaxy = covariance[:numx, numx:]
  sigmayx = covariance[numx:, :numx]
  sigmayy = covariance[numx:, numx:]

  # rescale covariance to make cca computation more stable
  xmax = np.max(np.abs(sigmaxx))
  ymax = np.max(np.abs(sigmayy))
  sigmaxx /= xmax
  sigmayy /= ymax
  sigmaxy /= np.sqrt(xmax * ymax)
  sigmayx /= np.sqrt(xmax * ymax)

  ([_, sx, vx], [_, sy, vy], invsqrt_xx, invsqrt_yy, x_idxs,
   y_idxs) = compute_ccas(sigmaxx, sigmaxy, sigmayx, sigmayy,
                          verbose)

  # if x_idxs or y_idxs is all false, return_dict has zero entries
  if (not np.any(x_idxs)) or (not np.any(y_idxs)):
    return create_zero_dict(compute_dirns, acts1.shape[1])

  if compute_dirns:
    # orthonormal directions that are CCA directions
    cca_dirns1 = np.dot(vx, np.dot(invsqrt_xx, acts1[x_idxs]))
    cca_dirns2 = np.dot(vy, np.dot(invsqrt_yy, acts2[y_idxs]))

  # get rid of trailing zeros in the cca coefficients
  idx1 = sum_threshold(sx, threshold)
  idx2 = sum_threshold(sy, threshold)

  return_dict["neuron_coeffs1"] = np.dot(vx, invsqrt_xx)
  return_dict["neuron_coeffs2"] = np.dot(vy, invsqrt_yy)
  return_dict["cca_coef1"] = sx
  return_dict["cca_coef2"] = sy
  return_dict["x_idxs"] = x_idxs
  return_dict["y_idxs"] = y_idxs
  # summary statistics
  return_dict["mean"] = (np.mean(sx[:idx1]), np.mean(sy[:idx2]))
  return_dict["sum"] = (np.sum(sx), np.sum(sy))

  if compute_dirns:
    return_dict["cca_dirns1"] = cca_dirns1
    return_dict["cca_dirns2"] = cca_dirns2

  return return_dict


def robust_cca_similarity(acts1, acts2, threshold=0.98, compute_dirns=True):
  """Calls get_cca_similarity multiple times while adding noise.

  This function is very similar to get_cca_similarity, and can be used if
  get_cca_similarity doesn't converge for some pair of inputs. This function
  adds some noise to the activations to help convergence.

  Args:
            acts1: (num_neurons1, data_points) a 2d numpy array of neurons by
                   datapoints where entry (i,j) is the output of neuron i on
                   datapoint j.
            acts2: (num_neurons2, data_points) same as above, but (potentially)
                   for a different set of neurons. Note that acts1 and acts2
                   can have different numbers of neurons, but must agree on the
                   number of datapoints

            threshold: float between 0, 1 used to get rid of trailing zeros in
                       the cca correlation coefficients to output more accurate
                       summary statistics of correlations.

            compute_dirns: boolean value determining whether actual cca
                           directions are computed. (For very large neurons and
                           datasets, may be better to compute these on the fly
                           instead of store in memory.)

  Returns:
            return_dict: A dictionary with outputs from the cca computations.
                         Contains neuron coefficients (combinations of neurons
                         that correspond to cca directions), the cca correlation
                         coefficients (how well aligned directions correlate),
                         x and y idxs (for computing cca directions on the fly
                         if compute_dirns=False), and summary statistics. If
                         compute_dirns=True, the cca directions are also
                         computed.
  """

  for trial in range(num_cca_trials):
    try:
      return_dict = get_cca_similarity(acts1, acts2, threshold, compute_dirns)
    except np.LinAlgError:
      acts1 = acts1 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
      acts2 = acts2 * 1e-1 + np.random.normal(size=acts1.shape) * epsilon
      if trial + 1 == num_cca_trials:
        raise

  return return_dict

def fft_resize(images, resize=False, new_size=None):
  """Function for applying DFT and resizing.
  This function takes in an array of images, applies the 2-d fourier transform
  and resizes them according to new_size, keeping the frequencies that overlap
  between the two sizes.
  Args:
            images: a numpy array with shape
                    [batch_size, height, width, num_channels]
            resize: boolean, whether or not to resize
            new_size: a tuple (size, size), with height and width the same
  Returns:
            im_fft_downsampled: a numpy array with shape
                         [batch_size, (new) height, (new) width, num_channels]
  """
  assert len(images.shape) == 4, ("expecting images to be"
                                  "[batch_size, height, width, num_channels]")

  im_complex = images.astype("complex64")
  im_fft = np.fft.fft2(im_complex, axes=(1, 2))

  # resizing images
  if resize:
    # get fourier frequencies to threshold
    assert (im_fft.shape[1] == im_fft.shape[2]), ("Need images to have same"
                                                  "height and width")
    # downsample by threshold
    width = im_fft.shape[2]
    new_width = new_size[0]
    freqs = np.fft.fftfreq(width, d=1.0 / width)
    idxs = np.flatnonzero((freqs >= -new_width / 2.0) & (freqs <
                                                         new_width / 2.0))
    im_fft_downsampled = im_fft[:, :, idxs, :][:, idxs, :, :]

  else:
    im_fft_downsampled = im_fft

  return im_fft_downsampled


def fourier_ccas(conv_acts1, conv_acts2, return_coefs=False,compute_dirns=False, verbose=False):
  """Computes cca similarity between two conv layers with DFT.
  This function takes in two sets of convolutional activations, conv_acts1,
  conv_acts2 After resizing the spatial dimensions to be the same, applies fft
  and then computes the ccas.
  Finally, it applies the inverse fourier transform to get the CCA directions
  and neuron coefficients.
  Args:
            conv_acts1: numpy array with shape
                        [batch_size, height1, width1, num_channels1]
            conv_acts2: numpy array with shape
                        [batch_size, height2, width2, num_channels2]
            compute_dirns: boolean, used to determine whether results also
                           contain actual cca directions.
  Returns:
            all_results: a pandas dataframe, with cca results for every spatial
                         location. Columns are neuron coefficients (combinations
                         of neurons that correspond to cca directions), the cca
                         correlation coefficients (how well aligned directions
                         correlate) x and y idxs (for computing cca directions
                         on the fly if compute_dirns=False), and summary
                         statistics. If compute_dirns=True, the cca directions
                         are also computed.
  """

  height1, width1 = conv_acts1.shape[1], conv_acts1.shape[2]
  height2, width2 = conv_acts2.shape[1], conv_acts2.shape[2]
  if height1 != height2 or width1 != width2:
    height = min(height1, height2)
    width = min(width1, width2)
    new_size = [height, width]
    resize = True
  else:
    height = height1
    width = width1
    new_size = None
    resize = False

  # resize and preprocess with fft
  fft_acts1 = fft_resize(conv_acts1, resize=resize, new_size=new_size)
  fft_acts2 = fft_resize(conv_acts2, resize=resize, new_size=new_size)

  # loop over spatial dimensions and get cca coefficients
  all_results = pd.DataFrame()
  for i in range(height):
    for j in range(width):
        results_dict = get_cca_similarity(
        fft_acts1[:, i, j, :].T, fft_acts2[:, i, j, :].T, compute_dirns,
                                                            verbose=verbose)

      # apply inverse FFT to get coefficients and directions if specified
        if return_coefs:
              results_dict["neuron_coeffs1"] = np.fft.ifft2(results_dict["neuron_coeffs1"])
              results_dict["neuron_coeffs2"] = np.fft.ifft2(results_dict["neuron_coeffs2"])
        else:
          del results_dict["neuron_coeffs1"]
          del results_dict["neuron_coeffs2"]

        if compute_dirns:
            results_dict["cca_dirns1"] = np.fft.ifft2(results_dict["cca_dirns1"])
            results_dict["cca_dirns2"] = np.fft.ifft2(results_dict["cca_dirns2"])

      # accumulate results
        results_dict["location"] = (i, j)
        all_results = all_results.append(results_dict, ignore_index=True)

  return all_results

def get_activations(model, model_inputs, layer_name=None):
    activations = []
    inp = model.input
    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False
    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs
    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions
    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(0.)
    else:
        list_inputs = [model_inputs, 0.]
    # Learning phase. 0 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 0.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
    return activations

def get_acts_from_model(model1,x_data):
    acts = get_activations(model1,x_data)
    for i in range(len(acts)):
        act_shape = acts[i].shape
        dim2 = 1
        for j in range(len(act_shape)-1):
            dim2 = dim2*act_shape[j+1]
        acts[i] = acts[i].reshape(act_shape[0],dim2)
    acts1 = np.transpose(np.concatenate(acts,axis=1))
    acts2 = np.around(acts1,decimals=3)
    return(acts2)
    
def output_model(history):
    model_info = pd.DataFrame({'val_loss':history.history['val_loss'],'val_acc':history.history['val_acc'],'loss':history.history['loss'],'acc':history.history['acc']})
    return(model_info)

# Define the model
def get_cnn_model(input_shape,num_classes):
    model = Sequential()
    model.add(Conv2D(2, (3, 3), padding='same',activation='relu',strides=(3, 3), input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(16))
    model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return(model)
    
    
def get_dense_model(input_shape,num_classes):
    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Conv2D(1, (1, 1),strides=(3, 3), padding='same',activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return(model)