#!/usr/bin/env python
"""
Python functions used for my research internship.

# INDEX
    # PARAMETERS
    # GENERATION OF DATASET
    # CONVOLUTIONAL FILTER ANALYSIS
    # VISUAL ANALYSIS
    # NEURAL NETWORK MODELS
    # CONTROL MODEL
"""
__author__ = "Elias Aoun Durand"
__email__ = "elias.aoundurand@gmail.com"

from numpy import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf

from numba import jit
import cv2
import cPickle as pickle

import numpy as np
import matplotlib.pyplot as plt
import keras

import time
import random
import scipy
import math
import matplotlib.animation as animation
import tqdm

from keras.models import Model
from keras.layers import Activation, Dense, Input, Multiply
from keras.layers import Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.layers import Dot, Lambda, Concatenate, RepeatVector
from keras.utils import plot_model
from PIL import Image
from keras.constraints import max_norm, non_neg


################################################################################

# PARAMETERS

################################################################################

matplotlib.rcParams.update({'font.size': 16})

L1 = 0.28
L2 = 0.28
L3 = 0.09
IMG_SIZE = 128
INPUT_ENCODER_SHAPE = (IMG_SIZE, IMG_SIZE, 2)
LATENT_DIM = 32
INPUT_DECODER_SHAPE = (1, 2 * LATENT_DIM,)

NB_POSTURE = 50
NB_COMMAND = 100
NB_DATA = NB_POSTURE*NB_COMMAND
BATCH_SIZE = 100
TEST_BUF = 1000

DIMS = (IMG_SIZE, IMG_SIZE,2)
N_TRAIN_BATCHES =int(NB_DATA/BATCH_SIZE)
N_TEST_BATCHES = int(TEST_BUF/BATCH_SIZE)

################################################################################

# GENERATION OF DATASET

################################################################################


def randrange(n, vmin, vmax):
    return (vmax - vmin) * rand(n) + vmin



def control_robot(angles):
    """function calculating end effector postion from joint angles

    Args :
        phi1, phi2, theta1, theta2 : joint angles (4 dof arm )

    Returns:
        x,y,z : end effector position
        vx,vy,vz : speed i guess...
    """
    phi1, phi2, theta1, psi1, psi2 = angles
    x = L1*cos(phi1)*cos(phi2)+L2*cos(phi1)*cos(phi2+theta1)+ L3*cos(psi1)*cos(psi2)
    y =  L1*sin(phi1)*cos(phi2)+L2*sin(phi1)*cos(phi2+theta1) + L3*cos(psi1)*sin(psi2) # ELBOW + HAND
    z = L1 * sin(phi2) + L2 * sin(phi2 ) + L3*sin(psi1)  # ELBOW + HAND

    return np.array([x, y,z])

def control_robot_elbow(angles):
    phi1, phi2, theta1, psi1, psi2 = angles
    x = L1 * cos(phi1) * cos(phi2)
    y = L1 * cos(phi1) * sin(phi2)
    z = L1 * sin(phi1)
    return np.array([x,y,z])

def compute_trajectory(postures):
    tmp = []
    for i in range(len(postures)):
        tmp.append(control_robot(postures[i][0]))
    return np.array(tmp)


def compute_elbow_trajectory(postures):
    tmp = []
    for i in range(len(postures)):
        tmp.append(control_robot_elbow(postures[i][0]))
    return np.array(tmp)


def plot_arm(angles, time):
    """function ploting and saving in images/ folder 3d arm plots
    from arg joint state input.

    Args :
        phi1, phi2, theta1 : joint angles (3 dof arm )

    Returns :
        ax : a matplotlib figure object
    """
    phi1, phi2, theta1, psi1, psi2 = angles
    filename = 'images/%s.png' % time

    x = [0, 0, L1 * cos(phi1) * cos(phi2),
         L1*cos(phi1)*cos(phi2)+L2*cos(phi1)*cos(phi2+theta1),
         L1*cos(phi1)*cos(phi2)+L2*cos(phi1)*cos(phi2+theta1)+ L3*cos(phi2+theta1+psi1)*cos(phi1)]
    y = [0, 0, L1 * sin(phi1) * cos(phi2),
         L1*sin(phi1)*cos(phi2)+L2*sin(phi1)*cos(phi2+theta1),
         L1*sin(phi1)*cos(phi2)+L2*sin(phi1)*cos(phi2+theta1) + L3*cos(psi1+phi2+theta1)*sin(phi1)]
    z = [0, 0, L1 * sin(phi2),
         L1 * sin(phi2) + L2 * sin(phi2+theta1)  ,
         L1 * sin(phi2) + L2 * sin(phi2 +theta1) + L3*sin(psi1+phi2+theta1)]

    # Hide grid lines
    fig = figure(facecolor=(0.0, 0.0, 0.0))
    ax = fig.gca(projection='3d')
    ax.grid(False)
    ax.set_facecolor((0.0, 0.0, 0.0))

    ax.set_xlim(left=-0.5, right=0.5)
    ax.set_ylim(bottom=-0.5, top=0.5)
    ax.set_zlim(bottom=-0.5, top=0.5)
    ax.axis('off')
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.plot(x, y, z, label='shoulder', lw=5, color='white')
    savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')
    close()

    return ax


def create_random_data(nb_posture, nb_command, typ='train'):
    """ func creates NB_POSTURE random joint angles and NB_COMMAND random command
    and plots, saves the corresponding fig and returns corresponding commands and positions


    Args :
        nb_posture : nb of posture to be created
        nb_command : nb of command to be generated

    Returns :
        train_data_x : corresponding initial posture (joint angles) [nb_posture*nb_command postures]
        train_data_y : corresponding final posture (joint angles) [nb_posture*nb_command postures]
        train_data_h : corresponding commands
        train_pos_x :  end effector position corresponding to train_data_x joint state
        train_pos_y : end effector position corresponding to train_data_y joint state
    """

    posture = zeros((nb_posture, 5))
    posture[:, 0] = randrange(nb_posture, -pi/2, pi/2)
    posture[:, 1] = randrange(nb_posture, -pi/2,  pi/2)
    posture[:, 2] = randrange(nb_posture, 0,  pi)
    posture[:, 3] = randrange(nb_posture, -pi, pi)
    posture[:, 4] = randrange(nb_posture, -pi/2,  pi/2)

    command = zeros((nb_command, 5))
    command[:, 0] = randrange(nb_command, -1, 1) * 0.1
    command[:, 1] = randrange(nb_command, -1, 1) * 0.1
    command[:, 2] = randrange(nb_command, -1, 1) * 0.1
    command[:, 3] = randrange(nb_command, -1, 1) * 0.1
    command[:, 4] = randrange(nb_command, -1, 1) * 0.1

    nb_data = nb_posture * nb_command

    train_data_x = zeros((nb_data, 1, 5))
    train_data_y = zeros((nb_data, 1, 5))
    train_data_h = zeros((nb_data, 1, 5))

    train_pos_x = zeros((nb_data, 1, 3))
    train_pos_y = zeros((nb_data, 1, 3))

    idx = 0
    for i in tqdm.tqdm(range(nb_posture), desc="train_data 1"):
        for j in range(nb_command):
            train_data_x[idx] = posture[i]
            train_data_y[idx] = posture[i] + command[j]
            train_data_h[idx] = command[j]

            tmp = control_robot(posture[i])
            ttmp = control_robot(posture[i] + command[j])

            train_pos_x[idx] = tmp
            train_pos_y[idx] = ttmp
            idx = idx + 1

    for i in tqdm.tqdm(range(nb_data), desc='figsave'):
        pos0, pos1, pos2, pos3, pos4 = train_data_x[i][0]
        dpos0, dpos1, dpos2, dpos3, dpos4 = train_data_y[i][0]

        before = typ + '/fig_before_%s' % i
        after = typ + '/fig_after_%s' % i
        plot_arm(train_data_x[i][0], before)
        plot_arm(train_data_y[i][0], after)

    return train_data_x, train_data_y, train_data_h, train_pos_x, train_pos_y


def sort_pictures(train_pos_x, train_pos_y, motion="up"):
    """ sorts the list of positions to extract particular motions such as going left or right
        up or down

        Args:
            train_pos_x : a list of end effector position before command
            train_pos_y : a list of end effecotr position after command

        Returns:
            sorted_pics : a list of tensor images corresponding to the desired motion
    """
    list_idx = sort_command(train_pos_x, train_pos_y, "up")

    for i in tqdm.tqdm(list_idx):
        before = 'images/' + typ + '/fig_before_%s.png' % i
        after = 'images/' + typ + '/fig_after_%s.png' % i

        tens_before = load_and_preprocess_image(before)
        tens_after = load_and_preprocess_image(after)

        noised_tens_before = noised_image(tens_before)
        noised_tens_after = noised_image(tens_after)

        t = tf.concat([noised_tens_before, noised_tens_after], -1)

        tf.reshape(t, [IMG_SIZE, IMG_SIZE, 2])
        tmp.append(t)

    return tf.stack(tmp)


def sort_command(train_pos_x, train_pos_y, motion):
    """ sort a list of positions according to a given motion

    Args :
        train_pos_x : a list of end effector command
        train_pos_y : a list of end effector command
        motion : either "up", "down", "right",  "left", "in" or "out"

    Returns :
        list_idx : a list of indices to use in sort_pictures
    """

    list_idx = []
    n = len(train_pos_x)

    if motion == "up":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][0][2] < train_pos_y[i][0][2]):
                list_idx.append(i)

    if motion == "down":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_y[i][0][2] > train_pos_y[i][0][2]):
                list_idx.append(i)

    if motion == "right":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][0][1] < train_pos_y[i][0][1]):
                list_idx.append(i)

    if motion == "left":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][0][1] > train_pos_y[i][0][1]):
                list_idx.append(i)

    if motion == "in":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][2][0] > train_pos_y[i][2][0]):
                list_idx.append(i)

    if motion == "out":
        for i in tqdm.tqdm(range(n)):
            if (train_pos_x[i][2][0] < train_pos_y[i][2][0]):
                list_idx.append(i)

    return list_idx


def visual_direction(train_pos_x, train_pos_y):
    """ returns a list of visual directions

    Args:
        train_pos_x : end effector before
        train_pos_y : end effector after

    Returns :
        visual_dir : a list of visual direction based on the mvmt given by end effector positions (3d vectors)
    """
    return train_pos_y - train_pos_x
# not used


def gaussian_kernel(size, mean, std):
    """Makes 2D gaussian Kernel for convolution."""

    d = tf.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)

    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def load_and_process_images(nb_data, typ):
    """
    Args :
        nb_data : number of images pairs to process
        typ : 'train' or 'test'
    Returns :
        a tensor of shape (nb_data, image_shape)
    """
    tmp = []

    for i in tqdm.tqdm(range(nb_data)):
        before = 'images/' + typ + '/fig_before_%s.png' % i
        after = 'images/' + typ + '/fig_after_%s.png' % i

        tens_before = load_and_preprocess_image(before)
        tens_after = load_and_preprocess_image(after)

        noised_tens_before = noised_image(tens_before)
        noised_tens_after = noised_image(tens_after)

        t = tf.concat([noised_tens_before, noised_tens_after], -1)

        tf.reshape(t, [IMG_SIZE, IMG_SIZE, 2])
        tmp.append(t)

    return tf.stack(tmp)


def noised_image(tens):
    """ put random gaussian noise on image

    Args:
        tens : an image tensor

    Returns:
        noised_tens : an image tensor with noise
    """
    tens_shape = shape(tens)
    tmp = tf.convert_to_tensor(np.random.random(
        tens_shape), dtype='float32') * 0.1

    return tf.add(tmp, tens)


def preprocess_image(img):
    """decode, resize and normalize a image
    """
    tmp = tf.image.decode_png(img, channels=1)
    tmp = tf.image.resize(tmp, [IMG_SIZE, IMG_SIZE])
    #gauss_kernel = gaussian_kernel(3, 0.0, 1.0)
    #gauss_kernel = gauss_kernel[:,:, tf.newaxis, tf.newaxis]
    #tmp = tf.nn.conv2d(tmp, gauss_kernel, strides=[1, 1, 1, 1], padding="SAME")
    tmp /= 255.0  # normalize to [0,1] range
    return tmp


def load_and_preprocess_image(path):
    image = tf.io.read_file(path)

    return preprocess_image(image)

################################################################################

# CONVOLUTIONAL FILTER ANALYSIS

################################################################################
@tf.function
def compute_conv_loss(model, img, filter_index):
    """computes loss for filter visualization

    Args :
        model : a tf.keras.Model object (preferably a convnet use compute_loss for others)
        img : an input image of shape model.input
        filter_index : a filter index for convnet

    Returns :
        loss : mean of filter_index filter output
    """
    output = model(img)
    # depends on the filter to analyze
    loss = tf.keras.backend.mean(output[:, :, :, filter_index])
    return loss


def generate_conv_pattern(model, filter_index, nb_pass):
    """ generates an input image which maximizes the computed filter loss
        for conv layers. (use generate_pattern for dense layers)

    Args :
        filter_index : the filter to compute
        nb_pass : number of calcul_loss iteration

    Returns :
        tmp : updated input image
    """
    input_img_data = tf.convert_to_tensor(
        np.random.random((1, IMG_SIZE, IMG_SIZE, 1)), dtype='float32') * 2 + 1.
    tmp = tf.compat.v1.get_variable(
        'tmp', dtype=tf.float32, initializer=input_img_data)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(tmp)
        loss = compute_conv_loss(model, tmp, filter_index)

    for i in range(nb_pass):
        grads = tape.gradient(loss, tmp, unconnected_gradients='zero')
        tmp.assign_add(grads)

    return tmp[0][:, :, :]


@tf.function
def compute_loss(model, img, filter_index):
    """compute loss for filter visualization

    Args :
        model : a tf.keras.Model object
        img : an input image of shape model.input
        filter_index : a filter index for convnet

    Returns :
        loss : mean of filter_index filter output
    """
    output = model(img)
    loss = (output[0][:, filter_index])
    return loss


def generate_pattern(model, filter_index, nb_pass):
    """ generate an input image which maximizes the computed filter loss

    Args :Example
        filter_index : the filter to compute
        nb_pass : number of calcul_loss iteration

    Returns :
        tmp : updated input image
    """

    input_img_data = tf.convert_to_tensor(
        np.random.random((1, IMG_SIZE, IMG_SIZE, 2)), dtype='float32') * 2 + 1.
    tmp = tf.compat.v1.get_variable(
        'tmp', dtype=tf.float32, initializer=input_img_data)

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(tmp)
        loss = compute_loss(model, tmp, filter_index)

    for i in range(nb_pass):
        grads = tape.gradient(loss, tmp, unconnected_gradients='zero')
        tmp.assign_add(grads)

    return tmp[0][:, :, :1]


def plot_and_compute_conv_filters(model, size=IMG_SIZE, margin=5, nb_pass=100000):
    """ compute max conv filter input responses.

    Args :
        model : keras model instance

    Returns :
        results : 64 images of conv filter max input response.
    """
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 1))

    for i in tqdm.tqdm(range(8)):
        for j in tqdm.tqdm(range(8)):
            filter_img = generate_conv_pattern(model, i + (j * 8), nb_pass)
            horizontal_start = i * size + i + margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                    vertical_start: vertical_end, :] = filter_img

    return results


def plot_and_compute_last_filters(model, size=IMG_SIZE, margin=5, nb_pass=10000):
    """ compute max dense filter input responses.

    Args :
        model : keras model instance

    Returns :
        results : 32 images of dense filter max input response.
    """
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 1))
    input_image_data = tf.convert_to_tensor(
        np.random.random((1, IMG_SIZE, IMG_SIZE, 2)), dtype='float32') * 2 + 1.
    t = []
    j = 3
    for i in tqdm.tqdm(range(8)):
        for j in tqdm.tqdm(range(4), leave=False):
            filter_img = generate_pattern(model, i + (j * 8), nb_pass)
            horizontal_start = i * size + i + margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end,
                    vertical_start: vertical_end, :] = filter_img
            t.append(filter_img)

    return t, results

def plot_and_save_visual_direction(train_position_before, train_position_after):
    """ plot and save visual direction

    Args :
        train_position_before : a list of end effectors position
        train_position_after : a list of end effectors position

    Returns :
        null
    """
    visual = visual_direction(train_position_before, train_position_after)
    for i in tqdm.tqdm(range(len(train_position_before))):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.grid(False)
        #ax.set_autoscale_on(True)
        #ax.set_facecolor((0.0, 0.0, 0.0))
        #ax.set_xlim(-0., 0.5)
        #ax.set_ylim(-0., 0.5)
        #ax.set_zlim(-0., 0.5)
        ax.axis("off")
        q = ax.quiver(train_position_before[i,0,0],
                  train_position_before[i,0,1],
                  train_position_before[i,0,2],
                  visual[i, 0, 0],
                  visual[i, 0, 1],
                  visual[i, 0, 2],
                  length = 0.1,
                  linewidth=20,
                  cmap='Reds')
        #q.set_array(
        filename = 'images/visual_direction/%s.png' %i
        savefig(filename,  facecolor=fig.get_facecolor(), edgecolor='none')
        close()

################################################################################

# VISUAL ANALYSIS

################################################################################


def compute_latent_filters(model, list, iterator, nb_data):
    """ compute output array for each image pair and categorize it according
    to command direction.

    Args :
        model : keras model instance
        iterator : tf data iterator
        nb_data : total number of images pairs

    Returns :
        t : array of size nb_data containing output of neural networks for each image pairs
        color_position : array of size nb_data with corresponding discretized command direction

    """

    t = []
    color_position = []
    for i in tqdm.tqdm(range(nb_data)):
        tmp = iterator.get_next()
        tmp = tf.expand_dims(tmp, 0)

        j = check_color_position(list, i)
        t.append(model.predict(tmp))
        color_position.append(j)

    return t, color_position


def check_color_position(list, i):
    """ 3d space is divided into 8 subspaces, checks in which subspaces one image pair belongs to.

    Args :
        i : position index in the dataset

    Returns :
        j : command index between 0 and 7
    """
    tmp = list[i]
    x, y, z = tmp[0][0], tmp[0][1], tmp[0][2]

    if (x > 0) and (y > 0) and (z > 0):
        j = 0
    elif (x > 0) and (y > 0) and (z < 0):
        j = 1
    elif (x > 0) and (y < 0) and (z > 0):
        j = 2
    elif (x > 0) and (y < 0) and (z < 0):
        j = 3
    elif (x < 0) and (y > 0) and (z > 0):
        j = 4
    elif (x < 0) and (y > 0) and (z < 0):
        j = 5
    elif (x < 0) and (y < 0) and (z > 0):
        j = 6
    elif (x < 0) and (y < 0) and (z < 0):
        j = 7
    else:
        j = 7

    return j

################################################################################

# NEURAL NETWORKS

################################################################################


def build_dense_encoder(custom_shape=INPUT_ENCODER_SHAPE):

    inputs = tf.keras.Input(shape=custom_shape, name='encoder_input')

    x = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(inputs)
    x = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

    y = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1])(inputs)
    y = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    fx = tf.keras.layers.Flatten()(x)
    fx = tf.keras.layers.Dense(
        LATENT_DIM, activation='relu', name='latent_enc_fx1')(fx)
    #fx = tf.keras.layers.Dense(LATENT_DIM, activation = 'relu', name = 'latent_enc_fx2')(fx)
    #fx = tf.keras.layers.Dense(LATENT_DIM, activation = 'relu', name = 'latent_enc_fx3')(fx)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fy = tf.keras.layers.Flatten()(y)
    fy = tf.keras.layers.Dense(
        LATENT_DIM, activation='relu', name='latent_enc_fy1')(fy)
    #fy = tf.keras.Dense(LATENT_DIM, activation = 'relu', name = 'latent_enc_fy2')(fy)
    #fy = tf.keras.Dense(LATENT_DIM, activation = 'relu', name = 'latent_enc_fy3')(fy)
    fy = tf.keras.layers.Reshape((1, LATENT_DIM,))(fy)

    matmul = tf.keras.layers.Multiply()([fx, fy])

    fh = tf.keras.layers.Flatten()(matmul)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_fh1')(fh)
    #fh = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_fh2')(fh)
    #fh = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_fh3')(fh)

    fx = tf.keras.layers.Reshape((1, LATENT_DIM,))(fx)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    outputs = tf.keras.layers.Concatenate()([fx, fh])
    encoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='encoder_model')

    return encoder


def build_dense_decoder():

    inputs = tf.keras.Input(shape=INPUT_DECODER_SHAPE, name='decoder_input')

    fx = tf.keras.layers.Lambda(lambda x: x[:, :, :LATENT_DIM])(inputs)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fh = tf.keras.layers.Lambda(lambda x: x[:, :, LATENT_DIM:])(inputs)
    fh = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fh)

    fh = tf.keras.layers.Flatten()(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh2')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh3')(fh)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    matmul = tf.keras.layers.Multiply()([fx, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fy1')(fy)
    #fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_dec_fy2')(fy)
    #fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_dec_fy3')(fy)

    y = tf.keras.layers.Dense(
        IMG_SIZE * IMG_SIZE, activation='relu', name='y_recon')(fy)
    outputs = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    decoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='decoder_model')

    return decoder


def build_conv2D_encoder(custom_shape=INPUT_ENCODER_SHAPE):

    inputs = tf.keras.Input(shape=custom_shape, name='encoder_input')

    x = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(inputs)
    x = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

    y = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1])(inputs)
    y = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    #fx = tf.keras.layers.Flatten()(x)
    fx = tf.keras.layers.Conv2D(filters=LATENT_DIM, kernel_size=7, strides=(
        2, 2), activation='relu', name='conv_x_1')(x)
    fx = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(fx)
    fx = tf.keras.layers.Conv2D(filters=LATENT_DIM * 2, kernel_size=3,
                                strides=(1, 1),activation='relu', name='conv_x_2')(fx)
    fx = tf.keras.layers.Flatten()(fx)
    #fx = tf.keras.layers.Dense(units=15*15*64, name = 'latent_fx1')(fx)
    fx = tf.keras.layers.Dense(LATENT_DIM, name='latent_fx2')(fx)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    #fy = tf.keras.layers.Flatten()(y)
    fy = tf.keras.layers.Conv2D(filters=LATENT_DIM, kernel_size=7, strides=(
        2, 2), activation='relu', name='conv_y_1')(y)
    fy = tf.keras.layers.Conv2D(filters=LATENT_DIM * 2, kernel_size=3,
                                strides=(2, 2), activation='relu', name='conv_y_2')(fy)
    fy = tf.keras.layers.Flatten()(fy)
    #fy = tf.keras.layers.Dense(units=15*15*64, name = 'latent_fy1')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name='latent_fy2')(fy)
    fy = tf.keras.layers.Reshape((1, LATENT_DIM,))(fy)

    matmul = tf.keras.layers.Multiply()([fx, fy])

    fh = tf.keras.layers.Flatten()(matmul)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_fh1')(fh)
    #fh = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_fh2')(fh)
    #fh = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_fh3')(fh)

    fx = tf.keras.layers.Reshape((1, LATENT_DIM,))(fx)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    outputs = tf.keras.layers.Concatenate()([fx, fh])
    encoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='encoder_model')

    return encoder


def build_conv2D_decoder():

    inputs = tf.keras.Input(shape=INPUT_DECODER_SHAPE, name='decoder_input')

    fx = tf.keras.layers.Lambda(lambda x: x[:, :, :LATENT_DIM])(inputs)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fh = tf.keras.layers.Lambda(lambda x: x[:, :, LATENT_DIM:])(inputs)

    fh = tf.keras.layers.Flatten()(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh2')(fh)
    #fh = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_dec_fh3')(fh)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    matmul = tf.keras.layers.Multiply()([fx, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(
        LATENT_DIM / 4 * LATENT_DIM / 4 * LATENT_DIM * 2, name='latent_dec_fy1')(fy)
    fy = tf.keras.layers.Reshape(
        (LATENT_DIM / 4, LATENT_DIM / 4, 2 * LATENT_DIM))(fy)
    fy = tf.keras.layers.Conv2DTranspose(
        filters=LATENT_DIM, activation='relu', kernel_size=3, strides=(2, 2))(fy)
    fy = tf.keras.layers.Conv2DTranspose(
        filters=LATENT_DIM / 2, name='conv_trans_y_2', activation='relu', kernel_size=3, strides=(2, 2))(fy)
    fy = tf.keras.layers.Conv2DTranspose(
        filters=LATENT_DIM / 4, name='conv_trans_y_3', activation='relu', kernel_size=3, strides=(2, 2))(fy)
    fy = tf.keras.layers.Conv2DTranspose(
        1, name='conv_trans_y_4', activation='sigmoid', kernel_size=3, strides=(1, 1))(fy)
    fy = tf.keras.layers.Cropping2D(cropping=((5, 4), (4, 5)))(fy)
    #y = tf.keras.layers.Dense(IMG_SIZE*IMG_SIZE, activation = 'relu', name = 'y_recon')(fy)
    outputs = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(fy)

    decoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='decoder_model')

    return decoder


def build_conv2D_pointwise_encoder(custom_shape= INPUT_ENCODER_SHAPE):

    inputs = tf.keras.Input(shape=custom_shape, name='encoder_input')

    x = tf.keras.layers.Lambda(lambda x: x[:, :, :, 0])(inputs)
    x = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(x)

    y = tf.keras.layers.Lambda(lambda x: x[:, :, :, 1])(inputs)
    y = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    #fx = tf.keras.layers.Flatten()(x)
    fx = tf.keras.layers.Conv2D(filters=LATENT_DIM, kernel_size=7, strides=(
        2, 2), activation='relu', name='conv_x_1')(x)
    fx = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(fx)
    fx = tf.keras.layers.Conv2D(filters=LATENT_DIM * 2, kernel_size=3,
                                strides=(1, 1),activation='relu', name='conv_x_2')(fx)
    fx = tf.keras.layers.Flatten()(fx)
    #fx = tf.keras.layers.Dense(units=15*15*64, name = 'latent_fx1')(fx)
    fx = tf.keras.layers.Dense(LATENT_DIM, name='latent_fx2')(fx)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    #fy = tf.keras.layers.Flatten()(y)
    fy = tf.keras.layers.Conv2D(filters=LATENT_DIM, kernel_size=7, strides=(
        2, 2), activation='relu', name='conv_y_1')(y)
    fy = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(fy)
    fy = tf.keras.layers.Conv2D(filters=LATENT_DIM * 2, kernel_size=3,
                                strides=(1,1), activation='relu', name='conv_y_2')(fy)
    fy = tf.keras.layers.MaxPool2D(pool_size = (3,3), strides = 2)(fy)
    fy = tf.keras.layers.Flatten()(fy)
    #fy = tf.keras.layers.Dense(units=15*15*64, name = 'latent_fy1')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name='latent_fy2')(fy)
    fy = tf.keras.layers.Reshape((LATENT_DIM,1,))(fy)

    matmul = tf.keras.layers.Multiply()([fx, fy])

    fh = tf.keras.layers.Flatten()(matmul)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_fh1')(fh)
    #fh = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_fh2')(fh)
    #fh = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_fh3')(fh)

    fx = tf.keras.layers.Reshape((1, LATENT_DIM,))(fx)
    fh = tf.keras.layers.Reshape((1, LATENT_DIM,))(fh)

    outputs = tf.keras.layers.Concatenate()([fx, fh])
    encoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='encoder_model')

    return encoder


def build_dense_pointwise_decoder():

    inputs = tf.keras.Input(shape=INPUT_DECODER_SHAPE, name='decoder_input')

    fx = tf.keras.layers.Lambda(lambda x: x[:, :, :LATENT_DIM])(inputs)
    fx = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fx)

    fh = tf.keras.layers.Lambda(lambda x: x[:, :, LATENT_DIM:])(inputs)
    fh = tf.keras.layers.Reshape((LATENT_DIM, 1,))(fh)

    fh = tf.keras.layers.Flatten()(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh2')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fh3')(fh)
    fh = tf.keras.layers.Reshape((LATENT_DIM,1,))(fh)

    matmul = tf.keras.layers.Multiply()([fx, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(LATENT_DIM, name='latent_dec_fy1')(fy)
    #fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_dec_fy2')(fy)
    #fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_dec_fy3')(fy)

    y = tf.keras.layers.Dense(
        IMG_SIZE * IMG_SIZE, activation='relu', name='y_recon')(fy)
    outputs = tf.keras.layers.Reshape((IMG_SIZE, IMG_SIZE, 1))(y)

    decoder = tf.keras.Model(
        inputs=inputs, outputs=outputs, name='decoder_model')

    return decoder

################################################################################

# CONTROL MODEL

################################################################################

def image_to_convnet(custom_shape = (IMG_SIZE, IMG_SIZE,1)):
    inputs = tf.keras.Input(shape = custom_shape, name = 'conv_input')
    x = tf.keras.layers.Conv2D(filters = LATENT_DIM,
                              kernel_size = 3,
                              strides = (2,2),
                              activation = 'relu',
                              name = 'conv_1')(inputs)
    x = tf.keras.layers.Conv2D(filters = 64,
                              kernel_size = 3,
                              strides = (2,2),
                              activation = 'relu',
                              name = 'conv_2')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_layer_1')(x)
    outputs = tf.keras.layers.Reshape((LATENT_DIM,1))(x)

    convnet = tf.keras.Model(inputs = inputs,
                            outputs = outputs,
                            name = 'conv_net_1')
    return convnet

def pos_to_dense(custom_shape = (1,3)):
    inputs = tf.keras.layers.Input(shape = custom_shape, name = 'dense_input')
    x = tf.keras.layers.Reshape(custom_shape)(inputs)
    x = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_1')(x)
    x = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_2')(x)
    outputs = tf.keras.layers.Reshape((LATENT_DIM,1))(x)

    densenet = tf.keras.Model(inputs = inputs, outputs = outputs, name = "dense_net")

    return densenet

def build_control_model():
    """ visual_direction and posture before as input
        motor command as output
    """

    inputs = tf.keras.layers.Input(shape=(2,5))

    h = tf.keras.layers.Lambda(lambda x: x[:,0,:3])(inputs)
    p = tf.keras.layers.Lambda(lambda x: x[:,1,:])(inputs)

    h = tf.keras.layers.Reshape((1,3))(h)
    p = tf.keras.layers.Reshape((1,5))(p)

    fh = tf.keras.layers.Flatten()(h)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_2')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_3')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_4')(fh)
    fh = tf.keras.layers.Reshape((LATENT_DIM, 1))(fh)

    fp = tf.keras.layers.Flatten()(p)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_1')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_2')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_3')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_4')(fp)
    fp = tf.keras.layers.Reshape((LATENT_DIM,1))(fp)

    matmul = tf.keras.layers.Multiply()([fp, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_1')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_2')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_3')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_4')(fy)
    fy = tf.keras.layers.Dense(5, name = 'latent_y_out')(fy)
    fy = tf.keras.layers.Reshape((1,5))(fy)

    outputs = fy

    model = tf.keras.Model(inputs = inputs, outputs = outputs, name='control_model')

    return model

def prepare_dataset(train_command, train_posture_before, train_posture_after, train_position_after, train_position_before):

    t_before = map(lambda x : x[0,:], train_posture_before)
    t_before = np.expand_dims(t_before, 1)

    t_command = map(lambda x : x[0,:], train_command)
    t_command = np.expand_dims(t_command, 1)

    t_visual_direction = normalize_vect(train_position_after - train_position_before)
    t_visual_direction = padding(t_visual_direction, 2)

    tmp_input = np.concatenate([t_visual_direction, t_before], axis = 1)

    train_control_dataset = (
        tf.data.Dataset.from_tensor_slices((tmp_input, t_command))
        .repeat(10)
        .shuffle(NB_DATA)
        .batch(BATCH_SIZE)
        )

    return train_control_dataset

def padding(t_visual, n):
    (a,b,c) = shape(t_visual)
    res = np.zeros((a,b,c+n))

    for i in range(len(t_visual)):

        res[i] = np.pad(t_visual[i][0], (0,n), 'constant')
    return res


def normalize_vect(visual_direction):
    res = np.zeros(shape(visual_direction))
    for i in range(len(visual_direction)):
        tmp = visual_direction[i]
        norm = np.linalg.norm(visual_direction[i])
        res[i]= tmp/norm
    return res


def test_visuomotor_control(control_model, current_posture,  visual_direction):
    # liste des postures successives
    postures = []
    postures.append(current_posture)
    posture = current_posture

    for i in range(400):

        vd = np.array([np.pad(visual_direction, (0,2), 'constant')])

        inputs = np.concatenate([vd, posture], axis = 0)
        inputs = np.expand_dims(inputs, 0)

        command = control_model.predict(inputs)

        posture = posture + command[0]
        posture = check_valid_posture(posture)
        postures.append(posture)

    return postures


def check_valid_posture(posture):
    valid_posture = np.zeros(shape(posture))

    if (np.abs(posture[0][0])>pi/2):
        valid_posture[0][0] = np.sign(posture[0][0])*pi/2
    else :
        valid_posture[0][0] = posture[0][0]

    if (np.abs(posture[0][1])>pi/2):
        valid_posture[0][1] = np.sign(posture[0][1])*pi/2
    else :
        valid_posture[0][1] = posture[0][1]

    if (np.abs(posture[0][2])>pi/2):
        valid_posture[0][2] = np.sign(posture[0][2])*pi/2
    else :
        valid_posture[0][2] = posture[0][2]

    if (np.abs(posture[0][3])>pi):
        valid_posture[0][3] = np.sign(posture[0][3])*pi
    else :
        valid_posture[0][3] = posture[0][3]

    if (np.abs(posture[0][4])>pi/2):
        valid_posture[0][4] = np.sign(posture[0][4])*pi/2
    else :
        valid_posture[0][4] = posture[0][4]

    return valid_posture

def plot_arm_from_posture(posture, target):
    """function ploting and saving in images/ folder 3d arm plots
    from arg joint state input.

    Args :
        phi1, phi2, theta1 : joint angles (3 dof arm )

    Returns :
        ax : a matplotlib figure object
    """
    phi1, phi2, theta1, psi1, psi2 = angles
    filename = 'images/%s.png' % time

    x = [0, 0, L1 * cos(phi1) * cos(phi2),
         (L1*cos(phi1)+L2*cos(theta1+phi1))*cos(phi2),
         (L1*cos(phi2)+L2*cos(theta1+phi2))*cos(phi1)+ L3*cos(psi1)*cos(psi2)]
    y = [0, 0, L1 * sin(phi1) * cos(phi2),
         (L1*cos(phi1)+L2*cos(theta1+phi1))*sin(phi2),
         (L1*cos(phi2)+L2*cos(theta1+phi2))*sin(phi1) + L3*cos(psi2)*sin(psi1)]
    z = [0, 0, L1 * sin(phi2),
         L1 * sin(phi1) + L2 * sin(theta1 + phi1)  ,
         L1 * sin(phi1) + L2 * sin(theta1 + phi1) + L3*sin(psi1)]

    # Hide grid lines
    fig = figure(facecolor=(0.0, 0.0, 0.0))
    ax = fig.gca(projection='3d')
    ax.grid(False)
    ax.set_facecolor((0.0, 0.0, 0.0))

    ax.set_xlim(left=-0.3, right=0.3)
    ax.set_ylim(bottom=-0.3, top=0.3)
    ax.set_zlim(bottom=-0.3, top=0.3)
    ax.axis('off')
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.plot(x, y, z, label='shoulder', lw=5, color='white')
    #savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')
    close()

    return ax

def plot_elbow_from_posture(posture):
    """function ploting and saving in /images 3d arm plot
    from arg joint state

    Args :
        phi1, phi2, theta1 : joint angles (3 dof arm )

    Returns :
        ax : a matplotlib figure object
    """
    phi1 = posture[0][0]
    phi2 = posture[0][1]
    theta1 = posture[0][2]

    fig = figure(facecolor=(0.0, 0.0, 0.0))
    ax = fig.gca(projection='3d')
    x = [0, 0, L1 * cos(phi1) * cos(theta1)]
    y = [0, 0, L1 * cos(phi1) * sin(theta1)]  # ELBOW + HAND
    z = [0, 0, L1 * sin(phi1)]  # ELBOW + HAND
    # ax.plot(x[0:1], y[0:1], z[0:1], label='shoulder', lw=2, color= 'k')
    # ax.plot(x[2:3], y[2:3], z[2:3], label='elbow', lw=2, color= 'c')
    # Hide grid lines
    ax.grid(False)
    # ax.set_autoscale_on(True)
    ax.set_facecolor((0.0, 0.0, 0.0))

    ax.set_xlim(left=-0.2, right=0.2)
    ax.set_ylim(bottom=-0.2, top=0.2)
    ax.set_zlim(bottom=-0.2, top=0.2)
    ax.axis('off')
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.plot(x, y, z, label='shoulder', lw=5, color='white')
    ax.scatter(-0.5,-0.5,0.1, 'r')
    #filename = 'images/%s.png' % time
    #savefig(filename, facecolor=fig.get_facecolor(), edgecolor='none')


def go_to_position(control_model, current_posture, target_position, nb_pass = 500):
    postures = []
    vd = []


    visual_direction = compute_vd_from_position(target_position, current_posture)

    postures.append(current_posture)
    vd.append((visual_direction))
    j = 0
    while (j < nb_pass) and (np.linalg.norm(target_position - np.array(control_robot(current_posture[0]))) > 0.1):


        inputs = np.expand_dims(np.concatenate([visual_direction, current_posture], axis=0), 0)

        new_command = control_model.predict(inputs)

        #new_command = command_bornee(new_command)
        current_posture = current_posture + new_command[0]
        current_posture = check_valid_posture(current_posture)

        visual_direction = compute_vd_from_position(target_position, current_posture)

        postures.append(current_posture)
        vd.append((visual_direction))
        visual_direction = compute_vd_from_position(target_position, current_posture)/np.linalg.norm(visual_direction)
        j +=1

    return postures, np.array(vd)

def compute_vd_from_position(target_position, current_posture):

    current_position = control_robot(current_posture[0])
    #current_position  = np.expand_dims(current_position, 0)
    tmp = (np.array(target_position)-np.array(current_position))
    return np.expand_dims(np.pad(tmp[0], (0,2), 'constant'), 0)


def is_distance_end_effector_to_target_ok(visual_direction):
    dx, dy, dz = visual_direction[0]

    dist = np.sqrt(dx*dx+dy*dy+dz*dz)

    return (dist > 0.01)

def command_bornee(command):
    new_command = np.zeros(shape(command))
    for i in range(3):
        if command[0][0][i] > 2:
            new_command[0][0][i] = 2
        elif command[0][0][i] < -2:
            new_command[0][0][i] = -2
        else :
            new_command[0][0][i] = command[0][0][i]
        return new_command

def calcul_angular_error(position, direction_visuelle):
    return np.arccos(np.dot(position, direction_visuelle)/(np.linalg.norm(position)*np.linalg.norm(direction_visuelle)))
def calcul_position_error(position, target):

    return np.linalg.norm(position-target)

def get_end_effector_orientation(angles):
    """function calculating end effector postion from joint angles

    Args :
        phi1, phi2, theta1, theta2 : joint angles (4 dof arm )

    Returns:
        x,y,z : end effector position
        vx,vy,vz : speed i guess...
    """
    phi1, phi2, theta1,psi1, psi2 = angles


    return np.array([psi1, psi2])


def prepare_dataset_with_orientation(train_command, train_posture_before, train_posture_after, train_position_after, train_position_before):

    t_before = map(lambda x : x[0,:], train_posture_before)
    t_before = np.expand_dims(t_before, 1)


    t_after = map(lambda x : x[0,:], train_posture_after)
    t_after = np.expand_dims(t_after, 1)

    t_command = map(lambda x : x[0,:], train_command)
    t_command = np.expand_dims(t_command, 1)


    orientation_before = np.array(map(lambda x : get_end_effector_orientation(x), t_before[:,0,:]))
    orientation_after = np.array(map(lambda x : get_end_effector_orientation(x), t_after[:,0,:]))

    t_orientation = orientation_after - orientation_before
    t_orientation = np.expand_dims(orientation, 1)


    t_visual_direction = normalize_vect(train_position_after - train_position_before)
    #t_visual_direction = padding(t_visual_direction, 2)

    info = np.concatenate((t_visual_direction, t_orientation), axis=2)

    tmp_input = np.concatenate([info, t_before], axis = 1)

    train_control_dataset = (
        tf.data.Dataset.from_tensor_slices((tmp_input, t_command))
        .repeat(10)
        .shuffle(NB_DATA)
        .batch(100)
        )

    return train_control_dataset

def build_control_orientation_model():
    """ visual_direction and posture before as input
        motor command as output
    """
    LATENT_DIM = 32
    inputs = tf.keras.layers.Input(shape=(2,5))

    h = tf.keras.layers.Lambda(lambda x: x[:,0,:])(inputs)
    p = tf.keras.layers.Lambda(lambda x: x[:,1,:])(inputs)

    h = tf.keras.layers.Reshape((1,5))(h)
    p = tf.keras.layers.Reshape((1,5))(p)

    fh = tf.keras.layers.Flatten()(h)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_1')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_2')(fh)
    fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_3')(fh)
    #fh = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_h_4')(fh)
    fh = tf.keras.layers.Reshape((LATENT_DIM, 1))(fh)

    fp = tf.keras.layers.Flatten()(p)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_1')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_2')(fp)
    fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_3')(fp)
    #fp = tf.keras.layers.Dense(LATENT_DIM, name = 'dense_p_4')(fp)
    fp = tf.keras.layers.Reshape((LATENT_DIM,1))(fp)

    matmul = tf.keras.layers.Multiply()([fp, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_1')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_2')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_3')(fy)
    fy = tf.keras.layers.Dense(LATENT_DIM, name = 'latent_y_4')(fy)
    fy = tf.keras.layers.Dense(5, name = 'latent_y_out')(fy)
    fy = tf.keras.layers.Reshape((1,5))(fy)

    outputs = fy

    model = tf.keras.Model(inputs = inputs, outputs = outputs, name='control_model')

    return model
