#!/usr/bin/env python

from numpy import *
from matplotlib.pylab import *
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
import time
import scipy
import math
import cv2
import cPickle as pickle
import random
import numpy as np
import matplotlib.pyplot as plt
import keras

import time
import random
import scipy
import math
import matplotlib.animation as animation
import tensorflow as tf
from keras.models import Model
from keras.layers import Activation, Dense, Input, Multiply
from keras.layers import Conv2D, Flatten, Reshape, Conv2DTranspose
from keras.layers import Dot, Lambda, Concatenate, RepeatVector
from keras.utils import plot_model
from PIL import Image
from keras.constraints import max_norm, non_neg

matplotlib.rcParams.update({'font.size': 16})
L1=0.3
L2=0.3
img_size = 128
input_encoder_shape = (img_size, img_size,2)
latent_dim = 32
input_decoder_shape = (1, 2*latent_dim,)

def randrange(n, vmin, vmax):
    return (vmax-vmin)*rand(n) + vmin

def generate_posture(state=randrange(4, 0, 2*pi),L1=0.3,L2=0.3):
	state[3] = 0 # NEW 170111
	x = cumsum([0,L1 * cos(state[0]) * cos(state[2]), L2 * cos(state[0]+state[1]) * cos(state[2]+state[3])])
	y = cumsum([0,L1 * cos(state[0]) * sin(state[2]), L2 * cos(state[0]+state[1]) * sin(state[2]+state[3])])
	z = cumsum([0,L1 * sin(state[0]), L2 * sin(state[0]+state[1])])
	return x,y,z

def generate_random_robot_data(n,L1=0.3,L2=0.3 ):
	phi1 = randrange(n, 0, 2*pi)
	theta1 = randrange(n, 0, 2*pi)
	phi2 = randrange(n, 0, 2*pi)
	theta2 =  zeros(n)
	x = L1 * cos(phi1) * cos(theta1) + L2 * cos(phi1 + phi2) * cos(theta1 + theta2) # ELBOW + HAND
	y = L1 * cos(phi1) * sin(theta1) + L2 * cos(phi1 + phi2) * sin(theta1 + theta2) # ELBOW + HAND
	z = L1 * sin(phi1)               + L2 * sin(phi1 + phi2) # ELBOW + HAND
	vx=L2 * cos(phi1 + phi2) * cos(theta1 + theta2)
	vy=L2 * cos(phi1 + phi2) * sin(theta1 + theta2)
	vz=L2 * sin(phi1 + phi2)
	return (phi1,phi2,theta1,x,y,z,vx,vy,vz)


def control_robot(phi1,phi2,theta1, theta2):
	x = L1 * cos(phi1) * cos(theta1) + L2 * cos(phi1 + phi2) * cos(theta1 + theta2) # ELBOW + HAND
	y = L1 * cos(phi1) * sin(theta1) + L2 * cos(phi1 + phi2) * sin(theta1 + theta2) # ELBOW + HAND
	z = L1 * sin(phi1)               + L2 * sin(phi1 + phi2) # ELBOW + HAND
	vx=L2 * cos(phi1 + phi2) * cos(theta1 + theta2)
	vy=L2 * cos(phi1 + phi2) * sin(theta1 + theta2)
	vz=L2 * sin(phi1 + phi2)
	return (x,y,z,vx,vy,vz)

def plot_arm(phi1, phi2, theta1, time):
    fig = figure()
    ax = fig.gca(projection='3d')
    x = [0, 0, L1 * cos(phi1) * cos(theta1), L2 * cos(phi1 + phi2) * cos(theta1)]
    y = [0, 0, L1 * cos(phi1) * sin(theta1), L2 * cos(phi1 + phi2) * sin(theta1)] # ELBOW + HAND
    z = [0, 0, L1 * sin(phi1) , L2 * sin(phi1 + phi2)] # ELBOW + HAND
    # ax.plot(x[0:1], y[0:1], z[0:1], label='shoulder', lw=2, color= 'k')
    # ax.plot(x[2:3], y[2:3], z[2:3], label='elbow', lw=2, color= 'c')
    # Hide grid lines
    ax.grid(False)
    # ax.set_autoscale_on(True)

    ax.set_xlim(left=-0.2, right=0.2)
    ax.set_ylim(bottom=-0.2, top=0.2)
    ax.set_zlim(bottom=-0.2, top=0.2)
    ax.axis('off')
    # Hide axes ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.plot(x, y, z, label='shoulder', lw=2, color= 'k')
    filename = 'images/%s.png' %time
    savefig(filename)
    close()

    return ax

def create_random_data(nb_posture, nb_command, typ = 'train'):
    posture = zeros((nb_posture, 4))
    posture[:,0] = randrange(nb_posture, 0, 2*pi)
    posture[:,1] = randrange(nb_posture, 0, 2*pi)
    posture[:,2] = randrange(nb_posture, 0, 2*pi)

    command = zeros((nb_command, 4))
    command[:,0] = randrange(nb_command, 0, 1)*0.5
    command[:,1] = randrange(nb_command, 0, 1)*0.5
    command[:,2] = randrange(nb_command, 0, 1)*0.5

    nb_data = nb_posture*nb_command

    train_data_x = zeros((nb_data, 1, 4))
    train_data_y = zeros((nb_data, 1, 4))
    train_data_h = zeros((nb_data, 1, 4))

    idx = 0
    for i in range(nb_posture):
        for j in range(nb_command):
            train_data_x[idx] = posture[i]
            idx = idx + 1

    idx = 0
    for i in range(nb_posture):
        for j in range(nb_command):
            train_data_y[idx] = posture[i]  + command[j]
            idx = idx + 1

    idx = 0
    for i in range(nb_posture):
        for j in range(nb_command):
            train_data_h[idx] = command[j]
            idx = idx + 1

    for i in range(nb_data):
        pos0, pos1, pos2, pos3 = train_data_x[i][0]
        dpos0, dpos1, dpos2, dpos3 = train_data_y[i][0]

        before = typ + '/fig_before_%s' %i
        after = typ + '/fig_after_%s' %i
        plot_arm(pos0, pos1, pos2, before)
        plot_arm(dpos0, dpos1, dpos2, after)

    return train_data_h

def load_and_process_images(nb_data, typ):
    tmp = []

    for i in range(nb_data):
        before = 'images/' + typ  + '/fig_before_%s.png' %i
        after = 'images/' + typ + '/fig_after_%s.png' %i

        tens_before = load_and_preprocess_image(before)
        tens_after = load_and_preprocess_image(after)

        t = tf.concat([tens_before, tens_after], -1)
        tf.reshape(t, [128,128,2])
        tmp.append(t)

    return tf.stack(tmp)

def preprocess_image(img):
    tmp = tf.image.decode_png(img, channels=1)
    tmp = tf.image.resize(tmp, [128, 128])
    tmp /= 255.0  # normalize to [0,1] range
    return tmp

def load_and_preprocess_image(path):
    image = tf.io.read_file(path)
    return preprocess_image(image)

def build_dense_encoder():

    inputs = tf.keras.Input(shape = input_encoder_shape, name = 'encoder_input')

    x = tf.keras.layers.Lambda(lambda x: x[:,:,:,0])(inputs)
    x = tf.keras.layers.Reshape((img_size, img_size,1))(x)

    y = tf.keras.layers.Lambda(lambda x: x[:,:,:,1])(inputs)
    y = tf.keras.layers.Reshape((img_size, img_size,1))(y)

    fx = tf.keras.layers.Flatten()(x)
    fx = tf.keras.layers.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fx1')(fx)
    #fx = tf.keras.layers.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fx2')(fx)
    #fx = tf.keras.layers.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fx3')(fx)
    fx = tf.keras.layers.Reshape((latent_dim,1,))(fx)

    fy = tf.keras.layers.Flatten()(y)
    fy = tf.keras.layers.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fy1')(fy)
    #fy = tf.keras.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fy2')(fy)
    #fy = tf.keras.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fy3')(fy)
    fy = tf.keras.layers.Reshape((1,latent_dim,))(fy)

    matmul = tf.keras.layers.Multiply()([fx, fy])

    fh = tf.keras.layers.Flatten()(matmul)
    fh = tf.keras.layers.Dense(latent_dim, name = 'latent_fh1')(fh)
    #fh = tf.keras.layers.Dense(latent_dim, name = 'latent_fh2')(fh)
    #fh = tf.keras.layers.Dense(latent_dim, name = 'latent_fh3')(fh)

    fx = tf.keras.layers.Reshape((1, latent_dim,))(fx)
    fh = tf.keras.layers.Reshape((1,latent_dim,))(fh)

    outputs = tf.keras.layers.Concatenate()([fx, fh])
    encoder = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'encoder_model')

    return encoder

def build_dense_decoder():

    inputs = tf.keras.Input(shape = input_decoder_shape, name = 'decoder_input')

    fx = tf.keras.layers.Lambda(lambda x: x[:,:,:latent_dim])(inputs)
    fx = tf.keras.layers.Reshape((latent_dim,1,))(fx)

    fh = tf.keras.layers.Lambda(lambda x: x[:,:,latent_dim:])(inputs)

    fh = tf.keras.layers.Flatten()(fh)
    fh = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fh1')(fh)
    fh = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fh2')(fh)
    fh = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fh3')(fh)
    fh = tf.keras.layers.Reshape((1,latent_dim,))(fh)

    matmul = tf.keras.layers.Multiply()([fx, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fy1')(fy)
    #fy = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fy2')(fy)
    #fy = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fy3')(fy)

    y = tf.keras.layers.Dense(img_size*img_size, activation = 'relu', name = 'y_recon')(fy)
    outputs = tf.keras.layers.Reshape((img_size, img_size,1))(y)

    decoder = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'decoder_model')

    return decoder


def build_conv2D_encoder():

    inputs = tf.keras.Input(shape = input_encoder_shape, name = 'encoder_input')

    x = tf.keras.layers.Lambda(lambda x: x[:,0,:,:])(inputs)
    x = tf.keras.layers.Reshape((img_size, img_size,1))(x)

    y = tf.keras.layers.Lambda(lambda x: x[:,1,:,:])(inputs)
    y = tf.keras.layers.Reshape((img_size, img_size,1))(y)

    fx = tf.keras.layers.Flatten()(x)
    fx = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu')(fx)
    #fx = tf.keras.layers.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fx2')(fx)
    #fx = tf.keras.layers.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fx3')(fx)
    fx = tf.keras.layers.Reshape((latent_dim,1,))(fx)

    fy = tf.keras.layers.Flatten()(y)
    fy = tf.keras.layers.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fy1')(fy)
    #fy = tf.keras.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fy2')(fy)
    #fy = tf.keras.Dense(latent_dim, activation = 'relu', name = 'latent_enc_fy3')(fy)
    fy = tf.keras.layers.Reshape((1,latent_dim,))(fy)

    matmul = tf.keras.layers.Multiply()([fx, fy])

    fh = tf.keras.layers.Flatten()(matmul)
    fh = tf.keras.layers.Dense(latent_dim, name = 'latent_fh1')(fh)
    #fh = tf.keras.layers.Dense(latent_dim, name = 'latent_fh2')(fh)
    #fh = tf.keras.layers.Dense(latent_dim, name = 'latent_fh3')(fh)

    fx = tf.keras.layers.Reshape((1, latent_dim,))(fx)
    fh = tf.keras.layers.Reshape((1,latent_dim,))(fh)

    outputs = tf.keras.layers.Concatenate()([fx, fh])
    encoder = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'encoder_model')

    return encoder

def build_conv2D_decoder():

    inputs = tf.keras.Input(shape = input_decoder_shape, name = 'decoder_input')

    fx = tf.keras.layers.Lambda(lambda x: x[:,:,:latent_dim])(inputs)
    fx = tf.keras.layers.Reshape((latent_dim,1,))(fx)

    fh = tf.keras.layers.Lambda(lambda x: x[:,:,latent_dim:])(inputs)

    fh = tf.keras.layers.Flatten()(fh)
    fh = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fh1')(fh)
    fh = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fh2')(fh)
    #fh = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fh3')(fh)
    fh = tf.keras.layers.Reshape((1,latent_dim,))(fh)

    matmul = tf.keras.layers.Multiply()([fx, fh])

    fy = tf.keras.layers.Flatten()(matmul)
    fy = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fy1')(fy)
    #fy = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fy2')(fy)
    #fy = tf.keras.layers.Dense(latent_dim, name = 'latent_dec_fy3')(fy)

    y = tf.keras.layers.Dense(img_size*img_size, activation = 'relu', name = 'y_recon')(fy)
    outputs = tf.keras.layers.Reshape((img_size, img_size,1))(y)

    decoder = tf.keras.Model(inputs = inputs, outputs = outputs, name = 'decoder_model')

    return decoder
