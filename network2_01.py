 # -*-coding:Latin-1 -*
# August 2017
# VISUAL TRANSFORMATION FOR BODY IMAGE RECONSTRUCTION
# IM1 x IM2 -> MOTOR SYNERGY -> IM1' x IM2' 
# XP 1 à une position fixe de la main faire bouger les objets autour de la main (en  étoile)
# XP 2 à une position fixe de l'objet faire bouger la main dans la direction de l'objet (en ligne droite?)

from numpy import *
from matplotlib.pylab import *

import time
import random
import scipy
import math
from mpl_toolkits.mplot3d import Axes3D # 3D

import cPickle as pickle
import cv2
from drawnow import *

matplotlib.rcParams.update({'font.size': 16})

to_backup=True
timeframe=time.strftime('%Y%m%d%H%M%S')

def save(numfig):
	timeframe=time.strftime('%Y%m%d%H%M%S')
	if to_backup == True:
	    filename = 'Fig_%d_%s.png' %(numfig,timeframe)
	    savefig(filename)
	    filename = 'Fig_%d_%s.svg' %(numfig,timeframe)
	    savefig(filename)

def makeFig():            
    # title(r'Posture to reach A $\rightarrow$ B '+strat)                                      
    # xlabel('X')
    # ylabel('Y')                        
    # clf()
    #print tt,name1,name2
    figure(3)
    clf()
    subplot(221)
    imshow(reshape(img2,(img_side,img_side)),interpolation='none')
    #imshow(img1)
    ylabel("Y")
    title("End effector position")
    subplot(222)
    imshow(reshape(img_transformation,(20,20)),interpolation='none')
    ylabel("Y")
    title("Predicted position of the object")
    subplot(223)
    imshow(reshape(img_y2,(20,20)),interpolation='none')
    #imshow(img2)
    xlabel("X")
    ylabel("Y")
    title("Object position")
    subplot(224)
    # #resp_outy=resp_out/img_x
    # imshow(reshape(resp_out,(13,13)),interpolation='none')
    # plot(synergy,lw=2)
    # xlabel("Motor neuron")
    # ylabel("Activity")
    # title("Motor synergy")
    # savefig("body_schema"+str(cpt)+".png")

def randrange(n, vmin, vmax):
    return (vmax-vmin)*rand(n) + vmin


def myNormalize(n, vmin, vmax):
    return (n-vmin)/(vmax-vmin)


def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    x = arange(0, size, 1, float)
    y = x[:,newaxis]
    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]    
    return exp(-4*log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)


#
ion()

who()

#####################
# CREATING DATABASE #
#####################

# ARM MOTOR ANGLES ON PLAN XY
L1=0.5 # length of arm link 1 in m
L2=0.3 # length of arm link 2 in m
L3=0.1 # length of arm link 3 in m

state = zeros(3) # vector for current state

x = cumsum([0,L1 * cos(state[0]), L2 * cos(state[0]+state[1]), L3 * cos(state[0]+state[2]+state[1])])
y = cumsum([0,L1 * sin(state[0]), L2 * sin(state[0]+state[1]), L3 * sin(state[0]+state[2]+state[1])])


# 1 MAKE A DATA BASE [POSITION;VISION;MOTION]
# 2 LEARN POSITION DELTA_W = C* - C   //  DELTA_W = C* V - S // DELTA_W = C* V - WP

nb_position=30
nb_vision=10

#position=reshape(randrange(3*nb_position, 0, 0.5*pi),(nb_position,3))
position=zeros((nb_position,3))
position[:,0]=randrange(nb_position, 0, 3*pi/4.0)
position[:,1]=randrange(nb_position, 0, pi)
position[:,2]=randrange(nb_position, -0.5*pi, 0.5*pi)

#position=reshape(randrange(3*nb_position, 0, 2*pi),(nb_position,3))

vision=arange(0,2*pi,2*pi/(1.0*nb_vision))

# MOTION RANDOM1
#motion=reshape(randrange(3*nb_motion, -0.05*2*pi, 0.05*2*pi),(nb_motion,3))
# MOTION RANDOM2
#motion=0.03*randint(low=-1,high=2,size=(nb_motion,3))

# MOTION RANGE
nb_motion1=3
nb_motion=nb_motion1**3

motion=zeros((nb_motion,3))

#ddq= 0.1*2*pi*(arange(nb_motion1)/(1.0*nb_motion1)-0.5) #0.05*2*pi
#ddq= 0.1*2*pi*arange(nb_motion1)/nb_motion1-0.05*2*pi
#ddq=arange(-nb_motion1/2,nb_motion1/2)
ddq=arange(-(nb_motion1)/2+1,(1+nb_motion1)/2)*0.05*2*pi

idx=0
for motion1 in range(nb_motion1):
    for motion2 in range(nb_motion1):
        for motion3 in range(nb_motion1):
            motion[idx,0]=ddq[motion1]
            motion[idx,1]=ddq[motion2]
            motion[idx,2]=ddq[motion3]
            idx=idx+1

# DATABASE
nb_data=nb_position*nb_motion
data=zeros((nb_position*nb_motion,13))

step=0
for pos_idx in range(nb_position):
    state = position[pos_idx, :]
    x = cumsum([0,L1 * cos(state[0]), L2 * cos(state[0]+state[1]), L3 * cos(state[0]+state[2]+state[1])])
    y = cumsum([0,L1 * sin(state[0]), L2 * sin(state[0]+state[1]), L3 * sin(state[0]+state[2]+state[1])])        
    for motion_idx in range(nb_motion):
        #dstate = randrange(3, -0.05*2*pi, 0.05*2*pi) + state # new position
        dstate = motion[motion_idx,:] + state 
        x1 = cumsum([0,L1 * cos(dstate[0]), L2 * cos(dstate[0]+dstate[1]), L3 * cos(dstate[0]+dstate[2]+dstate[1])])
        y1 = cumsum([0,L1 * sin(dstate[0]), L2 * sin(dstate[0]+dstate[1]), L3 * sin(dstate[0]+dstate[2]+dstate[1])])
        vpd=arctan2(y1-y,x1-x) # visual orientation
        # desired V x W P = desired C // desired C / desired V 
        # VECTOR DIM V [1 etat] // VECTOR DIM P [3 etats] // VECTOR DIM C [3 etats]
        data[step,0]=state[0]             #position
        data[step,1]=state[1]             #position
        data[step,2]=state[2]             #position
        data[step,3]=x[3]                 #end effector
        data[step,4]=y[3]                 #end effector
        data[step,5]=dstate[0]-state[0]   #motor command
        data[step,6]=dstate[1]-state[1]   #motor command
        data[step,7]=dstate[2]-state[2]   #motor command
        data[step,8]=x1[3]                #object position
        data[step,9]=y1[3]                #object position
        data[step,10]=vpd[3]              #visual orientation #NEW170316
        data[step,11]=pos_idx          #visual orientation #NEW170316
        data[step,12]=motion_idx          #visual orientation #NEW170316
        step=step+1


neg_rad=where(data[:,10]<0)[0]
#180*(pi-data[neg_rad,10])/pi
data[neg_rad,10]=2*pi+data[neg_rad,10]

pickle.dump(position,open("position.p","wb"))
pickle.dump(motion,open("motion.p","wb"))
pickle.dump(data,open("data.p","wb"))

position=pickle.load(open("position.p","rb"))
data=pickle.load(open("data.p","rb"))
motion=pickle.load(open("motion.p","rb"))    

# Fig 1 POINTS
figure(1)
clf()
plot(data[:,8],data[:,9],'r.')
plot(data[:,3],data[:,4],'b.')

save(1)

###############
###############

# IMAGE TO IMAGE TRANSFORMATION

nb_neuron=nb_motion

img_side=13
img_size=img_side*img_side

min_x=min(data[:,3])
min_y=min(data[:,4])

max_x=max(data[:,3])
max_y=max(data[:,4])

############################
## INITIALIZE THE NETWORKS #
############################

from neuron import Neuron
from neuron import Network

# NET1 IM1 x IM2 -> MOTOR SYNERGY
network_in=Network(nbNeuron=nb_neuron, inputDim=img_size, learningRate=0.1)
# NET2 MOTOR SYNERGY -> IM1 x IM2
network_out=Network(nbNeuron=img_size, inputDim=nb_neuron, learningRate=0.1)


# AUTO ENCODER RECONSTRUCTION

# AUTO ENCODER LEARNING 2 SDG
nb_step=75
nb_sdg=5

globErr1_dyn=zeros(nb_step)
globErr2_dyn=zeros(nb_step)

for step in range(nb_step):
    data_sdg = randint(nb_data, size=nb_sdg) # SGD
    globErr1=0
    globErr2=0
    for data_idx in data_sdg:
        norm_x1=img_side*myNormalize(data[data_idx,3], min_x, max_x)
        norm_y1=img_side*myNormalize(data[data_idx,4], min_y, max_y)
        norm_x2=img_side*myNormalize(data[data_idx,8], min_x, max_x)
        norm_y2=img_side*myNormalize(data[data_idx,9], min_y, max_y)
        img1=makeGaussian(img_side, fwhm = 5, center=[norm_x1, norm_y1])
        img2=makeGaussian(img_side, fwhm = 5, center=[norm_x2, norm_y2]) 
        matmult=ravel(img1*img2)
        #matmult=ravel(img_transformation)
        resp_in=network_in.response(ravel(matmult))
        motion_idx=int(data[data_idx,12])
        err=1-resp_in[motion_idx]
        #network_in.neuron[motion_idx].updateWeights(matmult,err)
        network_in.neuron[motion_idx].updateWeights(matmult,1)
        for idx in range(nb_motion):
            if idx != motion_idx:
                network_in.neuron[idx].updateWeights(matmult,0)        
        globErr1+=err
        resp_out=network_out.response(resp_in)
        for nn_idx in range(img_size):
            err=matmult[nn_idx]-resp_out[nn_idx]
            network_out.neuron[nn_idx].updateWeights(resp_in,err)
            globErr2+=err
    print('step %d glob err1 %f err2 %f' %(step, globErr1,globErr2))
    globErr1_dyn[step]=globErr1
    globErr2_dyn[step]=globErr2
    if globErr1<0.001:
        break


pickle.dump(network_in, open("network_in.p","wb"))
pickle.dump(network_out, open("network_out.p","wb"))
pickle.dump([globErr1_dyn,globErr2_dyn],open("error.p","wb"))

timeframe=time.strftime('%Y%m%d%H%M%S')
pickle.dump(network_in, open("network_in_%s.p" %timeframe,"wb"))
pickle.dump(network_out, open("network_out_%s.p" %timeframe,"wb"))
pickle.dump([globErr1_dyn,globErr2_dyn],open("error_%s.p" %timeframe,"wb"))


# Fig 2 ERROR
figure(2)
clf()
plot(globErr1_dyn,lw=2)
plot(globErr2_dyn,lw=2)

#save(2)

# XP 1 à une position fixe de la main faire bouger les objets autour de la main (en  étoile)

min_x=min(data[:,3])
min_y=min(data[:,4])

max_x=max(data[:,3])
max_y=max(data[:,4])


currErr1=0
currErr2=0

data_idx = randint(nb_data) # select one position

norm_x1=img_side*myNormalize(data[data_idx,3], min_x, max_x)
norm_y1=img_side*myNormalize(data[data_idx,4], min_y, max_y)

norm_x2=img_side*myNormalize(data[data_idx,8], min_x, max_x)
norm_y2=img_side*myNormalize(data[data_idx,9], min_y, max_y)

img1=makeGaussian(img_side, fwhm = 5, center=[norm_x1, norm_y1])
img2=makeGaussian(img_side, fwhm = 5, center=[norm_x2, norm_y2]) 
matmult=img1*img2

figure(3)
clf()
subplot(221)
imshow(img1)
subplot(223)
imshow(img2)
subplot(222)
#imshow(reshape(matmult,(img_side,img_side)))
plot(resp_in)

#matmult=ravel(img_transformation)
resp_in=network_in.response(ravel(matmult))
motion_idx=int(data[data_idx,12])
err=1-resp_in[motion_idx]
#network_in.neuron[motion_idx].updateWeights(matmult,err)
currErr1+=err
resp_out=network_out.response(resp_in)
for nn_idx in range(img_size):
    err=ravel(matmult)[nn_idx]-resp_out[nn_idx]
    #network_out.neuron[nn_idx].updateWeights(resp_in,err)
    currErr2+=err

print('step %d glob err1 %f err2 %f' %(step, currErr1,currErr2))

subplot(224)
imshow(reshape(resp_out,(img_side,img_side)))
#plot(resp_in)


save(3)
