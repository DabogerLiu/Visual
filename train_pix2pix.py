import os
from skimage.transform import resize
import numpy as np
from morph import dilate, erode, adapt
import keras
from keras.datasets import mnist
import tensorflow as tf
from keras import backend as K
from keras.layers import Input, Conv2D, Activation, Add, Subtract
from keras import optimizers
from keras.models import Model
import matplotlib.pyplot as plt
import cv2
import argparse
from model import *

# example of pix2pix gan for satellite to map image-to-image translation

from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from utils import *
from options import *


n_batch =args.batch_size
n_epochs = args.epochs

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=1):
	# determine the output square shape of the discriminator
	n_patch = d_model.output_shape[1]
	# unpack dataset
	trainA, trainB = dataset
	# calculate the number of batches per training epoch
	bat_per_epo = int(len(trainA) / n_batch)
	# calculate the number of training iterations
	n_steps = bat_per_epo * n_epochs
	# manually enumerate epochs
	for i in range(n_steps):
		# select a batch of real samples
		[X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
		# generate a batch of fake samples
		X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
		# update discriminator for real samples
		d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
		# update discriminator for generated samples
		d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
		# update the generator
		g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])
		# summarize performance
		print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)

def main():
	# load image data
	dataset = load_real_samples('maps_256.npz')
	print('Loaded', dataset[0].shape, dataset[1].shape)
	# define input shape based on the loaded dataset
	image_shape = dataset[0].shape[1:]
	# define the models
	d_model = discriminator(image_shape)
	g_model = generator(image_shape)
	# define the composite model
	gan_model = gan(g_model, d_model, image_shape)
	# train model
	train(d_model, g_model, gan_model, dataset)

if __name__ =='__main__':
	print('................')
	main()