'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
# Reference
- Auto-Encoding Variational Bayes
  https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from compare_imgs import calculated_distance_mixed
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, concatenate
from keras.layers import Conv2D, Conv2DTranspose,Embedding,multiply
from keras.models import Model
from keras import backend as K
from keras import metrics
from scipy.misc import imsave
from HandleData import read_utk_face
from PIL import Image
from keras.datasets import mnist
epochs = 40
# input image dimensions
num_classes = 11
img_rows, img_cols, img_chns = 150, 150, 1
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = 3

batch_size = 50
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 15
intermediate_dim = 75
epsilon_std = 0

x = Input(shape=original_img_size)
labels = Input(shape=(1,))
label_embedding = Flatten()(Embedding(num_classes,np.prod(original_img_size))(labels))
flat_img = Flatten()(x)
model_input = multiply([flat_img, label_embedding])
model_input = Reshape((img_rows,img_cols,img_chns))(model_input)
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(model_input)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
# tuki morem nekak konketat se lejbl zravn
z_cond = concatenate([z,labels])
# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * img_rows/2 * img_cols/2, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, img_rows/2, img_cols/2)
else:
    output_shape = (batch_size, img_rows/2, img_cols/2, filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=1,
                                   activation='relu')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size, filters, img_rows+1, img_cols+1)
else:
    output_shape = (batch_size, img_rows+1, img_cols+1, filters)
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid')

hid_decoded = decoder_hid(z_cond)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

# instantiate VAE model
vae = Model([x, labels], x_decoded_mean_squash)
# vae2 = Model..
# Compute VAE loss
#xent_loss = img_rows * img_cols * metrics.binary_crossentropy(
#   K.flatten(x),
#    K.flatten(x_decoded_mean_squash))
#kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
#vae_loss = K.mean(xent_loss + kl_loss)
#vae.add_loss(vae_loss)

def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1)
    return recon + 0.1*kl

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=1))

def recon_loss(y_true, y_pred):
	return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))




vae.compile(optimizer='rmsprop',loss=vae_loss,metrics=[KL_loss,recon_loss])
#m2 = vae.compile(op

# train the VAE on MNIST digits
#(x_train, y_train), (x_test, y_test) = mnist.load_data()


(x_train, y_train), (x_test, y_test) = read_utk_face('UTKFace/',size=(img_rows,img_cols))


x_train = x_train.astype('float32') / 255.
x_train = x_train.reshape((x_train.shape[0],) + original_img_size)
y_train  = y_train.astype('float32')
x_test = x_test.astype('float32') / 255.
x_test = x_test.reshape((x_test.shape[0],) + original_img_size)

print('x_train.shape:', x_train.shape)

vae.load_weights('saved_models/test13_lowdim.h5')

vae.fit([x_train,y_train],
	x_train,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=([x_test,y_test], x_test))

vae.save_weights('saved_models/test13_lowdim.h5')

#vae.load_weights('saved_models/100epoch_biggerconv_kl++.h5')
#vae.load_weights('saved_models/50epoch_biggerconv.h5')
#vae.load_weights('conv_model_faces_10_big_ajdloss.h5')
# build a model to project inputs on the latent space
encoder = Model([x, labels],z_mean)

# display a 2D plot of the digit classes in the latent space
#x_test_encoded = encoder.predict([x_test,y_test], batch_size=batch_size)

# build a digit generator that can sample from the learned distribution
decoder_input_1 = Input(shape=(latent_dim,))
cond = Input(shape=(1,))

decoder_input = concatenate([decoder_input_1, cond])
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model([decoder_input_1, cond], _x_decoded_mean_squash)


ref_img = Image.open('imgs/ref.jpg').convert('L')
ref_img.load()
ref_img = ref_img.resize((img_rows,img_cols),Image.BILINEAR)
ref_img = np.asarray(ref_img,dtype="float32") / 255. 

ref_img = ref_img.reshape(1,img_rows,img_cols,img_chns)

initial_age_group = 2
target_age_group = 6
init = encoder.predict([ref_img,np.asarray(initial_age_group,dtype='int32').reshape(1,1)])

generated = generator.predict([init,np.asarray(initial_age_group,dtype='int32')
				   .reshape(1,1)])

for i in range(0,10):
	generated_aged = generator.predict([init,np.asarray(i,dtype='int32')
				   .reshape(1,1)])
	formatted = (generated_aged * 255 / np.max(generated_aged)).astype('uint8')
	img = Image.fromarray(formatted[0][:,:,0],'L')
	img.save('imgs/aged_' + str(i) + '.jpg')
	

#formatted = (generated * 255 / np.max(generated)).astype('uint8')
#img = Image.fromarray(formatted[0][:,:,0],'L')
#img.save('imgs/init_approx.jpg')
	

def f(initial_approx):
	if (initial_approx.shape == (latent_dim,)):
		initial_approx = initial_approx.reshape(1,-1)

	generated = generator.predict([initial_approx,
				np.asarray(initial_age_group,dtype='int32')
				.reshape(1,1)])

	formatted = (generated * 255 / np.max(generated)).astype('uint8')
	img = Image.fromarray(formatted[0][:,:,0],'L')
	return calculated_distance_mixed('imgs/ref.jpg',img)

a = minimize(f,init,method='COBYLA')
#a = minimize(f,np.zeros(10),method='Nelder-Mead',options={'maxiter': 10000})
generated = generator.predict([a.x.reshape(1,-1),np.asarray(initial_age_group,dtype='int32')
				   .reshape(1,1)])

generated_aged = generator.predict([a.x.reshape(1,-1),np.asarray(target_age_group,dtype='int32')
				   .reshape(1,1)])


formatted = (generated * 255 / np.max(generated)).astype('uint8')
img = Image.fromarray(formatted[0][:,:,0],'L')
img.save('imgs/new_approx.jpg')
	
formatted_aged = (generated_aged * 255 / np.max(generated_aged)).astype('uint8')
img = Image.fromarray(formatted_aged[0][:,:,0],'L')
img.save('imgs/new_approx_aged.jpg')	

'''
imsave('generated_imgs/ref_img.jpg',x_test[tin].reshape(img_rows,img_cols))
for i in range(0,10):
	generated = generator.predict([a, np.array(i).reshape(1,1)])
	imsave('generated_imgs/generated_img_' + str(i) + '_age' + '_.jpg',
		generated.reshape((img_rows,img_cols)))
'''
