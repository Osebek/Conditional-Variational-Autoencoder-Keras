'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
# Reference
- Auto-Encoding Variational Bayes
  https://arxiv.org/abs/1312.6114
'''
from __future__ import print_function
import h5py
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from compare_imgs import calculated_distance_mixed
from keras.layers import Input, BatchNormalization,Dense, Lambda, Flatten, Reshape, concatenate
from keras.layers import Conv2D, Conv2DTranspose,Embedding,multiply
from keras.models import Model,Sequential,load_model
from keras import backend as K
from keras import metrics
from keras.optimizers import adam
from scipy.misc import imsave
from scipy.spatial import distance
from HandleData import enumerate_files,load_part,load_part_dfc,load_part_dfc_age,num_id,load_data_imdb,get_num_files
from PIL import Image
from networks_def_tf import vgg_face,InceptionV3,squeezenet
import gc
import tensorflow as tf
from keras.applications.vgg19 import VGG19
from keras.models import model_from_json


epochs = 15 
# input image dimensions
num_classes = 11
img_rows, img_cols, img_chns = 224, 224, 3
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = (3,3)

batch_size = 8
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
#latent_dim = 128
#intermediate_dim = 256
latent_dim = 64
intermediate_dim = 128
epsilon_std = 0
nf = (1.0/(2.0*51380224.0))

dfc_mode = 'vae_123'

x = Input(shape=original_img_size)

conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2))(conv_1)
#conv_2 = BatchNormalization()(conv_2)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1)(conv_2)
#conv_3 = BatchNormalization()(conv_3)
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



z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(filters * img_rows/2 * img_cols/2, activation='relu')

if K.image_data_format() == 'channels_first':
    output_shape = (batch_size,filters,img_rows/2,img_cols/2)
else:
    output_shape = (batch_size, img_rows/2, img_cols/2,filters)

decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=(1,1),
                                   activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=(1,1),
                                   activation='relu')
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid',
			     )

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)
x_decoded_mean_squash = Reshape((img_rows,img_cols,img_chns),name='img_out')(x_decoded_mean_squash)


vae = Model(x, x_decoded_mean_squash)

squ =  vgg_face(weights_path = 'TrainedNets/recognition-nets/weights/vgg_face_weights_tf.h5',output_layer=dfc_mode)

comp =  vgg_face(weights_path = 'TrainedNets/recognition-nets/weights/vgg_face_weights_tf.h5',output_layer='prob')
#squ = VGG19(weights='imagenet',output_layer=dfc_mode)
squ.trainable = False
end2end = squ(x_decoded_mean_squash)
end2end.append(x_decoded_mean_squash)
e2emodel = Model(inputs=x, outputs=end2end)
#e2emodel_frozen = Model(inputs=x,outputs=end2end)
#e2emodel_all = Model(inputs=x,outputs=end2end)

#for j in range(17,len(e2emodel_frozen.layers)):
#	e2emodel_frozen.layers[j].trainable = False

def vae_loss1(y_true,y_pred):
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)
	recon = K.mean(K.binary_crossentropy(y_pred,y_true))
	kl = -0.5*K.sum(1 + z_log_var - K.exp(z_log_var) - K.square(z_mean),axis=-1)
	#return 0.000*kl+recon 
	return kl+recon

def vae_loss(y_true, y_pred):
	recon = K.sum(K.sum(K.binary_crossentropy(y_pred,y_true),axis=-1),axis=-1)
	kl = K.sum(0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var,axis=-1),axis=-1)
	return K.mean(10*kl+recon)
#	return K.mean(0.0005*kl) ti parametri so precej dobri, skupaj z =1 epsilon_stda in  0.01 dfc_w weigh dfc_weightt 
	#return kl
def id_loss(y_true,y_pred):
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)
	return K.sum(K.square(y_true-y_pred),axis=-1)


def id_loss1(y_true,y_pred):
	return K.sum(K.sum(K.sum(K.sum(K.square(y_true-y_pred),axis=-1),axis=-1),axis=-1))


def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1))

def recon_loss(y_true, y_pred):
	return(K.sum(K.binary_crossentropy(y_true,y_pred), axis=1))


encoder = Model(x,z_mean)
decoder_input = Input(shape=(latent_dim,))
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model(decoder_input, _x_decoded_mean_squash)


losses = {'img_out': vae_loss, 'vgg-face': id_loss }


if dfc_mode == 'vae_whole':
	dfc_w = [x*0.5 for x in [nf,nf,nf/2,nf/2,nf/4,nf/4,nf/4,nf/8,nf/8,nf/8,nf/32,nf/32,nf/32]] 
elif dfc_mode == 'vae_123':
	dfc_w = [x*0.01 for x in [nf,nf/2,nf/4]]
elif dfc_mode == 'vae_345':
	dfc_w =[x*0.5 for x in [nf/4,nf/8,nf/32]]

optm = adam(lr=0.0005)
#e2emodel_all.compile(optimizer=optm,loss=losses,loss_weights=[0 for x in dfc_w]+[1])
#for j in range(17,len(e2emodel_frozen.layers)):
#	e2emodel_all.layers[j].trainable = True

vae.compile(optimizer='adam',loss=vae_loss,metrics=[KL_loss,recon_loss])
#e2emodel_frozen.compile(optimizer=optm,loss=losses,loss_weights=dfc_w+[0])
e2emodel.compile(optimizer=optm,loss=losses,loss_weights=dfc_w+[1])


num_data = get_num_files()
#num_data = 22000
num_data = 5000
#vae.load_weights('saved_models/color_vgg_1.h5')
#vae.load_weights('saved_models/color_noncond5.h5')
writer = tf.summary.FileWriter('logs/dfc/')
for e in range(epochs):
	print('Epoch ', e, '/', epochs)
	batches = 0
	for i in range(0,num_data/batch_size):
		(x_train,y_train,df) = load_part_dfc_age('UTKFace/',i*batch_size,(i+1)*batch_size,squ)
		if dfc_mode == 'vae_whole':
			for j in range(17,len(e2emodel.layers)):
				e2emodel.layers[j].trainable = False
			loss = e2emodel.train_on_batch([x_train,y_train],
			[df[0],df[1],df[2],df[3],df[4],df[5],df[6],df[7],df[8],df[9],df[10],df[11],df[11],x_train])
			
			print("Epoch: ",str(e),', Batch: ',str(i),'Loss KL: ', str(loss))
			for j in range(17,len(e2emodel.layers)):
				e2emodel.layers[j].trainable = True
			
			loss = e2emodel.train_on_batch([x_train,y_train],
			[df[0],df[1],df[2],df[3],df[4],df[5],df[6],df[7],df[8],df[9],df[10],df[11],df[11],x_train])

			summary_vae = tf.Summary(value=[tf.Summary.Value(tag='vae_loss',simple_value=loss[0]),])
			writer.add_summary(summary_vae)
			print("Epoch: ",str(e),', Batch: ',str(i),'Loss REC: ', str(loss))
		elif dfc_mode == 'vae_123'or dfc_mode=='vae_345':
			loss = e2emodel.train_on_batch(x_train,[df[0],df[1],df[2],x_train])
			print("Epoch: ",str(e),', Batch: ',str(i),'Loss REC: ', str(loss))
			summary_vae = tf.Summary(value=[tf.Summary.Value(tag='vae_loss',simple_value=loss[0]),])
			writer.add_summary(summary_vae)


		if (i % 25 == 0) :
			ref_img = Image.open('imgs/ref.jpg')
			ref_img.load()
			ref_img = ref_img.resize((img_rows,img_cols),Image.BILINEAR)
			ref_img = np.asarray(ref_img,dtype="float32") / 255. 

			ref_img = ref_img.reshape(1,img_rows,img_cols,img_chns)

			initial_age_group = 2
			target_age_group = 6
			init = encoder.predict(ref_img)

			generated_aged = generator.predict(init)
			formatted = (generated_aged * 255 / np.max(generated_aged)).astype('uint8')
			img = Image.fromarray(formatted[0][:,:,:])
			img.save('imgs/aged.jpg')
	vae.save_weights('saved_models/color_noncond2_2.h5')

#vae.load_weights('saved_models/test12.h5')
	
ref_img = Image.open('imgs/ref.jpg')
ref_img.load()
ref_img = ref_img.resize((img_rows,img_cols),Image.BILINEAR)
ref_img = np.asarray(ref_img,dtype="float32") / 255.
ref_img = ref_img.reshape(1,img_rows,img_cols,img_chns)
x = squ.predict(ref_img)

init = encoder.predict(ref_img)
generated = generator.predict(init)

bb = (generated * 255 / np.max(generated)).astype('uint8')
img = Image.fromarray(bb[0][:,:,:])
img.save('imgs/test.jpg')


def f(initial_approx):
	if (initial_approx.shape == (latent_dim,)):
		initial_approx = initial_approx.reshape(1,-1)

	generated = generator.predict(initial_approx)
	
	v1 = comp.predict(ref_img)[0]
	v2 = comp.predict(generated)[0]
	d_cos = distance.cosine(v1,v2)
	d_euc = distance.euclidean(v1,v2)
	return d_cos

#a = minimize(f,init,method='L-BFGS-B')
a = minimize(f,init,method='COBYLA')

#a = minimize(f,init,method='Nelder-Mead',options={'maxiter': 10000})
generated = generator.predict(a.x.reshape(1,-1))
generated_aged = generator.predict(a.x.reshape(1,-1))


formatted = (generated * 255 / np.max(generated)).astype('uint8')
img = Image.fromarray(formatted[0][:,:,:])
img.save('imgs/new_approx.jpg')


formatted = (generated_aged * 255 / np.max(generated_aged)).astype('uint8')
img = Image.fromarray(formatted[0][:,:,:])
img.save('imgs/new_approx_aged.jpg')
	


	

	
