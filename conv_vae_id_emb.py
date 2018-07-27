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
from keras.layers import Input, Dense, Lambda, Flatten, Reshape, concatenate
from keras.layers import Conv2D, Conv2DTranspose,Embedding,multiply
from keras.models import Model,Sequential
from keras import backend as K
from keras import metrics
from scipy.misc import imsave
from HandleData import num_id,get_num_files,load_data_imdb,enumerate_files,load_part,get_num_id_files
from PIL import Image
from keras.metrics import binary_crossentropy
from networks_def_tf import InceptionV3,vgg_face,squeezenet
import gc
epochs = 50
import tensorflow as tf

age_stats = 0 

class AgeStats:
	def __init__(self):
		self.processed_ages = [0]*11
	def add(self,ages):
		for i in ages:
			self.processed_ages[int(i)] = self.processed_ages[int(i)]+1
		

	def print_stats(self):
		x = sum(self.processed_ages)		
		for i in range(0,11):
			print("Age group" + str(i) + ": " + str(self.processed_ages[i]*100/x) + "%")


		
 
# input image dimensions
num_classes = 11
img_rows, img_cols, img_chns = 224, 224, 3
# number of convolutional filters to use
filters = 64
# convolution kernel size
num_conv = (3,3)

batch_size = 16
if K.image_data_format() == 'channels_first':
    original_img_size = (img_chns, img_rows, img_cols)
else:
    original_img_size = (img_rows, img_cols, img_chns)
latent_dim = 64
intermediate_dim = 128
epsilon_std = 0.001

x = Input(shape=original_img_size)
labels = Input(shape=(11,))
  
conv_1 = Conv2D(img_chns,
                kernel_size=(2, 2),
                padding='same', activation='relu',name='encoder_conv1')(x)
conv_2 = Conv2D(filters,
                kernel_size=(2, 2),
                padding='same', activation='relu',
                strides=(2, 2),name='encoder_conv2')(conv_1)
conv_3 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1,name='encoder_conv3')(conv_2)
conv_4 = Conv2D(filters,
                kernel_size=num_conv,
                padding='same', activation='relu',
                strides=1,name='encoder_conv4')(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu',name='encoder_hidden')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon



z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])
z_cond = concatenate([z,labels])
decoder_hid = Dense(intermediate_dim, activation='relu',name='decoder_hid')
decoder_upsample = Dense(filters * img_rows/2 * img_cols/2, activation='relu',name='decoder_upsmp')
if K.image_data_format() == 'channels_first':
    output_shape = (batch_size,filters,img_rows/2,img_cols/2)
    #we had ou	
else:
    output_shape = (batch_size, img_rows/2, img_cols/2,filters)
decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=(1,1),
                                   activation='relu',name='decoder_deconv1')
decoder_deconv_2 = Conv2DTranspose(filters,
                                   kernel_size=num_conv,
                                   padding='same',
                                   strides=(1,1),
                                   activation='relu',name='decoder_deconv2')
decoder_deconv_3_upsamp = Conv2DTranspose(filters,
                                          kernel_size=(3, 3),
                                          strides=(2, 2),
                                          padding='valid',
                                          activation='relu',name='decoder_deconv3')
decoder_mean_squash = Conv2D(img_chns,
                             kernel_size=2,
                             padding='valid',
                             activation='sigmoid',
			     name='decoder_mean'
			     )


hid_decoded = decoder_hid(z_cond)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

x_decoded_mean_squash = Reshape((img_rows,img_cols,img_chns),name='img_out')(x_decoded_mean_squash)



vae = Model([x, labels], x_decoded_mean_squash)

#squ = squeezenet(num_id,inshape=(224,224,3),output="denseFeatures")
#squ =  vgg_face(weights_path = 'db_trained_vgg.h5',N_classes=num_id)
squ = InceptionV3(N_classes=num_id,include_top=False)
squ.load_weights('db_trained_iv3_2.h5',by_name=True)
squ.name = "feats_out"
squ.trainable = False
squ.summary()
#combined model 
end2end = Sequential()
end2end.add(vae)
end2end.add(squ)

e2emodel = Model(inputs=end2end.input, outputs=[end2end.get_layer('img_out').output,end2end.output])

def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    #y_true = K.flatten(y_true)
    #y_pred = K.flatten(y_pred)
    recon = K.sum(K.sum(K.binary_crossentropy(y_pred, y_true), axis=-1),axis=-1)
    #recon = K.sum(K.binary_crossentropy(y_pred,y_true),axis=1)
    #recon = img_rows*img_cols * binary_crossentropy(y_pred,y_true)
    # D_KL(Q(z|X) || P(z|X))
    kl = K.sum(0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1),axis=-1)
    #kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var,axis=1)
    #kl = -0.5 * K.mean(1 + z_log_var - K.exp(z_log_var) - K.square(z_mean),axis=-1)
    return K.mean(recon + 0.01*kl)

def id_loss(y_true,y_pred):
    y_true = K.l2_normalize(y_true,axis=-1)
    y_pred = K.l2_normalize(y_pred,axis=-1)
    return 1-K.sum(y_true*y_pred,axis=-1)


def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, axis=-1))

def recon_loss(y_true, y_pred):
	return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))


encoder = Model([x, labels],z_mean)
decoder_input_1 = Input(shape=(latent_dim,))
cond = Input(shape=(11,))
decoder_input = concatenate([decoder_input_1, cond])
_hid_decoded = decoder_hid(decoder_input)
_up_decoded = decoder_upsample(_hid_decoded)
_reshape_decoded = decoder_reshape(_up_decoded)
_deconv_1_decoded = decoder_deconv_1(_reshape_decoded)
_deconv_2_decoded = decoder_deconv_2(_deconv_1_decoded)
_x_decoded_relu = decoder_deconv_3_upsamp(_deconv_2_decoded)
_x_decoded_mean_squash = decoder_mean_squash(_x_decoded_relu)
generator = Model([decoder_input_1, cond], _x_decoded_mean_squash)




vae.compile(optimizer='rmsprop',loss=vae_loss,metrics=[KL_loss,recon_loss])
losses = {'img_out': vae_loss, 'feats_out': id_loss }
e2emodel.compile(optimizer='adam',loss=losses, loss_weights=[1,0])
age_stats = AgeStats()


#num_data = get_num_files()
num_data = get_num_id_files()
writer = tf.summary.FileWriter('logs/conv_cvae')
#vae.load_weights('saved_models/deep_feature_consistent.h5')
for e in range(epochs):
	print('Epoch ', e, '/', epochs)
	batches = 0
	for i in range(0,num_data/batch_size):
		
		(x_train,y_train,y_id) = load_data_imdb(i*batch_size,(i+1)*batch_size,squ)
		age_stats.add(y_train)
		age_stats.print_stats()
		y_train = map(lambda x: np.eye(11)[int(x)],y_train)
		y_train = np.asarray(y_train)
		loss = e2emodel.train_on_batch([x_train,y_train],[x_train,np.squeeze(y_id)])
		print("Epoch: ",str(e),', Batch: ',str(i),'Loss: ', str(loss))
		weighted_loss = tf.Summary(value=[tf.Summary.Value(tag='weighted_loss',simple_value=loss[0]),])
		img_loss = tf.Summary(value=[tf.Summary.Value(tag='img_loss',simple_value=loss[1]),])
		id_loss  = tf.Summary(value=[tf.Summary.Value(tag='id_loss',simple_value=loss[2]),])
		writer.add_summary(weighted_loss)
		writer.add_summary(img_loss)
		writer.add_summary(id_loss)
		if (i % 100 == 0):

			vae.save_weights('saved_models/deep_feature_consistent.h5')
			ref_img = Image.open('imgs/ref.jpg')
			ref_img.load()
			ref_img = ref_img.resize((img_rows,img_cols),Image.BILINEAR)
			ref_img = np.asarray(ref_img,dtype="float32") / 255. 

			ref_img = ref_img.reshape(1,img_rows,img_cols,img_chns)

			initial_age_group = 2
			target_age_group = 6
			init = encoder.predict([ref_img,np.eye(11)[initial_age_group].reshape(1,-1)])

			for i in range(0,10):
				generated_aged = generator.predict([init,np.eye(11)[i].reshape(1,-1)])
				formatted = (generated_aged * 255 / np.max(generated_aged)).astype('uint8')
				img = Image.fromarray(formatted[0][:,:,:])
				img.save('imgs/aged_' + str(i) + '.jpg')
				

ref_img = Image.open('imgs/ref.jpg')
ref_img.load()
ref_img = ref_img.resize((img_rows,img_cols),Image.BILINEAR)
ref_img = np.asarray(ref_img,dtype="float32") / 255. 

ref_img = ref_img.reshape(1,img_rows,img_cols,img_chns)

initial_age_group = 2
target_age_group = 6
init = encoder.predict([ref_img,np.eye(11)[initial_age_group].reshape(1,-1)])

for i in range(0,10):
	generated_aged = generator.predict([init,np.eye(11)[i].reshape(1,-1)])
	formatted = (generated_aged * 255 / np.max(generated_aged)).astype('uint8')
	img = Image.fromarray(formatted[0][:,:,:])
	img.save('imgs/aged_' + str(i) + '.jpg')



def f(initial_approx):
	if (initial_approx.shape == (20,)):
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

