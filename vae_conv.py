import numpy as np
from keras.layers import Input, Dense, Lambda, Flatten, multiply,Embedding,Conv2D
from keras.layers import Conv2DTranspose,Reshape
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from scipy.misc import imsave
from HandleData import read_utk_face
from keras.models import load_model
from PIL import Image
# load mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows = 128
img_cols = 128
img_chnls = 1
img_shape =  (img_rows,img_cols,img_chnls)
num_classes = 11
latent_dim = 2 
intermediate_dim = 128 
epsilon_std = 1.0
num_conv = 3 
filters = 64 

(X_train,y_train),(X_test,y_test) = read_utk_face('UTKFace/',size=(img_rows,img_cols))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_test = y_test[0:int(y_test.shape[0]/50)*50,:]
X_test = X_test[0:int(X_test.shape[0]/50)*50,:,:]
X_train = X_train[0:int(X_train.shape[0]/50)*50,:,:]
y_train = y_train[0:int(y_train.shape[0]/50)*50,:]

# convert y to one-hot, reshape x
#y_train = to_categorical(y_train)
y_train = y_train.astype('float32')

#y_test = to_categorical(y_test)
y_test = y_test.astype('float32')
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), img_rows,img_cols,img_chnls))
X_test = X_test.reshape((len(X_test), img_rows,img_cols,img_chnls))

# select optimizer
optim = 'adam'

# dimension of latent space (batch size by latent dim)
m = 50
batch_size = m
n_z = latent_dim

# dimension of input (and label)
n_x = X_train.shape[1]
n_y = y_train.shape[1]
# nubmer of epochs
n_epoch = 40


##  ENCODER ##

# encoder inputs
#X = Input(shape=(int(img_rows*img_cols), ))
#cond = Input(shape=(n_y, ))

img = Input(shape=img_shape)
label = Input(shape=(1,),dtype='int32')

label_embedding = Flatten()(Embedding(num_classes,np.prod(img_shape))(label))
flat_img = Flatten()(img)
model_input = multiply([flat_img,label_embedding])
model_input = Reshape((img_rows,img_cols,img_chnls))(model_input)

# merge pixel representation and label
#inputs = concatenate([X, cond])


conv_1 = Conv2D(img_chnls,
		kernel_size=(2,2),
		padding='same',
		activation='relu')(model_input)
conv_2 = Conv2D(filters,
		kernel_size=(2,2),
		padding='same',
		activation='relu',
		strides=(2,2))(conv_1)
conv_3 = Conv2D(filters,
		kernel_size=num_conv,
		padding='same',
		activation='relu',
		strides=1)(conv_2)
conv_4 = Conv2D(filters,
		kernel_size=num_conv,
		padding='same',
		activation='relu',
		strides=1)(conv_3)

flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim,activation='relu')(flat)
z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


# dense ReLU layer to mu and sigma
#h_q = Dense(512, activation='relu')(inputs)
#mu = Dense(n_z, activation='linear')(h_q)
#log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    z_mean, z_log_var = args
    eps = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), 
			  mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * eps


z = Lambda(sample_z, output_shape = (latent_dim, ))([z_mean, z_log_var])


# Sampling latent space

print(z.shape)
# merge latent space with label
#z_cond = concatenate([z, label])

##  DECODER  ##
decoder_hid = Dense(intermediate_dim,activation='relu')
decoder_upsample = Dense(filters*img_rows/2 * img_cols/2,activation='relu')
output_shape = (batch_size,img_rows/2,img_cols/2,filters)
decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Conv2DTranspose(filters,
				  kernel_size=num_conv,
				  padding='same',
				  strides=1,
				  activation='relu')
decoder_deconv_2 = Conv2DTranspose(filters,
				   kernel_size=num_conv,
				   strides=(2,2),
			    	   padding='valid',
				   activation='relu')
output_shape = (batch_size,img_rows,img_cols,filters)

decoder_deconv_3_upsamp = Conv2DTranspose(filters,
					  kernel_size = (3,3),
					  strides = (2,2),
					  padding='valid',
					  activation='relu')

decoder_mean_squash = Conv2D(img_chnls,
			     kernel_size=2,
			     padding='valid',
			     activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)				
vae = Model([img, label], x_decoded_mean_squash)
#encoder = Model([img, label],z_mean)


'''
# dense ReLU to sigmoid layers
decoder_hidden = Dense(512, activation='relu')
decoder_out = Dense(int(img_rows*img_cols), activation='sigmoid')
h_p = decoder_hidden(z_cond)
outputs = decoder_out(h_p)

# define cvae and encoder models
cvae = Model([X, cond], outputs)
encoder = Model([X, cond], mu)

# reuse decoder layers to define decoder separately
d_in = Input(shape=(n_z+n_y,))
d_h = decoder_hidden(d_in)
d_out = decoder_out(d_h)
decoder = Model(d_in, d_out)
'''
# define loss (sum of reconstruction and KL divergence)

def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(z_log_var) + K.square(z_mean) - 1. - z_log_var, 
	 axis=1)
    return recon + kl

def KL_loss(y_true, y_pred):
    return(0.5 * K.sum(K.exp(z_log_var)+K.square(z_mean) - 1.- z_log_var
		, axis=1))

def recon_loss(y_true, y_pred):
	return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))


# compile and fit
vae.compile(optimizer='rmsprop', loss=vae_loss, metrics = [KL_loss, recon_loss])
#cvae.load_weights('cvae_weights_40.h5')
vae_hist = vae.fit([X_train, y_train], X_train, batch_size=m, epochs=n_epoch,
							validation_data = ([X_test, y_test], X_test),
							callbacks = [EarlyStopping(patience = 5)])



#cvae.save_weights('cvae_weights_40.h5')
#cvae.load_weights('cvae_weights.h5')
# this loop prints the one-hot decodings

#for i in range(n_z+n_y):
#	tmp = np.zeros((1,n_z+n_y))
#	tmp[0,i] = 1
#	generated = decoder.predict(tmp)
#	file_name = './img' + str(i) + '.jpg'
#	print(generated)
#	imsave(file_name, generated.reshape((28,28)))
#	sleep(0.5)

# this loop prints a transition through the number line

'''

pic_num = 0
variations = 3

veronika = Image.open('veronika_ref.jpg').convert('L')
veronika.load()
veronika = veronika.resize((img_rows,img_cols),Image.BILINEAR)
veronika = np.asarray(veronika,dtype="float32") / 255.
veronika = veronika.reshape(1,np.prod(veronika.shape))
veronika_age = np.eye(11)[2].reshape(1,-1)

tin  = 5
a = encoder.predict([X_test[tin,:].reshape(1,-1),y_test[tin].reshape(1,-1)])
veronika_prd = encoder.predict([veronika.reshape(1,-1),veronika_age])
imsave('transition_50/ref_img.jpg',X_test[tin].reshape(img_rows,img_cols))
print(y_test[0])
for i in range(0,10): 
	#generated = decoder.predict(np.append(a,np.eye(11)[i]).reshape(1,-1))
	generated = decoder.predict(np.append(veronika_prd,np.eye(11)[i]).reshape(1,-1))
	imsave('transition_50/generated_img_' + str(i) + '_age' + '_.jpg',generated.reshape((img_rows,img_cols)))
'''
'''

for j in range(n_z, n_z + n_y - 1):
	for k in range(variations):
		v = np.zeros((1, n_z+n_y))
		v[0, j] = 1 - (k/variations)
		v[0, j+1] = (k/variations)
		generated = decoder.predict(v)
		pic_idx = j - n_z + (k/variations)
		file_name = 'transition_50/' + str(j) + '_' + str(k) + '.jpg'
		imsave(file_name, generated.reshape((img_rows,img_cols)))
		pic_num += 1
		
'''
		
