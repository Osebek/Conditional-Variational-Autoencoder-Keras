import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from scipy.misc import imsave
from HandleData import read_utk_face
from keras.models import load_model
# load mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows = 128
img_cols = 128
img_chnls = 1
img_shape =  (img_rows,img_cols,img_chnls)
num_classes = 11
(X_train,y_train),(X_test,y_test) = read_utk_face('UTKFace/',size=(img_rows,img_cols))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_test = y_test[0:int(y_test.shape[0]/50)*50,:]
X_test = X_test[0:int(X_test.shape[0]/50)*50,:,:]
X_train = X_train[0:int(X_train.shape[0]/50)*50,:,:]
y_train = y_train[0:int(y_train.shape[0]/50)*50,:]

# convert y to one-hot, reshape x
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# select optimizer
optim = 'adam'

# dimension of latent space (batch size by latent dim)
m = 50
n_z = 100

# dimension of input (and label)
n_x = X_train.shape[1]
n_y = y_train.shape[1]
# nubmer of epochs
n_epoch = 15


##  ENCODER ##

# encoder inputs
X = Input(shape=(int(img_rows*img_cols), ))
cond = Input(shape=(n_y, ))

# merge pixel representation and label
inputs = concatenate([X, cond])


# dense ReLU layer to mu and sigma
h_q = Dense(512, activation='relu')(inputs)
mu = Dense(n_z, activation='linear')(h_q)
log_sigma = Dense(n_z, activation='linear')(h_q)

def sample_z(args):
    mu, log_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(log_sigma / 2) * eps


# Sampling latent space
z = Lambda(sample_z, output_shape = (n_z, ))([mu, log_sigma])

# merge latent space with label
z_cond = concatenate([z, cond])

##  DECODER  ##

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

# define loss (sum of reconstruction and KL divergence)
def vae_loss(y_true, y_pred):
    # E[log P(X|z)]
    recon = K.sum(K.binary_crossentropy(y_pred, y_true), axis=1)
    # D_KL(Q(z|X) || P(z|X))
    kl = 0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1)
    return recon + kl

def KL_loss(y_true, y_pred):
	return(0.5 * K.sum(K.exp(log_sigma) + K.square(mu) - 1. - log_sigma, axis=1))

def recon_loss(y_true, y_pred):
	return(K.sum(K.binary_crossentropy(y_pred, y_true), axis=1))


# compile and fit
cvae.compile(optimizer=optim, loss=vae_loss, metrics = [KL_loss, recon_loss])
cvae_hist = cvae.fit([X_train, y_train], X_train, batch_size=m, epochs=n_epoch,
							validation_data = ([X_test, y_test], X_test),
							callbacks = [EarlyStopping(patience = 5)])



cvae.save_weights('cvae_weights.h5')
#cvae.load_weights('saved_models/cvae_weights.h5')
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



pic_num = 0
variations = 3


print(y_test.shape)
print(X_test[0,:].reshape(1,-1).shape)
print(y_test[0])
a = encoder.predict([X_test[0,:].reshape(1,-1),y_test[0].reshape(1,-1)])
print(np.append(a,y_test[0]))

imsave('transition_50/ref_img.jpg',X_test[0].reshape(img_rows,img_cols))
print(y_test[0])
for i in range(0,10): 
	generated = decoder.predict(np.append(a,np.eye(11)[i]).reshape(1,-1))
	imsave('transition_50/generated_img_' + str(i) + '_age' + '_.jpg',generated.reshape((img_rows,img_cols)))
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
