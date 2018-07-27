import numpy as np
from keras.layers import Input, Dense, Lambda,Embedding,Flatten,Conv2D,UpSampling2D,multiply,Reshape,MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model
from keras import backend as K
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping,TensorBoard
from scipy.misc import imsave
from HandleData import read_utk_face
from keras.models import load_model
from PIL import Image

# load mnist
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_rows = 128
img_cols = 128
img_chnls = 1
img_shape =  (img_rows,img_cols,1)
num_classes = 11
(X_train,y_train),(X_test,y_test) = read_utk_face('UTKFace/',size=(img_rows,img_cols))
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
y_test = y_test[0:int(y_test.shape[0]/50)*50,:]
X_test = X_test[0:int(X_test.shape[0]/50)*50,:,:]
X_train = X_train[0:int(X_train.shape[0]/50)*50,:,:]
y_train = y_train[0:int(y_train.shape[0]/50)*50,:]
# convert y to one-hot, reshape x
#y_train = to_categorical(y_train)
#y_test = to_categorical(y_test)
#print(y_train[0])
X_train = X_train.astype('float32') / 255.
X_test = X_test.astype('float32') / 255.

X_train = X_train.reshape((len(X_train),img_rows,img_cols,img_chnls))
X_test = X_test.reshape((len(X_test),img_rows,img_cols,img_chnls))
print(X_test[0].shape)
# select optimizer
optim = 'adam'

# dimension of latent space (batch size by latent dim)
m = 50
n_z = 150

# dimension of input (and label)
n_x = X_train.shape[1]
n_y = y_train.shape[1]
# nubmer of epochs
n_epoch = 30


##  ENCODER ##

img = Input(shape=img_shape)

label = Input(shape=(1,),dtype='int32')
label_embedding = Flatten()(Embedding(num_classes,np.prod(img_shape))(label))
flat_img = Flatten()(img)
model_input = multiply([flat_img,label_embedding])
model = Reshape((img_rows,img_cols,1))(model_input)
model = Conv2D(32,(3,3),activation='relu',padding='same')(model)
model = MaxPooling2D((2,2),padding='same')(model)
model = Conv2D(16,(3,3),activation='relu',padding='same')(model)
model = MaxPooling2D((2,2),padding='same')(model)
model = Conv2D(16,(3,3),activation='relu',padding='same')(model)
encoded = MaxPooling2D((2,2),padding='same')(model)		

gene_shape = (int(encoded.shape[1]),int(encoded.shape[2]),int(encoded.shape[3]))
label_embed = Flatten()(Embedding(num_classes,np.prod(gene_shape))(label))
flat_gene = Flatten()(encoded)
model_input = multiply([flat_gene,label_embed])
model = Reshape(gene_shape)(model_input)
model = Conv2D(16,(3,3),activation='relu',padding='same')(model)
model = UpSampling2D((2,2))(model)
model = Conv2D(16,(3,3),activation='relu',padding='same')(model)
print(model.shape)
model = UpSampling2D((2,2))(model)
model = Conv2D(32,(3,3),activation='relu',padding='same')(model)
print(model.shape)
model = UpSampling2D((2,2))(model)
decoded = Conv2D(1,(3,3),name='zadnji_conv_layer',activation='sigmoid',padding='same')(model)
#decoded = Reshape((-1,int(decoded.shape[1]),int(decoded.shape[2])))(decoded)
print(decoded.shape)
autoencoder = Model([img, label],decoded)

autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
autoencoder.load_weights('conv_model_128_20.h5')
autoencoder.fit([X_train, y_train],
		 X_train,
		 batch_size=m,
		 shuffle=True,
		 epochs = n_epoch,
		 validation_data = ([X_test, y_test],X_test),
		 callbacks=[TensorBoard(log_dir='logs/conv/')])

autoencoder.save_weights('conv_model_128_60.h5')
pic_num = 0
variations = 3

veronika = Image.open('veronika_ref.jpg').convert('L')
veronika.load()
veronika = veronika.resize((img_rows,img_cols),Image.BILINEAR)
veronika = np.asarray(veronika,dtype="float32") / 255.
veronika = veronika.reshape((1,img_rows,img_cols,img_chnls))
veronika_age = np.eye(11)[2].reshape(1,-1)

tin  = 5
for i in range(0,10): 
	generated = autoencoder.predict([veronika,np.asarray([i])])
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
