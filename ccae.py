from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,Flatten,multiply,Embedding, Reshape,concatenate
from keras.models import Model
from keras.optimizers import adam
from keras import backend as K
from HandleData import load_part,load_part_dfc
from PIL import Image
from networks_def_tf import vgg_face
import numpy as np


def dfc_loss(y_true,y_pred):
	y_true = K.flatten(y_true)
	y_pred = K.flatten(y_pred)
	return K.mean(K.square(y_true-y_pred),axis=-1)	

dfc_mode = 'vae_123'
num_classes = 11
batch_size = 16
num_data = 22000
epochs = 30
img_rows,img_cols,img_chns = 224,224,3
original_img_size = (img_rows,img_cols,img_chns)
nf = (1.0/(2.0*51380224.0))
filters = 64

input_img = Input(shape=(224, 224, 3)) 
input_label = Input(shape=(1,)) # adapt this if using `channels_first` image data format
x = Conv2D(filters*2, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
#encoded = MaxPooling2D((2, 2), padding='same')(x)

x = Flatten()(x)
encoded = Dense(53)(x)

encoded = concatenate([encoded,input_label])

decoder_upsamp = Dense (filters* img_rows/8* img_cols/8,activation='relu')(encoded)
encoded = Reshape((img_rows/8,img_cols/8,filters))(decoder_upsamp)
# at this point the representation is (4, 4, 8) i.e. 128-dimensional
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(filters*2, (3, 3), activation='relu',padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same',name='img_out')(x)

autoencoder = Model([input_img,input_label], decoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

squ =  vgg_face(weights_path = 'TrainedNets/recognition-nets/weights/vgg_face_weights_tf.h5',output_layer=dfc_mode)
squ.trainable = False

end2end = squ(decoded)
end2end.append(decoded)
e2emodel = Model(inputs=[input_img,input_label], outputs=end2end)

losses = {'img_out': 'binary_crossentropy','vgg-face': dfc_loss }

if dfc_mode == 'vae_whole':
	dfc_w = [x*0.5 for x in [nf,nf,nf/2,nf/2,nf/4,nf/4,nf/4,nf/8,nf/8,nf/8,nf/32,nf/32,nf/32]] 
elif dfc_mode == 'vae_123':
	dfc_w = [x*0.5 for x in [nf,nf/2,nf/4]]
elif dfc_mode == 'vae_345':
	dfc_w =[x*0.5 for x in [nf/4,nf/8,nf/32]]

optm = adam(lr=0.005)

e2emodel.compile(optimizer=optm,loss=losses,loss_weights=dfc_w+[1])

for e in range(epochs):
	for i in range(0,num_data/batch_size):
		(x_train,y_train,df) = load_part_dfc('UTKFace',i*batch_size,(i+1)*batch_size,squ)
#		loss = autoencoder.train_on_batch([x_train,y_train],x_train)		
		loss = e2emodel.train_on_batch([x_train,y_train],[df[0],df[1],df[2],x_train])
		print('Epoch: ',e,' Batch: ',i,' Loss: ', str(loss))
		if i % 50 == 0:
			ref_img = Image.open('imgs/ref.jpg')
			ref_img.load()
			ref_img = ref_img.resize((img_rows,img_cols),Image.BILINEAR)
			ref_img = np.asarray(ref_img,dtype="float32") / 255. 

			ref_img = ref_img.reshape(1,img_rows,img_cols,img_chns)
			for m in range(0,num_classes):
				generated = autoencoder.predict([ref_img,np.asarray(m,dtype='int32').reshape(1,1)])
				formatted = (generated * 255 / np.max(generated)).astype('uint8')
				img = Image.fromarray(formatted[0][:,:,:])
				img.save('imgs/autoencoder_' + str(m) + '.jpg')
	autoencoder.save_weights('saved_models/autoencoder_dfc_3.h5')
