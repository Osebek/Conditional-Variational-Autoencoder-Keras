from HandleData import load_data_id_critic, get_num_id_files,num_id
from networks_def_tf import squeezenet
from keras.losses import binary_crossentropy
from keras.optimizers import Adam,SGD
import numpy as np
EPOCHS = 100
print(num_id)
sgd = SGD(lr=0.001,decay=0.0002,momentum=0.9,nesterov=True)
squ = squeezenet(num_id,inshape=(224,224,3),output="prob")
squ.compile(loss='categorical_crossentropy',optimizer=sgd,metrics=['accuracy'])

n = get_num_id_files()
batch_size = 64
for e in range(EPOCHS):
	for i in range(n/batch_size):
		(x,y,_) = load_data_id_critic(i*batch_size,(i+1)*batch_size)
		loss  = squ.train_on_batch(x,y)
		print(np.nonzero(y[0]))
		print("Epoch: ",str(e)," , Iteration: ",str(i)," , Loss: ", str(loss))
	squ.save_weights("db_trained_squ.h5") 
