from HandleData import load_data_id_critic, get_num_id_files,num_id
from networks_def_tf import InceptionV3
from keras.losses import binary_crossentropy
from keras.optimizers import Adam,SGD
import numpy as np

#1430416
EPOCHS = 100
print(num_id)
iv3 = InceptionV3(N_classes=num_id)

n = get_num_id_files()
batch_size = 32
for e in range(EPOCHS):
	for i in range(n/batch_size):
		(x,y,_) = load_data_id_critic(i*batch_size,(i+1)*batch_size)
		loss  = iv3.train_on_batch(x,y)
		print("Epoch: ",str(e)," , Iteration: ",str(i)," , Loss: ", str(loss))
	iv3.save_weights("db_trained_iv3_2.h5") 

