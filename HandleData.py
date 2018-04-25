import numpy as np 
import os 
from PIL import Image 
import os 
from random import randint
def  read_utk_face(folder,ageGroup=None,size=(200,200)): 
	imgs_train = []
	labels_train = []
	imgs_test = []
	labels_test = []
	
	for f in os.listdir(folder):
		if os.path.isfile(folder + '/' + f) and len(str(f).split('_')) >= 4:
			img = Image.open(folder + '/' + f).convert('L')
			img.load()
			img = img.resize(size,Image.BILINEAR)
			properties = str(f).split('_')
			#img = img.resize((128,128),Image.ANTIALIAS)
			age = properties[0]
			gender = properties[1]
			race = properties[2]
			date_time = properties[3].split('.')[0]
			data = np.asarray(img,dtype="int32")
			if ageGroup==None or ageGroup==min(int(age)/10,10):	
				if randint(0,100) < 90: 
					imgs_train.append(data)
					labels_train.append(int(age))
				else:
					imgs_test.append(data)
					labels_test.append(int(age))

	labels_train = map(lambda x: min(int(x / 10),10),labels_train)  
	labels_test =  map(lambda x: min(int(x / 10),10),labels_test) 
	return (np.asarray(imgs_train),np.asarray(labels_train)),(np.asarray(imgs_test),np.asarray(labels_test))


