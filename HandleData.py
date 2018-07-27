import numpy as np 
import os 
from PIL import Image 
import os
import bz2
import pickle
from networks_def_tf import squeezenet,vgg_face
from random import randint
import klepto
import gc
import h5py
import scipy.io as sio
import datetime
import math 
from collections import Counter 
import pdb 

dbase = 'imdb'
img_rows,img_cols,img_chns = 224,224,3
NUM_PIC_ID = 100
FACE_SCORE_CUTOFF = 3.0 
files = {}
files_critic = {}	
num_id = 0 

def serializeData(data,name):
	outfile = open(name,'wb')
	pickle.dump(data,outfile)
	outfile.close()


def enumerate_files(folder):
	file_map = {}
	idx = 0 
	for f in os.listdir(folder):
		if os.path.isfile(folder + '/' + f) and len(str(f).split('_')) >= 4:
			file_map[idx] = f
			idx = idx+1
	return file_map

def enumerate_files_age(folder,age_string):
	file_map = {}
	idx = 0 
	for f in os.listdir(folder):
		if os.path.isfile(folder + '/' + f) and len(str(f).split('_')) >= 4 and f.startswith(age_string) and f[2] == '_': 
			file_map[idx] = f 
			idx = idx+1
	return file_map

utk_face_map = enumerate_files('UTKFace')
utk_face_map_age = enumerate_files_age('UTKFace','2')
data = sio.loadmat('imdb_db/imdb_crop/imdb.mat')
dob = data[dbase]['dob'][0][0][0]
dob1 = []
for x in dob:
	if x < 1000:
		x = 716588	
	dob1.append(datetime.datetime.fromordinal(int(x)) + datetime.timedelta(days=int(x)%1) - datetime.timedelta(days = 366)) 
#dob = map(lambda x: datetime.datetime.fromordinal(int(x)) + datetime.timedelta(days=int(x)%1) - datetime.timedelta(days = 366),dob)
dob = dob1
taken = data[dbase]['photo_taken'][0][0][0]
face_location = data[dbase]['face_location'][0][0][0]
second_face_score = data[dbase]['second_face_score'][0][0][0]
face_score = data[dbase]['face_score'][0][0][0]
name = data[dbase]['name'][0][0][0]
name = map(lambda x: x,name)
full_path = data[dbase]['full_path'][0][0][0]
full_path = map(lambda x: x[0], full_path)

name1 = map(lambda x: (x[0].encode('utf-8'),1) if len(x) == 1 else (x,0),name)

zipped = zip(full_path,taken,dob,name1,face_score,face_location,second_face_score)

zipped_filter = filter(lambda x: x[4] > FACE_SCORE_CUTOFF and x[3][1] == 1 and math.isnan(x[6]),zipped)


del zipped
del taken
del face_location
del face_score
del name
del full_path
gc.collect()

for (c,img_info) in enumerate(zipped_filter):
	taken = img_info[1]
	dob = img_info[2]
	files[c] = (img_info[0],taken-int(dob.year),img_info[3][0],img_info[5])



names = map(lambda x: x[1][2],files.items())
x = Counter(names).items()
multiple_id = set()
x = filter(lambda x: x[1] > NUM_PIC_ID, x)
for e in x:
	multiple_id.add(e[0])

name_id_map = {n: i for (i,n) in enumerate(multiple_id)}
num_id = len(name_id_map)
multiple_names = {}
c = 0
for i in files.items():
	name = i[1][2]
	if name in multiple_id:
		multiple_names[c] = (i[1][0],i[1][1],i[1][2],i[1][3],name_id_map[name])
		c=c+1


files_critic = multiple_names
perm_map = np.random.permutation(len(files_critic))

def load_part_srgan(folder,min_idx,max_idx,model):
	ageGroup = None
	imgs_train = []
	imgs_train_lr = []
	labels_train = []
	imgs_test = []
	labels_test = []
	id_vectors_train  = []
	id_vectors_test = []
	file_map = utk_face_map
	it = 0 
	for f in range(min_idx,max_idx):
		img = Image.open(folder+ '/' + file_map[f])
		img.load()
		img = img.resize((256,256),Image.BICUBIC)
		img_lr = img.resize((64,64),Image.BICUBIC)
		if model is not None:
			id_vector = model.predict(np.expand_dims(np.asarray(img),0))
		properties = str(file_map[f]).split('_')
		#img = img.resize((128,128),Image.ANTIALIAS)
		age = properties[0]
		data = np.asarray(img,dtype="int32")
		data_lr = np.asarray(img_lr,dtype="int32")
		it=it+1
		if ageGroup==None or ageGroup==min(int(age)/10,10):	
			imgs_train.append(data)
			imgs_train_lr.append(data_lr)
			if model is not None: 
				id_vectors_train.append(id_vector)
			labels_train.append(int(age))

	labels_train = map(lambda x: min(int(x / 10),10),labels_train)  
	labels_test =  map(lambda x: min(int(x / 10),10),labels_test)
	im_train = np.asarray(imgs_train)
	im_train.astype('float32') 
	im_train = im_train / 255.

	im_train_lr = np.asarray(imgs_train_lr)
	im_train_lr.astype('float32')
	im_train_lr = im_train_lr / 255. 

	lbl_train = np.asarray(labels_train,dtype=np.float32)
	if model is not None:
		id_train = np.asarray(id_vectors_train,dtype=np.float32)
		del id_vectors_train
	del imgs_train
	del labels_train
	gc.collect()
	if model is not None:
		return (im_train,im_train_lr,lbl_train,id_train)  
	else:
		return (im_train,im_train_lr,lbl_train,None)


def load_part(folder,min_idx,max_idx,model):
	ageGroup = None
	imgs_train = []
	labels_train = []
	imgs_test = []
	labels_test = []
	id_vectors_train  = []
	id_vectors_test = []
	file_map = utk_face_map
	it = 0 
	for f in range(min_idx,max_idx):
		img = Image.open(folder+ '/' + file_map[f])
		img.load()
		img = img.resize((224,224),Image.BICUBIC)
		if model is not None:
			id_vector = model.predict(np.expand_dims(np.asarray(img),0))
		properties = str(file_map[f]).split('_')
		#img = img.resize((128,128),Image.ANTIALIAS)
		age = properties[0]
		data = np.asarray(img,dtype="int32")
		it=it+1
		if ageGroup==None or ageGroup==min(int(age)/10,10):	
			imgs_train.append(data)
			if model is not None: 
				id_vectors_train.append(id_vector)
			labels_train.append(int(age))

	labels_train = map(lambda x: min(int(x / 10),10),labels_train)  
	labels_test =  map(lambda x: min(int(x / 10),10),labels_test)
	im_train = np.asarray(imgs_train)
	im_train.astype('float32') 
	im_train = im_train / 255.
	lbl_train = np.asarray(labels_train,dtype=np.float32)
	if model is not None:
		id_train = np.asarray(id_vectors_train,dtype=np.float32)
		del id_vectors_train
	del imgs_train
	del labels_train
	gc.collect()
	if model is not None:
		return (im_train,lbl_train,id_train)  
	else:
		return (im_train,lbl_train,None)



def load_part_dfc(folder,min_idx,max_idx,model):
	ageGroup = None
	imgs_train = []
	labels_train = []
	imgs_test = []
	labels_test = []
	id_vectors_train  = []
	id_vectors_test = []
	file_map = utk_face_map
	it = 0
	df = {} 

	for f in range(min_idx,max_idx):
		img = Image.open(folder+ '/' + file_map[f])
		img.load()
		img = img.resize((224,224),Image.BICUBIC)
		if model is not None:
			id_vector = model.predict(np.expand_dims(np.asarray(img),0))
			for (i,e) in enumerate(id_vector):
				if f == min_idx: 
					df[i] = e
				else: 
					df[i] = np.concatenate((df[i],e),axis=0)

		properties = str(file_map[f]).split('_')
		#img = img.resize((128,128),Image.ANTIALIAS)
		age = properties[0]
		data = np.asarray(img,dtype="int32")
		it=it+1
		if ageGroup==None or ageGroup==min(int(age)/10,10):	
			imgs_train.append(data)
			labels_train.append(int(age))

	labels_train = map(lambda x: min(int(x / 10),10),labels_train)  
	labels_test =  map(lambda x: min(int(x / 10),10),labels_test)
	im_train = np.asarray(imgs_train)
	im_train.astype('float32') 
	im_train = im_train / 255.
	id_vectors_train = df.items()
	print(len(id_vectors_train))
	lbl_train = np.asarray(labels_train,dtype=np.float32)
	del labels_train
	gc.collect()
	if model is not None:
		return (im_train,lbl_train,df)  
	else:
		return (im_train,lbl_train,None)


def load_part_dfc_age(folder,min_idx,max_idx,model):
	ageGroup = None
	imgs_train = []
	labels_train = []
	imgs_test = []
	labels_test = []
	id_vectors_train  = []
	id_vectors_test = []
	file_map = utk_face_map_age
	it = 0
	df = {} 

	for f in range(min_idx,max_idx):
		img = Image.open(folder+ '/' + file_map[f])
		img.load()
		img = img.resize((224,224),Image.BICUBIC)
		if model is not None:
			id_vector = model.predict(np.expand_dims(np.asarray(img),0))
			for (i,e) in enumerate(id_vector):
				if f == min_idx: 
					df[i] = e
				else: 
					df[i] = np.concatenate((df[i],e),axis=0)

		properties = str(file_map[f]).split('_')
		#img = img.resize((128,128),Image.ANTIALIAS)
		age = properties[0]
		data = np.asarray(img,dtype="int32")
		it=it+1
		if ageGroup==None or ageGroup==min(int(age)/10,10):	
			imgs_train.append(data)
			labels_train.append(int(age))

	labels_train = map(lambda x: min(int(x / 10),10),labels_train)  
	labels_test =  map(lambda x: min(int(x / 10),10),labels_test)
	im_train = np.asarray(imgs_train)
	im_train.astype('float32') 
	im_train = im_train / 255.
	id_vectors_train = df.items()
	print(len(id_vectors_train))
	lbl_train = np.asarray(labels_train,dtype=np.float32)
	del labels_train
	gc.collect()
	if model is not None:
		return (im_train,lbl_train,df)  
	else:
		return (im_train,lbl_train,None)

def load_part_age(folder,min_idx,max_idx):
	ageGroup = None
	imgs_train = []
	labels_train = []
	imgs_test = []
	labels_test = []
	id_vectors_train  = []
	id_vectors_test = []
	file_map = utk_face_map_age
	it = 0
	df = {} 

	for f in range(min_idx,max_idx):
		img = Image.open(folder+ '/' + file_map[f])
		img.load()
		img = img.resize((224,224),Image.BICUBIC)
		properties = str(file_map[f]).split('_')
		age = properties[0]
		data = np.asarray(img,dtype="int32")
		if ageGroup==None or ageGroup==min(int(age)/10,10):	
			imgs_train.append(data)
			labels_train.append(int(age))

	labels_train = map(lambda x: min(int(x / 10),10),labels_train)  
	labels_test =  map(lambda x: min(int(x / 10),10),labels_test)
	
	im_train = np.asarray(imgs_train)
	im_train.astype('float32') 
	im_train = im_train / 255.
	lbl_train = np.asarray(labels_train,dtype=np.float32)
	return (im_train,lbl_train)

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

def get_num_files():
	return len(files)

def get_num_id_files():
	return len(files_critic)

def  read_utk_face_id(folder,ageGroup=None,size=(224,224)): 
	imgs_train = []
	labels_train = []
	imgs_test = []
	labels_test = []
	# model  = vgg_face(
	model = squeezenet(50,inshape=(224,224,3),output="denseFeatures",
		simple_bypass=False,fire11_1024=True,input_override=None)
	id_vectors_train  = []
	id_vectors_test = []
	it = 0 
	for f in os.listdir(folder):
		if it > 12000:
			break
		if os.path.isfile(folder + '/' + f) and len(str(f).split('_')) >= 4:
			img = Image.open(folder+ '/' + f)
			img.load()
			img = img.resize((224,224),Image.BICUBIC)
			id_vector = model.predict(np.expand_dims(np.asarray(img),0))
			properties = str(f).split('_')
			#img = img.resize((128,128),Image.ANTIALIAS)
			age = properties[0]
			gender = properties[1]
			race = properties[2]
			date_time = properties[3].split('.')[0]
			data = np.asarray(img,dtype="int32")
			it=it+1
			if ageGroup==None or ageGroup==min(int(age)/10,10):	
				if randint(0,100) < 90: 
					imgs_train.append(data)
					id_vectors_train.append(id_vector)
					labels_train.append(int(age))
				else:
					imgs_test.append(data)
					id_vectors_test.append(id_vector)
					labels_test.append(int(age))
	
	labels_train = map(lambda x: min(int(x / 10),10),labels_train)  
	labels_test =  map(lambda x: min(int(x / 10),10),labels_test)
	del model 
	gc.collect()
	im_train = np.asarray(imgs_train)
	im_train.astype('float32') 

	del imgs_train
	gc.collect()
	im_train = im_train / 255.
	lbl_train = np.asarray(labels_train,dtype=np.float32)
	id_train = np.asarray(id_vectors_train,dtype=np.float32)
	im_test = np.asarray(imgs_test,dtype=np.float32) / 255.
	lbl_test = np.asarray(labels_test,dtype=np.float32)
	id_test = np.asarray(id_vectors_test,dtype=np.float32)
	h5f = h5py.File('serializedData/serialized_set_squ.h5','w')
	h5f.create_dataset('im_train',data=im_train)
	h5f.create_dataset('lbl_train',data=lbl_train)
	h5f.create_dataset('id_train',data=id_train)
	h5f.create_dataset('im_test',data=im_test)
	h5f.create_dataset('lbl_test',data=lbl_test)
	h5f.create_dataset('id_test',data=id_test)
	h5f.close()
	#(np.asarray(imgs_test),np.asarray(labels_test),np.asarray(id_vectors_test))


def load_data_imdb(xmin,xmax,model):
	ageGroup = None
	imgs_train = []
	id_vectors_train = []
	labels_train = []
	img_size = (224,224)
	for i in range(xmin,xmax):
		(filename,age,name,crop,name_id) = files_critic[i]
		#(filename,age,name,crop) = files[i]
		ccrop = tuple(crop[0])
		img = Image.open('imdb_db/aligned/' + filename)
		#img.show(command='fim')
		#width,height = img.size

		#x = (int(ccrop[1]),int(height-ccrop[2]),int(ccrop[3]),int(height-ccrop[0]))
		#img = img.crop(x)
		#img.load()
		#img.show(command='fim')
		img = img.resize(img_size,Image.BICUBIC)
		#id_vector = model.predict(np.expand_dims(np.asarray(img),0))

		data = np.asarray(img,dtype="int32")
		if data.shape == img_size:
			data =  np.stack((data,)*3, -1)
		if model is not None:
		
			id_vector = model.predict(np.expand_dims(data,0))
		else: 
			id_vector = [1,2,3]
		imgs_train.append(data)
		labels_train.append(min(int(age / 10),10))
		id_vectors_train.append(id_vector)

	im_train = np.asarray(imgs_train)
	im_train.astype('float32')
	im_train = im_train / 255.
	
	lbl_train = np.asarray(labels_train,dtype=np.float32)
	id_train  = np.asarray(id_vectors_train,dtype=np.float32)
	return (im_train,lbl_train,id_train)


def load_data_id_critic(xmin,xmax):
	ageGroup = None
	imgs_train = []
	labels_train = []
	labels_age = []
	img_shape = (224,224)
	for i in range(xmin,xmax):
		(filename,age,name,crop,name_id) = files_critic[perm_map[i]]
		ccrop = tuple(crop[0])
		img = Image.open('imdb_db/aligned/' + filename)
		img = img.resize(img_shape,Image.BICUBIC)

		data = np.asarray(img,dtype="int32")
		if data.shape == img_shape:
			data =  np.stack((data,)*3, -1)
		imgs_train.append(data)
		test_img = (data * 255 / np.max(data)).astype('uint8')
		test = Image.fromarray(test_img)
		#img.save('testt/img' + str(i) + '_' + str(age) + '_' + name + '.jpg')
		#print(name,',',name_id)
		labels_train.append(np.eye(len(name_id_map))[name_id])
		labels_age.append(age)
	im_train = np.asarray(imgs_train)
	im_train.astype('float32')
	im_train = im_train / 255.
	lbl_train = np.asarray(labels_train,dtype=np.float32)
	age_train = np.asarray(labels_age,dtype=np.int64)
	return (im_train,lbl_train,age_train)


#read_utk_face_id('UTKFace/i')
#load_data_id_critic(0,600)
#mdl = vgg_face(weights_path='TrainedNets/recognition-nets/weights/vgg_face_weights_tf.h5')
#(x,y,z) = load_part_dfc('UTKFace/',0,32,mdl)
