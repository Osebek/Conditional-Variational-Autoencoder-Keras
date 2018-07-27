import numpy as np 
import os 
import cv2
from imageio import imread  
from scipy.spatial import distance
from keras.models import load_model
from PIL import Image
import importlib
from inception_resnet_v1 import InceptionResNetV1
cascade_path = '../keras-facenet/model/cv2/haarcascade_frontalface_alt2.xml'
model_path = 'model/keras/facenet_keras.h5'
weights_path = '../keras-facenet/model/keras/facenet_keras_weights.h5'
image_size = 128 

model = InceptionResNetV1(weights_path=weights_path)


def prewhiten(x):
	if x.ndim == 4:
		axis = (1,2,3)
		size = x[0].size
	elif x.ndim == 3:
		axis = (0,1,2)
		size = x.size
	else:
		raise ValueError('Dimension should be 3 or 4')
	
	mean = np.mean(x,axis=axis,keepdims=True)
	std = np.std(x,axis=axis,keepdims=True)
	std_adj = np.maximum(std,1.0/np.sqrt(size))
	y = (x-mean) / std_adj
	return y 


def align_images(img,margin):
	cascade = cv2.CascadeClassifier(cascade_path)
	faces = cascade.detectMultiScale(img,scaleFactor=1.1,minNeighbors=3)
	print(faces)
	(x,y,w,h) = faces[0]
	cropped = img[y-margin//2:y+h+margin//2,
		      x-margin//2:x+h+margin//2,:] 
	aligned = Image.resize(cropped,
			       (image_size,image_size),
			       mode='reflect')
	return aligned 


def calc_emb(img,margin=10):
	img_processed = prewhiten(align_images(img,margin))
	model.predict(img_processed)

def calc_emb_no_align(img):
	model.predict(img)


def load_preprocess_img(filename):
	image = Image.open(filename).convert('L')
	image.load()
	image = image.resize((160,160))
	stacked_img = np.stack((image,)*3,-1)
	return stacked_img	

def preprocess_generated(img):
	img = img.resize((160,160))
	stacked_img = np.stack((img,)*3,-1)
	return stacked_img 

def calculate_distance(filename1,filename2):
	img1 = load_preprocess_img(filename1)
	img2 = load_preprocess_img(filename2)
	imgs = [img1,img2]
	imgs = np.asarray(imgs)
	comparison = model.predict_on_batch(imgs)
	dst = distance.euclidean(comparison[0],comparison[1])
	return dst

def calculated_distance_mixed(filename1,generated_img):
	ref_img = load_preprocess_img(filename1)
	gen = preprocess_generated(generated_img)
	imgs = [ref_img,gen]
	imgs = np.asarray(imgs)
	comparison = model.predict_on_batch(imgs)
	dst = distance.euclidean(comparison[0],comparison[1])
	return dst

