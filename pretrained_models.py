from keras.applications.resnet50 import ResNet50 
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
import numpy as np 
from keras.preprocessing import image 
from keras.applications.vgg16 import preprocess_input
from scipy.spatial import distance 
model = VGG16(weights='imagenet',include_top=False)


def compare_imgs(img1,img2):
	print(img1)
	x1 = np.expand_dims(img1,axis=0)
	x1 = preprocess_input(x1)

	x2 = np.expand_dims(img2,axis=0)
	x2 = preprocess_input(x2)

	features1 = model.predict(x1)
	features2 = model.predict(x2)
	return distance.cosine(features1.flatten(),features2.flatten())

