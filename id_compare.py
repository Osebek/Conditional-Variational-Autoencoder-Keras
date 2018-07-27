
def id_score(img1,img2,model):
	vector1 = model.predict(img1)
	vector2 = model.predict(img2)
	print(vector1)
	print(vector2)
