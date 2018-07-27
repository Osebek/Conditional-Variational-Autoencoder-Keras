def make_encoder():
	img = Input(img_shape)
	label = Input(shape=(1,),dtype='int32')
	label_embedding = Flatten()(Embedding(num_classes,np.prod(img_shape))(label))
	flat_img = Flatten()(img)
	model_input = multiply([flat_img,label_embedding])
	model = Reshape((img_rows,img_cols,img_chnls))(model_input)
	model = Conv2D(16,(3,3),activation='relu',padding='same')(model)
	model = MaxPooling2D((2,2),padding='same')(model)
	model = Conv2D(8,(3,3),activation='relu',padding='same')(model)
	model = MaxPooling2D((2,2),padding='same')(model)
	model = Conv2D(8,(3,3),activation='relu',padding='same')(model)
	encoded = MaxPooling2D((2,2),padding='same')(model)			

def make_decoder():
	gene_shape = (4,4,8)
	gene = Input(gene_shape)
	label = Input(shape=(1,),dtype='int32')
	label_embedding = Flatten()(Embedding(num_classes,np.prod(gene_shape)))(label)
	flat_gene = Flatten()(gene)

	model_input = multiply([flat_gene,label_embedding])
	model = Reshape(gene_shape)(model_input)
	model = Conv2D(8,(3,3),activation='relu',padding='same')(model)
	model = UpSampling2D((2,2))(model)
	model = Conv2D(8,(3,3),activation='relu',padding='same')(model)
	model = UpSampling2D((2,2))(model)
	model = Conv2D(16,(3,3),activation='relu',padding='same')(model)
	















