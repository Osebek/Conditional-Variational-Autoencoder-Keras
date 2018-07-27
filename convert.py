from keras import backend as K
import tensorflow as tf

from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard #LearningRateScheduler, BaseLogger
from keras.models import load_model, Model
from networks_def_tf import vgg_face,VGG_16,VGG_19

if __name__ == "__main__":

    h,w,ch= (224,224,3)

    K.set_image_dim_ordering('th')
    model.load_weights('vgg-16/vgg16_weights.h5') 
    print("Made the model")
    
    #model = squeezenet(50, output="denseFeatures",
    #                   simple_bypass=True, fire11_1024=True)

    #model.load_weights("weights/luksface-weights.h5")
    #model.load_weights('my_we16_weightsh5')

    from keras import backend as K
    from keras.utils.conv_utils import convert_kernel
    import tensorflow as tf
    ops = []
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.W, converted_w).op)

    K.get_session().run(ops)
    model.save_weights('vgg-16/vgg19_weights_tf.h5')
