from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Dropout, BatchNormalization, GlobalAveragePooling2D
import keras
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
import time
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.preprocessing import image
import numpy as np
from keras import layers

def prepare_image(file):
    img_path = ''
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)


def create_model( lr, momentum, optimizer_param='sgd', depth = 'mobilenet_base', freeze_layer=1, img_size=224, test=False, path="none") :
#### INIT MODEL ##########       
    base_model = keras.applications.mobilenet.MobileNet(classes=8, include_top=False, input_shape=(img_size, img_size, 3))
    x=base_model.output

    if depth == 'mobilenet_base':
        x=GlobalAveragePooling2D()(x)
        x=Dense(1024,activation='relu')(x) 

    elif depth == 'comb1':
        x=Flatten()(x)
        x=Dense(1024, kernel_initializer='random_uniform', activation='relu')(x)
        x=Dense(512, kernel_initializer='random_uniform', activation='relu')(x)

    elif depth == 'comb2':
        x=Flatten()(x)
        x=Dropout(0.5)(x)
        x=Dense(1024, kernel_initializer='random_uniform', activation='relu')(x)
       
    elif depth == 'comb3':
        x=Flatten()(x)
        x=Dropout(0.5)(x)
        x=Dense(1024, kernel_initializer='random_uniform', activation='relu')(x)
        x=Dropout(0.5)(x)
        x=Dense(512, kernel_initializer='random_uniform', activation='relu')(x)
       
#### END MODEL ######
    preds = Dense(8, activation='softmax')(x) 
    model = Model(inputs=base_model.input, outputs=preds)

    if test:
        model.load_weights( MODEL_FNAME )
    else:
        for layer in model.layers[:-freeze_layer]: 
            layer.trainable = False

    
    optimizer_param = init_optimizer(optimizer_param, lr, momentum)
    model.compile(loss='categorical_crossentropy',
                optimizer=optimizer_param,
                metrics=['accuracy'])    

    return model

def init_optimizer(optimizer_param, lr, momentum):
    if optimizer_param == "adam":
        optimizer_param = keras.optimizers.Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

    elif optimizer_param == "adadelta":
        optimizer_param = keras.optimizers.Adadelta(lr=lr, rho=0.95, epsilon=None, decay=0.0)

    elif optimizer_param == "sgd":
        optimizer_param = keras.optimizers.SGD(lr=lr, momentum=momentum)

    elif optimizer_param == "RMSprop":
        optimizer_param = keras.optimizers.RMSprop(lr=lr, rho=0.95, epsilon=None, decay=0.0)

    else:
        print ("Bad Request")

    return optimizer_param

def model_create_callback(filepath, call='default'):
   callback = [] 
   if call == 'default':
        callback = [
            EarlyStopping(monitor='val_loss', min_delta=0,patience=0, verbose=0, mode='auto'),
            ModelCheckpoint(filepath,monitor='val_acc', mode='max', save_best_only=True, save_weights_only=True)
        ]

   return callback
