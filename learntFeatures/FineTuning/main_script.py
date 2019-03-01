import os
import getpass

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import plot_model
from keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.misc import imresize
import time
import configparser
import sys

from network_descriptor import *
from utils import *
from test_network import test
from load_data import data_augmentation


#### IMPORT CONFIGURATION ####
RESULTS_DIR = sys.argv[1]+'/'
if not os.path.exists(RESULTS_DIR):
    os.makedirs(RESULTS_DIR)

config = configparser.ConfigParser()
config.read('config.ini')

MODEL_FNAME = RESULTS_DIR+'model_cnn.h5'
IMG_SIZE    = int(config.get('DEFAULT','IMG_SIZE'))
BATCH_SIZE  = int(config.get('DEFAULT','BATCH_SIZE'))
ACTIVATION_FUNCTION1 = config.get('DEFAULT','ACTIVATION_FUNCTION1')
ACTIVATION_FUNCTION2 = config.get('DEFAULT','ACTIVATION_FUNCTION2')
OPTIMIZER = config.get('DEFAULT','OPTIMIZER')
MOMENTUM =int(config.get('DEFAULT','MOMENTUM'))
LR = int(config.get('DEFAULT','LR'))
DENSITY = config.get('DEFAULT','DENSITY')
FREEZE = int(config.get('DEFAULT','FREEZE'))

dataset_dir = config.get('DEFAULT','PATH_DATASET')
CATEGORIES = config.get('DEFAULT','CLASSES')
DATA_AUG = config.get('DEFAULT','DATA_AUG')


if not os.path.exists(dataset_dir):
  print(Color.RED, 'ERROR: dataset directory '+dataset_dir+' do not exists!\n')
  quit()


##### LOAD DATA ########

train_datagen = data_augmentation(DATA_AUG)
test_datagen = data_augmentation('test')


train_generator = train_datagen.flow_from_directory(
        dataset_dir+'/train',  
        target_size=(IMG_SIZE, IMG_SIZE),  
        batch_size=BATCH_SIZE,
        classes = CATEGORIES,
        class_mode='categorical')  

validation_generator = test_datagen.flow_from_directory(
        dataset_dir+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = CATEGORIES,
        class_mode='categorical')


##### BUILD MODEL ######
print('Building model...\n')

model = create_model(LR, MOMENTUM, optimizer_param=OPTIMIZER, depth=DENSITY, freeze_layer=FREEZE, img_size=IMG_SIZE)

print(model.summary())
plot_model(model, to_file=RESULTS_DIR+'modelMLP.png', show_shapes=True, show_layer_names=True)
print('Done!\n')
if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')


###### TRAINING ######
print('Start training...\n')

# Callbacks
filepath = RESULTS_DIR+"weights.best.h5"
callback = model_create_callback(filepath, call='default')

history = model.fit_generator(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE,
        callbacks = callback)

print('Done!\n')
print('Saving the model into '+MODEL_FNAME+' \n')
model.save_weights(MODEL_FNAME) 
print('Done!\n')

plot_train(RESULTS_DIR, history)

############ TEST ###############

model = create_model(LR, MOMENTUM, optimizer_param=OPTIMIZER, depth=DENSITY, 
                      freeze_layer=FREEZE, img_size=IMG_SIZE, test=True, path=MODEL_FNAME )

directory = dataset_dir+'/test'

classes = {'coast':0,'forest':1,'highway':2,'inside_city':3,'mountain':4,'Opencountry':5,'street':6,'tallbuilding':7} 
correct = 0.
count   = 0

for class_dir in os.listdir(directory):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory,class_dir)):
        im = Image.open(os.path.join(directory,class_dir,imname))
        image = np.expand_dims(imresize(im, (img_size, img_size, 3)), axis=0)
        predicted_cls = model.predict(image/255.)
        predicted_cls = np.argmax( softmax(np.mean(predicted_cls,axis=0)) )
        if predicted_cls == cls:
            correct+=1
        count += 1
print ("Images: ", count, "\nCorrect: ", correct)
        
colorprint(Color.BLUE, 'Done!\n')
colorprint(Color.GREEN, 'Test Acc. = '+str(correct/count)+'\n')

print ("All Done ;)")
