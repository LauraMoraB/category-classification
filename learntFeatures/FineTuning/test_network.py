import keras
from network_descriptor import create_model
import numpy as np
from scipy.misc import imresize
from PIL import Image
import os
import sys
from sklearn import svm
from keras.models import Sequential, Model
from sklearn.model_selection import GridSearchCV
import configparser
from utils import *

def read_model(IMG_SIZE, PATH_TO_MODEL):
    # Init model with right architecture
    model = create_model(IMG_SIZE, optimizer_param='sgd', depth='mobilenet_base')
    model.load_weights(PATH_TO_MODEL)
    model.compile(loss='categorical_crossentropy',
                optimizer='sgd',
                metrics=['accuracy'])
    return model
    
def test(path_info):

    path_config = path_info+'config.ini'
    path_model  = path_info+'model_cnn.h5'

    config = configparser.ConfigParser()
    config.read(path_config)
    img_size = int(config.get('DEFAULT','IMG_SIZE'))
    dataset_small = config.getboolean('DEFAULT','DATASET_SMALL')

    if dataset_small:
        dataset_dir = '/home/grupo01/dataset/MIT_split_small'
    else:
        dataset_dir = '/home/mcv/datasets/MIT_split'

    # import model
    model = read_model(img_size, path_model)

    # path tset complete de test, sense patch size
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
    return str(correct/count)
