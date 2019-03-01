from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from keras.models import Sequential
from keras.layers import Flatten, Dense, Reshape, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
import configparser
import sys

from utils import *
from test_network import test
from load_data import data_augmentation
from network_descriptor import create_model


def getXYfromGenerator(generator):
    """
    Input: ImageDataGenerator
    Output: X data, Y labels
    """    
    X, Y = generator.next()
    batch_index = 1

    while batch_index <= generator.batch_index:
        auxX, auxY = generator.next()
        X = np.concatenate((X, auxX))
        Y = np.concatenate((Y, auxY))
        batch_index = batch_index + 1

    return X, Y


##### SET VARIABLES ######
config = configparser.ConfigParser()
config.read('config_hyperparameter.ini')

IMG_SIZE    = int(config.get('DEFAULT','IMG_SIZE'))
BATCH_SIZE  = int(config.get('DEFAULT','BATCH_SIZE'))
ARCHITECTURE = config.get('DEFAULT','NET_ARCHITECTURE')
DATASET = config.get('DEFAULT','PATH_DATASET')
CATEGORIES = config.get('DEFAULT','CLASSES')
RANDOM = config.getboolean('DEFAULT', 'RANDOM_SEARCH')
DATA_AUG = config.get('DEFAULT','DATA_AUG')

dataset_train = DATASET+'train/'
dataset_test = DATASET+'test/'


####  LOAD DATA #######
train_datagen = data_augmentation(DATA_AUG)
test_datagen = data_augmentation('test') # sempre a test

train_generator =  compute_data_generator(dataset_train, train_datagen, IMG_SIZE, BATCH_SIZE, CATEGORIES)
validation_generator =  compute_data_generator(dataset_test, test_datagen, IMG_SIZE, BATCH_SIZE, CATEGORIES)

        
print('Building model...\n')

model = KerasClassifier(build_fn=create_model)

print ('Define Variables... \n')

lr = [0.01, 0.1, 0.3, 0.5, 1]
momentum = [0, 0.25, 0.5, 0.75, 1]
optimizer_param = ['sgd', 'adam', 'RMSprop', 'adadelta']

param_grid = dict(lr=lr, momentum=momentum, optimizer_param=optimizer_param, depth=ARCHITECTURE, IMG_SIZE=IMG_SIZE)

print ("Start Search.....")

if RANDOM:
    searcher = RandomizedSearchCV(model, param_grid, n_jobs=1)
else:
    searcher = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=2)

X, Y = getXYfromGenerator(train_generator)
result = searcher.fit(X,Y)

print("Best: %f using %s" % (result.best_score_, result.best_params_))
means = result.cv_results_['mean_test_score']
stds = result.cv_results_['std_test_score']
params = result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
print ("Done!")