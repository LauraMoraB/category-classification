from keras.preprocessing.image import ImageDataGenerator

def data_augmentation(config='default'):

    if config == 'default':
        generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True)

    elif config == 'feature_std':
        ## standarizes images
        generator = ImageDataGenerator(
        rescale=1./255,
        featurewise_center=True, 
        featurewise_std_normalization=True)

    elif config == 'whitening':
        # reduces redundancy
        generator = ImageDataGenerator(
        rescale=1./255,
        zca_whitening=True)

    elif config == 'rotations':
        # maybe, no need to rotate more data
        generator = ImageDataGenerator(
        rescale=1./255,
        rotation_range=70)

    elif config == 'zoom':
        generator = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.3)

    elif config == 'intensity':
        generator = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2)

    elif config == 'combi1':
        generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        featurewise_center=True, 
        featurewise_std_normalization=True)

    elif config == 'combi2':
        generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,
        featurewise_center=True, 
        featurewise_std_normalization=True,
        zca_whitening=True)

    elif config == 'combi3':
        generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,rotation_range=70,
        zca_whitening=True)

    elif config == 'combi4':
        generator = ImageDataGenerator(
        rescale=1./255,
        horizontal_flip=True,featurewise_center=True, 
        featurewise_std_normalization=True,
        zca_whitening=True, shear_range=0.2, zoom_range=0.3)

    elif config == 'test': 
        generator = ImageDataGenerator(
        rescale=1./255)
    else:
        print ("Bad Request")

    return generator