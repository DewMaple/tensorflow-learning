from __future__ import print_function

from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Flatten, Dropout, Activation, Permute
from keras.models import Sequential

K.set_image_data_format('channels_last')  # WARNING : important for images and tensors dimensions ordering


def convblock(cdim, nb, bits=3):
    L = []

    for k in range(1, bits + 1):
        convname = 'conv' + str(nb) + '_' + str(k)
        # L.append( Convolution2D(cdim, 3, 3, border_mode='same', activation='relu', name=convname) ) # Keras 1
        L.append(Convolution2D(cdim, kernel_size=(3, 3), padding='same', activation='relu', name=convname))  # Keras 2

    L.append(MaxPooling2D((2, 2), strides=(2, 2)))

    return L


def vgg_face_blank():
    withDO = True  # no effect during evaluation but usefull for fine-tuning

    if True:
        mdl = Sequential()

        # First layer is a dummy-permutation = Identity to specify input shape
        mdl.add(Permute((1, 2, 3), input_shape=(224, 224, 3)))  # WARNING : 0 is the sample dim

        for l in convblock(64, 1, bits=2):
            mdl.add(l)

        for l in convblock(128, 2, bits=2):
            mdl.add(l)

        for l in convblock(256, 3, bits=3):
            mdl.add(l)

        for l in convblock(512, 4, bits=3):
            mdl.add(l)

        for l in convblock(512, 5, bits=3):
            mdl.add(l)

        # mdl.add( Convolution2D(4096, 7, 7, activation='relu', name='fc6') ) # Keras 1
        mdl.add(Convolution2D(4096, kernel_size=(7, 7), activation='relu', name='fc6'))  # Keras 2
        if withDO:
            mdl.add(Dropout(0.5))
        # mdl.add( Convolution2D(4096, 1, 1, activation='relu', name='fc7') ) # Keras 1
        mdl.add(Convolution2D(4096, kernel_size=(1, 1), activation='relu', name='fc7'))  # Keras 2
        if withDO:
            mdl.add(Dropout(0.5))
        # mdl.add( Convolution2D(2622, 1, 1, name='fc8') ) # Keras 1
        mdl.add(Convolution2D(2622, kernel_size=(1, 1), activation='relu', name='fc8'))  # Keras 2
        mdl.add(Flatten())
        mdl.add(Activation('softmax'))

        return mdl

    else:
        # See following link for a version based on Keras functional API :
        # gist.github.com/EncodeTS/6bbe8cb8bebad7a672f0d872561782d9
        raise ValueError('not implemented')


facemodel = vgg_face_blank()
facemodel.summary()
