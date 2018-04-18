import numpy as np
from keras.models import Model
from scipy.spatial.distance import cosine as dcos

from l_vgg_face.model import vgg_face_blank

face_model = vgg_face_blank()
face_feature = Model(input=face_model.layers[0].input, output=face_model.layers[-2].output)


def extract_feature(im):
    im_arr = np.array(im).astype(np.float32)
    im_arr = np.expand_dims(im_arr, axis=0)
    feature_vector = face_feature.predict(im_arr)[0, :]
    normfvec = np.math.sqrt(feature_vector.dot(feature_vector))
    return feature_vector / normfvec


def cos_distance(feature_vector1, feature_vector2):
    return dcos(feature_vector1, feature_vector2)


def calculate_distance(im1, im2):
    fvec1 = extract_feature(im1)
    fvec2 = extract_feature(im2)
    dcos_1_2 = dcos(fvec1, fvec2)
    print('----------: {}'.format(dcos_1_2))
    return dcos_1_2
