import numpy as np
from scipy.io import loadmat

from l_vgg_face.model import vgg_face_blank


def load_weight(from_vlfeat=True):
    l = None
    if from_vlfeat:  # INFO : use this if you downloaded weights from vlfeat.org
        data = loadmat('vgg-face.mat', matlab_compatible=False, struct_as_record=False)
        l = data['layers']
        description = data['meta'][0, 0].classes[0, 0].description
    else:  # INFO : use this if you downloaded weights from robots.ox.ac.uk
        data = loadmat('vgg_face_matconvnet/data/vgg_face.mat', matlab_compatible=False, struct_as_record=False)
        net = data['net'][0, 0]
        l = net.layers
        description = net.classes[0, 0].description
    return l, description


def weight_compare(kmodel):
    kerasnames = [lr.name for lr in kmodel.layers]

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    # prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0, 1, 2, 3)  # INFO : for 'channels_last' setting of 'image_data_format'
    l, description = load_weight()

    for i in range(l.shape[1]):
        matname = l[0, i][0, 0].name[0]
        mattype = l[0, i][0, 0].type[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            print(matname, mattype)
            print(l[0, i][0, 0].weights[0, 0].transpose(prmt).shape, l[0, i][0, 0].weights[0, 1].shape)
            print(kmodel.layers[kindex].get_weights()[0].shape, kmodel.layers[kindex].get_weights()[1].shape)
            print('------------------------------------------')
        else:
            print('MISSING : ', matname, mattype)
            print('------------------------------------------')


def copy_mat_to_keras(kmodel):
    kerasnames = [lr.name for lr in kmodel.layers]

    # WARNING : important setting as 2 of the 4 axis have same size dimension
    # prmt = (3,2,0,1) # INFO : for 'th' setting of 'dim_ordering'
    prmt = (0, 1, 2, 3)  # INFO : for 'channels_last' setting of 'image_data_format'
    l, description = load_weight()
    for i in range(l.shape[1]):
        matname = l[0, i][0, 0].name[0]
        if matname in kerasnames:
            kindex = kerasnames.index(matname)
            # print matname
            l_weights = l[0, i][0, 0].weights[0, 0]
            l_bias = l[0, i][0, 0].weights[0, 1]
            f_l_weights = l_weights.transpose(prmt)
            # f_l_weights = np.flip(f_l_weights, 2) # INFO : for 'th' setting in dim_ordering
            # f_l_weights = np.flip(f_l_weights, 3) # INFO : for 'th' setting in dim_ordering
            assert (f_l_weights.shape == kmodel.layers[kindex].get_weights()[0].shape)
            assert (l_bias.shape[1] == 1)
            assert (l_bias[:, 0].shape == kmodel.layers[kindex].get_weights()[1].shape)
            assert (len(kmodel.layers[kindex].get_weights()) == 2)
            kmodel.layers[kindex].set_weights([f_l_weights, l_bias[:, 0]])
            # print '------------------------------------------'


def pred(kmodel, crpimg, transform=False):
    # transform=True seems more robust but I think the RGB channels are not in right order

    imarr = np.array(crpimg).astype(np.float32)

    if transform:
        imarr[:, :, 0] -= 129.1863
        imarr[:, :, 1] -= 104.7624
        imarr[:, :, 2] -= 93.5940
        #
        # WARNING : in this script (https://github.com/rcmalli/keras-vggface) colours are switched
        aux = copy.copy(imarr)
        # imarr[:, :, 0] = aux[:, :, 2]
        # imarr[:, :, 2] = aux[:, :, 0]

        # imarr[:,:,0] -= 129.1863
        # imarr[:,:,1] -= 104.7624
        # imarr[:,:,2] -= 93.5940

    # imarr = imarr.transpose((2,0,1)) # INFO : for 'th' setting of 'dim_ordering'
    imarr = np.expand_dims(imarr, axis=0)

    out = kmodel.predict(imarr)

    best_index = np.argmax(out, axis=1)[0]
    best_name = description[best_index, 0]
    print(best_index, best_name[0], out[0, best_index], [np.min(out), np.max(out)])


facemodel = vgg_face_blank()
# weight_compare(facemodel)
copy_mat_to_keras(facemodel)
