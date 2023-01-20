import numpy as np
import numpy 
import math
from models import interpolation , SRCNN_train , SRCNN_model, SRCNN_predict , DNCNN_train , DNCNN_model , DNCNN_predict
#from scipy.misc import imresize
from scipy.io import loadmat
import matplotlib.pyplot as plt



if __name__ == "__main__":
    # load datasets 
    channel_model = "VehA"
    SNR = 22
    Number_of_pilots = 48
    num_pilots = Number_of_pilots
    perfect = loadmat("Perfect_"+ channel_model +".mat")['My_perfect_H']
    noisy_input = loadmat("Noisy_" + channel_model + "_" + "SNR_" + str(SNR) + ".mat") ['My_noisy_H']
    # [channel_model+"_noisy_"+ str(SNR)]             
                      
    interp_noisy = interpolation(noisy_input , SNR , Number_of_pilots , 'rbf')

    perfect_image = numpy.zeros((len(perfect),72,14,2))
    perfect_image[:,:,:,0] = numpy.real(perfect)
    perfect_image[:,:,:,1] = numpy.imag(perfect)
    perfect_image = numpy.concatenate((perfect_image[:,:,:,0], perfect_image[:,:,:,1]), axis=0).reshape(2*len(perfect), 72, 14, 1)
    # print (perfect_image.shape)->(80000, 72, 14, 1)
    
    ####### ------ training SRCNN ------ #######
    idx_random = numpy.random.rand(len(perfect_image)) < (1/9)  # uses 32000 from 36000 as training and the rest as validation
    
    # x= perfect_image[idx_random,:,:,:].shape
    # print (x) (8784, 72, 14, 1)
    train_data, train_label = interp_noisy[idx_random,:,:,:] , perfect_image[idx_random,:,:,:]
    val_data, val_label = interp_noisy[~idx_random,:,:,:] , perfect_image[~idx_random,:,:,:]    


# print (train_data.shape)        
# print (train_label.shape)
# print (val_data.shape)
# print (val_label.shape) 

# (8970, 72, 14, 1)
# (8970, 72, 14, 1)
# (71030, 72, 14, 1)
# (71030, 72, 14, 1)