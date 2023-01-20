
'''
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
    
    
    ####### ------ training SRCNN ------ #######
    idx_random = numpy.random.rand(len(perfect_image)) < (1/9)  # uses 32000 from 36000 as training and the rest as validation
    train_data, train_label = interp_noisy[idx_random,:,:,:] , perfect_image[idx_random,:,:,:]
    val_data, val_label = interp_noisy[~idx_random,:,:,:] , perfect_image[~idx_random,:,:,:]    

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
 


fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')

z = np.linspace (1,2,10)
x = np.linspace (1,3,10)
y = np.linspace (1,4,10)
 
# defining all 3 axis

# plotting

# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:

plot = ax.scatter(x, y, z, c = z, cmap='viridis', marker='o') # s is marker size # norm = Normalize 

plt.colorbar(plot) # mappable was found to use for colorbar creation

ax.set_title('3D line plot geeks for geeks')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
