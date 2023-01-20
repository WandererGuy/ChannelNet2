
from scipy.io import loadmat
import numpy 
import pandas as pd 
test = loadmat(r'/home/manh/ChannelNet/Perfect_VehA.mat')
# print (test.keys())


data22 = loadmat(r'/home/manh/ChannelNet/Noisy_VehA_SNR_22.mat')
print (data22.keys())

x = data22['My_noisy_H'] # data 
# print (len (x)) --> 40000
# print (type(x))
# df = pd.DataFrame(x)
# df.csv(r'/home/manh/ChannelNet/data.csv')  #shape=(40000, 72, 14)

# print (x.shape)
# print (x[1,1,1])
# print (x[2,1,1])

# (40000, 72, 14)
# (0.052781646432488685+0.49190977572232725j)
# (-1.1254479591460862-0.6540447912313162j)
# (siso) manh@Manh-Vostro-5490:~/ChannelNet$ 

from ChannelNet_train import train_data, train_label, val_data, val_label
print (train_data.shape)
print (train_label.shape)
print (val_data.shape)
print (val_label.shape) 
