
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
 


fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')

            
for k in range (3):
    for j in range (5):         
        for i in range (7):
            plot = ax.scatter(k, j, i, c = i , cmap='viridis', marker=".", norm = matplotlib.colors.Normalize(vmin=0, vmax=6, clip=False)) # s is marker size # norm = Normalize 



plt.colorbar(plot) # mappable was found to use for colorbar creation

ax.set_title('3D line plot geeks for geeks')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()