import numpy as np
import matplotlib.pyplot as plt

y = 5
x = 5
U = np.ones(x*y)

ys = np.linspace(0,4,y)
zs = np.linspace(0,4,x)
X,Y = np.meshgrid(ys,zs)

#u_plane = np.array([[-6],[-4],[-2],[-0.7],[1]])

u_plane = np.array([[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],
                   [-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]])

thresholds = np.arange(-12.0,0.0,2)
thresholds = np.append(thresholds,-0.7)
markers = ["o","v","^","s","x","D","*"]

fig = plt.figure()
plt.xlim([ys[0],ys[-1]])
plt.ylim((zs[0],zs[-1]))

for t in np.arange(0,len(thresholds)):
    storage = np.zeros(len(ys))
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)-1):
            
            print(u_plane[j,k+1], thresholds[t])
            if u_plane[j,k+1] > thresholds[t]:
                
                storage[j] = zs[int(k)]

                break

    plt.plot(ys,storage,linestyle="-",marker=markers[t])


plt.xlabel("y axis [m]")
plt.ylabel("z axis [m]")
plt.legend(thresholds)
plt.show()