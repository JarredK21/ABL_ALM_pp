from scipy import interpolate
import numpy as np


i = 0
j = 0
rez = 1

threshold = 0.5

u = np.zeros(shape=(2,2))

fc = interpolate.interp1d([i*rez,i*rez + rez],[u[i,j],u[i+1,j]])
c_vector = [float(fc(threshold)), j*rez]

fa = interpolate.interp1d([i*rez,i*rez + rez],[u[i,j+1],u[i+1,j+1]])
a_vector = [float(fa(threshold)), j*rez+rez]

fd = interpolate.interp1d([j*rez,j*rez + rez],[u[i,j],u[i,j+1]])
d_vector = [i*rez, float(fd(threshold))]

fb = interpolate.interp1d([j*rez,j*rez + rez],[u[i+1,j],u[i+1,j+1]])
b_vector = [i*rez + rez, float(fb(threshold))]

print(c_vector,a_vector, d_vector, b_vector)