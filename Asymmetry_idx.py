import matplotlib.pyplot as plt
import numpy as np
import glob 
from netCDF4 import Dataset
from scipy import interpolate as interp

case = "Ex1"

sampling = glob.glob("{0}/post_processing/sampling*".format(case))
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_rotor"]

time = a.variables["time"]

#create R,theta space over rotor
R = np.linspace(0,63,500)
Theta = np.arange(0,2*np.pi,(2*np.pi)/729)

#convert R,theta space to y,z space over rotor 
Y = [[0]*len(Theta)]*len(R)
Z = [[0]*len(Theta)]*len(R)
ir = 0
for r in R:
    itheta = 0
    for theta in Theta:
        Y[ir][itheta] = r*np.cos(theta)
        Z[ir][itheta] = r*np.sin(theta)

        itheta+=1
    ir+=1

#center rotor on 0,0 coordinate
Y = Y - 2560
Z = Z - 90

#create dy,dz arrays
dy = [[0]*len(Theta)]*len(R)
dz = [[0]*len(Theta)]*len(R)
for iy in np.arange(len(Y)):
    for iz in np.arange(len(Z)):
        if iz == len(Z) or iy == len(Y):
            dz[iz] = dz[iz-1]
            dz[iy] = dy[iy-1]
        else:
            dz[iy][iz] = Z[iz+1] - Z[iz]
            dy[iy][iz] = Y[iy+1] - Z[iy]


#create y,z space for interpolation
y = p_rotor.ijk_dims[1] #no. data points
z = p_rotor.ijk_dims[2] #no. data points

y_array = np.linspace(2560-200,2560+200,y)
z_array = np.linspace(p_rotor.origin[2],(p_rotor.origin[2]+p_rotor.axis2[2]),z)
Y_og,Z_og = np.meshgrid(y_array,z_array)

#loop over all times
for it in np.arange(0,len(time)):

    #calculate horizontal velocity magnitude
    velx = p_rotor.variables["velocityx"][it]
    vely = p_rotor.variables["velocityy"][it]
    hvelmag = np.sqrt(np.square(velx) + np.square(vely))    

    #interpolate onto R,theta space over rotor
    np.reshape(hvelmag, (y,z))    
    f = interp.interp2d(Y_og, Z_og, hvelmag, kind='linear')
    hvelmag_r_theta = f(R, Theta)

    #search function to find index of theta +/- 2pi/3
    def search_2pi(theta):

        for i in np.arange(0,len(Theta)):
            if Theta[i+1] >= theta:
                break
        
        return i

    #calculate delta_Ux(r,theta)
    delta_Ux = [[0]*len(Theta)]*len(R)
    for r in np.arange(0,len(R)):
        for theta in np.arange(0,len(Theta)):

            #search for index +/- 2pi/3 either side of current theta
            if Theta[theta]+(2*np.pi)/3 >= (2*np.pi):
                blade_2 = Theta[theta]+(2*np.pi)/3 - (2*np.pi)  
            else:
                blade_2 = Theta[theta]+(2*np.pi)/3
            theta_2 = search_2pi(blade_2)

            if Theta[theta]-(2*np.pi)/3 < 0:
                blade_3 = Theta[theta]+(2*np.pi)/3 + (2*np.pi)
            else:
                blade_3 = Theta[theta]-(2*np.pi)/3
            theta_3 = search_2pi(blade_3)


            Ux_1 = hvelmag[r][theta]
            Ux_2 = hvelmag[r][theta_2]
            Ux_3 = hvelmag[r][theta_3]

            delta_Ux[r][theta] = np.max( [abs( Ux_1 - Ux_2 ), abs( Ux_1 - Ux_3 )] )

    IA = 0
    for j in np.arange(R):
        for k in np.arange(Theta):
            R_yz = np.sqrt( np.square(Y[j]) + np.square(Z[k]) )

            IA+= R_yz * delta_Ux[j][k] * dz[k] * dy[j]



