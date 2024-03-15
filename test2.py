import numpy as np
import matplotlib.pyplot as plt

theta_180 = [-3.1284718017074007, -0.4542504906113279, -0.4920159757201647]
theta_loc = [3.1547135054721855, 5.828934816568259, 5.791169331459422]
theta_order = [5.791169331459422, 5.828934816568259, 9.437898812651772]

direction =  "clockwise"
Atheta = 5.791169331459422

if direction == "anticlockwise":
    if theta_loc[1] < Atheta:
    
        theta_AB = np.linspace(theta_loc[1],Atheta,int(abs(theta_180[1]-theta_180[0])/5e-03))
    elif theta_loc[1] > Atheta:
        theta_AB1 = np.linspace(theta_180[1],0,int(abs(theta_180[1])/5e-03))
        theta_AB2 = np.linspace(0,Atheta,int(abs(Atheta)/5e-03))
        theta_AB = np.concatenate((theta_AB1,theta_AB2))
elif direction == "clockwise":
    if theta_loc[1] > Atheta:
        theta_AB = np.linspace(theta_loc[1],Atheta,int(abs(theta_loc[1]-Atheta)/5e-03))
    elif theta_loc[1] < Atheta:
        theta_AB1 = np.linspace(theta_loc[1],0,int(abs(theta_180[1])/5e-03))
        theta_AB2 = np.linspace(0,theta_180[0],int(abs(theta_180[0])/5e-03))
        theta_AB = np.concatenate((theta_AB1,theta_AB2))

for j in np.arange(0,len(theta_AB)):
    if theta_AB[j] < 0:
        theta_AB[j]+=2*np.pi
    elif theta_AB[j] >= 2*np.pi:
        theta_AB[j]-=2*np.pi

print("theta arc",theta_AB)