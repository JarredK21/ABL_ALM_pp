import numpy as np
import matplotlib.pyplot as plt

direction = "clockwise"

theta_180 = [-0.8312255436829765, 0.4142869849226743, -2.2174038593554912]
theta_loc = [5.4519597634966095, 0.4142869849226743, 4.0657814478240955]
theta_order = [-0.8312255436829767, 0.4142869849226743, 3.779925796744674]

#check this part not working all the time
if direction == "anticlockwise":
    if theta_loc[1] < theta_loc[0]:
    
        theta_AB = np.linspace(theta_loc[1],theta_loc[0],int(abs(theta_180[1]-theta_180[0])/5e-03))
    elif theta_loc[1] > theta_loc[0]:
        theta_AB1 = np.linspace(theta_180[1],0,int(abs(theta_180[1])/5e-03))
        theta_AB2 = np.linspace(0,theta_loc[0],int(theta_loc[0]/5e-03))
        theta_AB = np.concatenate((theta_AB1,theta_AB2))
elif direction == "clockwise":
    if theta_loc[1] > theta_loc[0]:
        theta_AB = np.linspace(theta_loc[1],theta_loc[0],int(abs(theta_180[1]-theta_180[0])/5e-03))
    elif theta_loc[1] < theta_loc[0]:
        theta_AB1 = np.linspace(theta_loc[1],0,int(abs(theta_loc[1])/5e-03))
        theta_AB2 = np.linspace(0,theta_180[0],int(abs(theta_180[0])/5e-03))
        theta_AB = np.concatenate((theta_AB1,theta_AB2))
print(theta_AB)
for i in np.arange(0,len(theta_AB)):
    if theta_AB[i] < 0:
        theta_AB[i]+=2*np.pi
    elif theta_AB[i] >= 2*np.pi:
        theta_AB[i]-=2*np.pi

print("theta arc",theta_AB)