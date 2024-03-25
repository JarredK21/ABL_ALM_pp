import numpy as np


def ux_offset_perc(ux_anti,ux_clock,theta,theta_180,perc):
    r = 63

    if ux_anti == np.nan:
        theta_anti = theta + abs(theta_180[2] - theta_180[1]) / (1/perc)

        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi

        x_anti = 2560 + r*np.cos(theta_anti)
        y_anti = 90 + r*np.sin(theta_anti)

        ux_anti = ux_interp([x_anti,y_anti])

    if ux_clock == np.nan:
            
        theta_clock = theta - abs(theta_180[1] - theta_180[0]) / (1/perc)

        if theta_clock < 0:
            theta_clock +=2*np.pi

        x_clock = 2560 + r*np.cos(theta_clock)
        y_clock = 90 + r*np.sin(theta_clock)     
        ux_clock = ux_interp([x_clock,y_clock])

    print(theta_anti,theta_clock,ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock)
    return ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock


def ux_offset_deg(type,theta,theta_order,dtheta):
    r = 63

    theta_anti = theta + dtheta

    if type == 2:

        theta_clock = theta - dtheta      
    else:
        theta_clock = theta - dtheta

    if round(theta_clock,2) <= round(theta_order[0],2):
        ux_clock = np.nan
        x_clock = np.nan
        y_clock = np.nan
    else:
        if theta_clock < 0:
            theta_clock +=2*np.pi
        x_clock = 2560 + r*np.cos(theta_clock)
        y_clock = 90 + r*np.sin(theta_clock)
        ux_clock = ux_interp([x_clock,y_clock])
    
    if round(theta_anti,2) >= round(theta_order[2],2):
        ux_anti = np.nan
        x_anti = np.nan
        y_anti = np.nan
    else:
        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi
        x_anti = 2560 + r*np.cos(theta_anti)
        y_anti = 90 + r*np.sin(theta_anti)
        ux_anti = ux_interp([x_anti,y_anti])


    print(theta_anti,theta_clock,ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock)
    return ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock


def isOutside(type,theta,theta_order,theta_180):

    dtheta_arr = np.radians([2,4,6,8,10,12,14,16,18,20,24,26])
    percentage = [0.5,0.55,0.45,0.60,0.40,0.65,0.35,0.70,0.30,0.75,0.35,0.80]

    ip = 0
    for dtheta in dtheta_arr:
        ux_anti = 0.78;ux_clock = np.nan

        if np.isnan(ux_anti) == True or np.isnan(ux_clock) == True:
            ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock = ux_offset_perc(ux_anti,ux_clock,theta,theta_180,percentage[ip])
            ip+=1

        if threshold > 0.0:
            if ux_anti >= threshold and ux_clock >= threshold:
                continue
            if ux_anti >= threshold or ux_clock >= threshold:
                break
        elif threshold < 0.0:
            if ux_anti<=threshold and ux_clock<=threshold:
                continue
            if ux_anti <= threshold or ux_clock<= threshold:
                break


threshold = 0.7
theta_loc = [3.545176343688043, 1.0366479284920258, 1.4042316265949564, 1.379600196612268, 3.545176343688043]

theta_order = np.sort(theta_loc)
theta_order = theta_order.tolist()

theta_180 = []
for theta in theta_order:
    if theta > np.pi:
        theta_180.append(theta-(2*np.pi))
    else:
        theta_180.append(theta)

type = 2
theta = 1.40
isOutside(type,theta,theta_order,theta_180)