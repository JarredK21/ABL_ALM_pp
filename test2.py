import numpy as np

theta_start = 3.2944793534995145 

Atheta = 9.577664660679101

Atheta-=2*np.pi

if round(Atheta,2) == round(theta_start,2):
    print(theta_start)

def ux_offset_perc(ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock,theta,theta_180,perc):
    r = 63

    if np.isnan(ux_anti) == True:
        theta_anti = theta + abs(theta_180[2] - theta_180[1]) / (1/perc)

        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi

        x_anti = 2560 + r*np.cos(theta_anti)
        y_anti = 90 + r*np.sin(theta_anti)

        ux_anti = 0.7

    if np.isnan(ux_clock) == True:
            
        theta_clock = theta - abs(theta_180[1] - theta_180[0]) / (1/perc)

        if theta_clock < 0:
            theta_clock +=2*np.pi

        x_clock = 2560 + r*np.cos(theta_clock)
        y_clock = 90 + r*np.sin(theta_clock)     
        ux_clock = 0.6

    print(ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock)
    return ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock


#ux_offset_perc(ux_anti=0.7,ux_clock=0.7,x_anti=1,y_anti=1,x_clock=0,y_clock=0,theta=0.5,theta_180=1,perc=0.5)