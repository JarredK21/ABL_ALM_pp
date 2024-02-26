import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import cm
from math import floor, ceil
from netCDF4 import Dataset
from matplotlib.patches import Circle
import pandas as pd


def isInside(x, y):
     
    if ((x - 50) * (x - 50) +
        (y - 50) * (y - 50) <= 20 * 20):
        return True
    else:
        return False


class eddy:
    def __init__(self, Centroid_x,Centroid_y, Area):
        self.number = np.nan
        self.Centroid_x = Centroid_x
        self.Centroid_y = Centroid_y
        self.Area = Area


#how to track eddies that move in time and change shape and might leave rotor disk
def eddyIndentification():

    #identification number for eddies at t=0
    if T == 0:
        for ic in np.arange(0,len(Eddies)):
            Eddies[ic].number = ic

    else:
        #identify eddies at t=1 from eddies at t=0
        Centroid_diff = np.inf; Area_difference = np.inf
        for ii in np.arange(0,len(Eddies_N)):
            for ij in np.arange(0,len(Eddies)):

                if abs(Eddies[ij].centroid - Eddies_N[ii].centroid) < Centroid_diff and abs(Eddies[ij].Area - Eddies_N[ii].Area) < Area_difference:
                    Centroid_diff = np.average(Eddies[ij].centroid - Eddies_N[ii].centroid)
                    Area_difference = Eddies[ij].Area - Eddies_N[ii].Area < Area_difference
                    Eddies[ij].number = Eddies_N[ii].number


    Eddies.sort(key=lambda Eddies: Eddies.number)

    #number all new eddies
    for ic in np.arange(0,len(Eddies)):
        if Eddies[ic].number == np.nan:
            Eddies[ic].number = ic



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/test/"

Time = np.arange(0,10,1)

columns = []
for T in Time:
    columns.append("Centroid_x_{}".format(T))
    columns.append("Centroid_y_{}".format(T))
    columns.append("Area_{}".format(T))

#parallize
df = pd.DataFrame(None)
print(df)

for T in Time:
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []

    u = np.random.rand(100,100)
    threshold = 0.7
    x,y = np.shape(u)
    rez = 1
    xs = np.arange(0,x,rez)
    ys = np.arange(0,y,rez)

    X,Y = np.meshgrid(xs,ys)

    idx_left = np.searchsorted(xs,30,side="left"); idx_right = np.searchsorted(xs,70,side="right")
    idx_top = np.searchsorted(ys,70,side="right"); idx_bottom = np.searchsorted(ys,30,side="left")
    X_temp = X[idx_left:idx_right,idx_bottom:idx_top]; Y_temp = Y[idx_left:idx_right,idx_bottom:idx_top]
    u_temp = u[idx_left:idx_right,idx_bottom:idx_top]

    cmin = floor(np.min(u)); cmax = ceil(np.max(u)); levels = np.linspace(cmin,cmax,11)

    levels_pos = np.linspace(threshold,cmax,4)
    CS = plt.contour(X_temp, Y_temp, u_temp, levels=levels_pos, colors='r')

    fig,ax = plt.subplots(figsize=(40,40))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(X,Y,u,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)
    cb = plt.colorbar(cs)

    Drawing_uncolored_circle = Circle( (50, 50),radius=20 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)

    lines = CS.allsegs[0]

    for line in lines:
        X, Y = line[:,0], line[:,1]

        #check if any point in line is inside circle
        cc = []
        for X_line, Y_line in zip(X,Y):
            cc.append(isInside(X_line,Y_line))

        X_temp = np.copy(X); Y_temp = np.copy(Y); cc_temp = np.copy(cc)
        #if any point is inside cirlce plot #stop points outside of circle
        res = not any(cc)
        if res == False:     
            ix = 0
            for ic in np.arange(0,len(cc)-1):
                if cc[ic+1] != cc[ic]:


                    #equation of line intersecting circle
                    m = (Y[ic+1]-Y[ic])/(X[ic+1]-X[ic])
                    if m == np.inf or m ==-np.inf:
                        f = interpolate.interp1d([X[ic+1],X[ic]], [Y[ic+1],Y[ic]], fill_value='extrapolate')
                        c = float(f(0))
                        x_root = c
                    else:
                        f = interpolate.interp1d([X[ic+1],X[ic]], [Y[ic+1],Y[ic]], fill_value='extrapolate')
                        c = float(f(0))
                        x_roots = np.roots([(1+(m)**2), ((2*-50)+(2*m*(c-50))), ((-50)**2 + (c-50)**2 - 20**2)])
                        if x_roots[0] > np.min([X[ic], X[ic+1]]) and x_roots[0] < np.max([X[ic], X[ic+1]]):
                            x_root = x_roots[0]
                        else:
                            x_root = x_roots[1]
                        del x_roots

                    #y roots    
                    y_roots = np.roots([1, -100, (2500+(x_root-50)**2 - 20**2)])
                    if y_roots[0] > np.min([Y[ic], Y[ic+1]]) and y_roots[0] < np.max([Y[ic], Y[ic+1]]):
                        y_root = y_roots[0]
                    else:
                        y_root = y_roots[1]
                    del y_roots

                    #insert x_root,y_root into X,Y and insert true at same index in cc
                    X_temp = np.insert(X_temp, ix+1, x_root); Y_temp = np.insert(Y_temp, ix+1, y_root); cc_temp = np.insert(cc_temp,ix+1,"True")

                    ix+=1 #add one for inserting
                ix+=1 #add one to increase index

        X = X_temp[cc_temp]; Y = Y_temp[cc_temp]; del X_temp; del Y_temp; del cc_temp

        plt.plot(X, Y,"-k")

        if len(X) > 0:
            #calculate area and centroid of each contour
            np.append(X,X[-1]); np.append(Y,Y[-1])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)
            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]

            plt.plot(Centroid[0],Centroid[1],"ok",markersize=5)
            
            #fix so next time step starts at index 0
            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

    Eddies_it = {"Centroid_x_{}".format(T): Eddies_Cent_x, "Centroid_y_{}".format(T): Eddies_Cent_y, "Area_{}".format(T): Eddies_Area}
    
    #saving figure
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("{}".format(T))
    plt.grid()
    plt.tight_layout()
    plt.savefig(in_dir+"{}.png".format(T))
    plt.cla()
    cb.remove()
    plt.close(fig)

    #turned off for now
    #eddyIndentification()

    df_0 = pd.DataFrame(Eddies_it)
    df = pd.concat([df,df_0],axis=1)

    print(df)

#saving data

print(df)
df.to_csv(in_dir+"Eddies.csv"); del df


#store current time step for comparison with t+ti turned off for now
#Eddies_N = Eddies; del Eddies