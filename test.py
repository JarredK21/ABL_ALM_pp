import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from matplotlib import cm
from math import floor, ceil
from netCDF4 import Dataset
from matplotlib.patches import Circle
import pandas as pd

def testBoolan(x):
    if not any(x) != True and all(x) != True:
        return False
    elif all(x) == True and not any(x) == False:
        return True
    elif all(x) == False and not any(x) == True:
        return True


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


# #how to track eddies that move in time and change shape and might leave rotor disk
# def eddyIndentification():

#     #identification number for eddies at t=0
#     if T == 0:
#         for ic in np.arange(0,len(Eddies)):
#             Eddies[ic].number = ic

#     else:
#         #identify eddies at t=1 from eddies at t=0
#         Centroid_diff = np.inf; Area_difference = np.inf
#         for ii in np.arange(0,len(Eddies_N)):
#             for ij in np.arange(0,len(Eddies)):

#                 if abs(Eddies[ij].centroid - Eddies_N[ii].centroid) < Centroid_diff and abs(Eddies[ij].Area - Eddies_N[ii].Area) < Area_difference:
#                     Centroid_diff = np.average(Eddies[ij].centroid - Eddies_N[ii].centroid)
#                     Area_difference = Eddies[ij].Area - Eddies_N[ii].Area < Area_difference
#                     Eddies[ij].number = Eddies_N[ii].number


#     Eddies.sort(key=lambda Eddies: Eddies.number)

#     #number all new eddies
#     for ic in np.arange(0,len(Eddies)):
#         if Eddies[ic].number == np.nan:
#             Eddies[ic].number = ic



def openContour(cc,X,Y):
    crossing = []
    if any(cc) == False and all(cc) == False: #if all points are false skip loop in array
        return "skip", X, Y, cc, crossing
    elif any(cc) == True and all(cc) == True:
        return "closed", X, Y, cc, crossing
    else:
        #if there are points inside and outside remove points outside while interpolating between points to find points on edge of rotor
        X_temp = np.copy(X); Y_temp = np.copy(Y); cc_temp = np.copy(cc)
        ix = 0
        for i in np.arange(0,len(cc)-1):

            if cc[i+1] != cc[i]: #if next point is not the same as current (True -> False) or (False -> True) find point inbetween on rotor edge

                #equation of line intersecting circle
                m = (Y[i+1]-Y[i])/(X[i+1]-X[i])
                if m == np.inf or m ==-np.inf:
                    f = interpolate.interp1d([X[i+1],X[i]], [Y[i+1],Y[i]], fill_value='extrapolate')
                    c = float(f(0))
                    y_root = c
                else:
                    f = interpolate.interp1d([X[i+1],X[i]], [Y[i+1],Y[i]], fill_value='extrapolate')
                    c = float(f(0))
                    y_roots = np.roots([(1+(m)**2), ((2*-50)+(2*m*(c-50))), ((-50)**2 + (c-50)**2 - 20**2)])
                    if y_roots[0] > np.min([X[i], X[i+1]]) and y_roots[0] < np.max([X[i], X[i+1]]):
                        y_root = y_roots[0]
                    else:
                        y_root = y_roots[1]
                    del y_roots

                #z roots    
                z_roots = np.roots([1, (2*-50), (50**2+(y_root-50)**2 - 20**2)])
                if z_roots[0] > np.min([Y[i], Y[i+1]]) and z_roots[0] < np.max([Y[i], Y[i+1]]):
                    z_root = z_roots[0]
                else:
                    z_root = z_roots[1]
                del z_roots

                #insert x_root,y_root into temporary X,Y
                X_temp = np.insert(X_temp,ix+1,y_root); Y_temp = np.insert(Y_temp, ix+1,z_root); cc_temp = np.insert(cc_temp, ix+1, True)
                if cc[i] == True and cc[i+1] == False:
                    crossing.append(["inOut", ix+1])
                elif cc[i] == False and cc[i+1] == True:
                    crossing.append(["outIn", ix+1])

                ix+=1 #add one for inserting new value
            ix+=1 #add one in for loop

        return "open", X_temp, Y_temp, cc_temp, crossing


def closeContour(X, Y, cc):

    X_contour = []; Y_contour = []
    ix = np.nan; iy = np.nan
    for i in np.arange(0,len(cc)-1):
        if cc[i] == True:
            X_contour.append(X[i]); Y_contour.append(Y[i])
        
        if i < len(cc)-1 and cc[i] != cc[i+1]:
            ix = i

        if ix != np.nan and i != ix and cc[i] != cc[i+1]:
            iy = i
            theta_0 = np.arctan2(Y[ix], X[ix])
            theta_2 = np.arctan2(Y[iy], X[iy])

            theta_arc = np.arange(theta_0,theta_2,5e-03)

            for theta in theta_arc:
                if theta < 0.0 and theta >= -np.pi/2: #bottom right quadrant
                    r = -20; theta = np.pi/2 - theta
                    x_i = 50 - r*np.sin(theta)
                    y_i = 50 + r*np.cos(theta)
                elif theta < 0.0 and theta < -np.pi/2: #bottom left quadrant
                    r = -20; theta = theta - np.pi/2
                    x_i = 50 + r*np.sin(theta)
                    y_i = 50 + r*np.cos(theta)
                elif theta >= 0.0 and theta <= np.pi/2: #top right quadrant
                    r = 20; theta = np.pi/2 - theta
                    x_i = 50 + r*np.sin(theta)
                    y_i = 50 + r*np.cos(theta)
                elif theta >= 0.0 and theta >= np.pi/2: #top left quadrant
                    r = 20; theta = theta - np.pi/2
                    x_i = 50 - r*np.sin(theta)
                    y_i = 50 + r*np.cos(theta)

                X_contour.append(x_i); Y_contour.append(y_i)

            ix = np.nan; iy = np.nan

    return X_contour, Y_contour


def isInsideContour(Centroid, X_contour, Y_contour):
    inside = False

    x,y = Centroid[0],Centroid[1]
    p1x = X_contour[0]; p1y = Y_contour[0]

    num_vertices = len(X_contour)

    for i in range(1, num_vertices + 1):
            # Get the next point in the polygon
            p2x = X_contour[i % num_vertices]; p2y = Y_contour[i % num_vertices]
    
            # Check if the point is above the minimum y coordinate of the edge
            if y > min(p1y, p2y):
                # Check if the point is below the maximum y coordinate of the edge
                if y <= max(p1y, p2y):
                    # Check if the point is to the left of the maximum x coordinate of the edge
                    if x <= max(p1x, p2x):
                        # Calculate the x-intersection of the line connecting the point to the edge
                        x_intersection = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
    
                        # Check if the point is on the same line as the edge or to the left of the x-intersection
                        if p1x == p2x or x <= x_intersection:
                            # Flip the inside flag
                            inside = not inside
    
            # Store the current point as the first point for the next iteration
            p1x = p2x; p1y = p2y
    
    
    # Return the value of the inside flag
    return inside


    


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/test/"

Time = np.arange(0,10,1)

#parallize
df = pd.DataFrame(None)
print(df)

for T in Time:

    u = np.random.rand(100,100)
    threshold = 0.7
    x,y = np.shape(u)
    rez = 1
    xs = np.arange(0,x,rez)
    ys = np.arange(0,y,rez)

    X,Y = np.meshgrid(xs,ys)

    f_ux = interpolate.interp2d(X,Y,u)

    cmin = floor(np.min(u)); cmax = ceil(np.max(u)); levels = np.linspace(cmin,cmax,11)

    levels_pos = np.linspace(threshold,cmax,4)
    CS = plt.contour(X, Y, u, levels=levels_pos, colors='r')

    fig,ax = plt.subplots(figsize=(40,40))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(X,Y,u,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)
    cb = plt.colorbar(cs)

    Drawing_uncolored_circle = Circle( (50, 50),radius=20 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)


    #for +0.7m/s threshold
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []
    lines = CS.allsegs[0] #plot only threshold velocity
    for line in lines:
        X, Y = line[:,0], line[:,1]

        #check if any point in line is inside circle
        cc = []
        for X_line, Y_line in zip(X, Y):
            cc.append(isInside(X_line,Y_line))

        #separate line into N contours if line is outside rotor disk
        C, X, Y,cc,crossings = openContour(cc,X,Y)
        #all points are outside of rotor disk
        if C == "skip":
            continue
        else:

            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            ux = f_ux(Centroid[0], Centroid[1])

            if C == "open":                

                #inside = isInsideContour(Centroid, X, Y)
                #print(inside)

                for crossing in crossings:
                    if crossing[0] == "inOut":
                        Bx = X[:crossing[1]-1]
                        Ax = X[crossing[1]:]
                        By = Y[:crossing[1]-1]
                        Ay = Y[crossing[1]:]
                        Bcc = cc[:crossing[1]-1]
                        Acc = cc[crossing[1]:]
                X = np.concatenate((Ax,Bx))
                Y = np.concatenate((Ay,By))
                cc = np.concatenate((Acc,Bcc))

                X,Y = closeContour(X,Y,cc)


        #calculate area and centroid of each contour
        Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
        X = np.append(X,X[0]); Y = np.append(Y,Y[0])
        Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)
        print(C)
        print(cc)
        print(X,Y)
        plt.plot(X,Y,"-k",linewidth=4)
        plt.plot(Centroid[0],Centroid[1],"ok",markersize=3)

    #     Eddies_Cent_x.append(Centroid[0])
    #     Eddies_Cent_y.append(Centroid[1])
    #     Eddies_Area.append(Area)

    # Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area}

    x_c = [-1,-10]; y_c = [-1,-10]
    plt.plot(x_c,y_c,"-k",label="0.7 m/s")
    #saving figure
    plt.xlim([0,x]); plt.ylim([0,y])
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("{}".format(T))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(in_dir+"{}.png".format(T))
    plt.cla()
    cb.remove()
    plt.close(fig)

    #turned off for now
    #eddyIndentification()

    # df_0 = pd.DataFrame(Eddies_it_pos)
    # df = pd.concat([df,df_0],axis=1)

    # print(df)

#saving data

#print(df)
#df.to_csv(in_dir+"Eddies.csv"); del df


#store current time step for comparison with t+ti turned off for now
#Eddies_N = Eddies; del Eddies