from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

def moments(y):
    mu = np.mean(y)
    std = np.std(y)
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    return mu, std, skewness,kurotsis

def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    dX = X[1] - X[0]
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    print(np.sum(P)*dX)
    return P,X

def correlation_coef(x,y):

    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df = Dataset(in_dir+"Dataset.nc")

Time = np.array(df.variables["time_OF"])

Time_start_idx = np.searchsorted(Time,200)
Time = Time[Time_start_idx:]

LSShftFxa = np.array(df.variables["LSShftFxa"][Time_start_idx:])
LSShftFys = np.array(df.variables["LSShftFys"][Time_start_idx:])
LSShftFzs = np.array(df.variables["LSShftFzs"][Time_start_idx:])
LSSTipMys = np.array(df.variables["LSSTipMys"][Time_start_idx:])
LSSTipMzs = np.array(df.variables["LSSTipMzs"][Time_start_idx:])

L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

print("FBR",moments(FBR))
print("FBa",moments(LSShftFxa))

Fa_Fr = np.true_divide(LSShftFxa,FBR)

#print(np.mean(Fa_Fr),np.mean(Fa_Fr)+np.std(Fa_Fr),np.mean(Fa_Fr)-np.std(Fa_Fr))

out_dir=in_dir+"Role_of_Thrust/"

plt.rcParams.update({'font.size': 18})
fig = plt.figure(figsize=(14,8))
plt.plot(Time,Fa_Fr,"-b")
plt.axhline(y=1.0,linestyle="dashdot",color="r")
plt.axhline(y=1.5,linestyle="dashdot",color="r",label="$e=1.5tan(45)$")
plt.axhline(y=0.22,linestyle="--",color="k",label="$e=1.5tan(8.34)$")
plt.axhline(y=0.3,linestyle="--",color="k",label="$e=1.5tan(11.34)$")
plt.xlabel("Time [s]")
plt.ylabel("$F_{B_x}/F_{B_R}$ - ratio axial bearing force component\nto radial bearing force")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"FBX_FBR_ratio.png")
plt.close()

E = [0.22,0.3]

for e in E:

    alpha = np.arctan(e/1.5)

    Pr = []
    XFr = []
    YFa = []
    e_it = []
    XFr_2 = []
    YFa_2 = []
    for it in np.arange(0,len(Time)):

        if Fa_Fr[it] <= e:
            e_it.append(0)
            X = 1; Y = 0.45*(1/np.tan(alpha))
        elif Fa_Fr[it] > e:
            e_it.append(1)
            X = 0.67; Y = 0.67*(1/np.tan(alpha))
            XFr_2.append(X*FBR[it]); YFa_2.append(Y*LSShftFxa[it])

        XFr.append(X*FBR[it]); YFa.append(Y*LSShftFxa[it])
        Pr.append(X*FBR[it]+Y*LSShftFxa[it])
    
    print(correlation_coef(Pr,FBR))
    print(correlation_coef(Pr,LSShftFxa))
    print("XFr_2",moments(XFr_2))
    print("YFa_2",moments(YFa_2))

    print(e_it.count(0))
    print(e_it.count(1))

    print("XFr",moments(XFr))
    print("YFa",moments(YFa))

    print("Pr",moments(Pr))

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time,XFr,"-b",label="$XF_{B_R}$")
    plt.plot(Time,YFa,"-r",label="$YF_{B_x}$")
    plt.xlabel("Time [s]")
    plt.ylabel("Modified Bearing force components [kN]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"XFr_YFa_{}.png".format(e))
    plt.close()


    fig = plt.figure(figsize=(14,8))
    P,X = probability_dist(XFr)
    plt.plot(X,P,"-b",label="$XF_{B_R}$")
    P,X = probability_dist(YFa)
    plt.plot(X,P,"-r",label="$YF_{B_x}$")
    plt.ylabel("Probability [-]")
    plt.xlabel("Modified Bearing force components [kN]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_XFr_YFa_{}.png".format(e))
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.plot(Time,Pr)
    plt.xlabel("Time [s]")
    plt.ylabel("Dynamic equivalent radial load [kN]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Pr_{}.png".format(e))
    plt.close()
    



