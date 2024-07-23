import pyFAST.input_output as io
import numpy as np
import matplotlib.pyplot as plt

df = io.fast_output_file.FASTOutputFile("NREL_5MW_Main.out").toDataFrame()

Time = np.array(df["Time_[s]"])
Tstart_idx = np.searchsorted(Time,200)
Tend_idx = np.searchsorted(Time,1201)

act_stations = 300

x = np.linspace(0,1,act_stations)

Vars = ["Fn","Ft","Fx","Fy"]; units = ["[N/m]","[N/m]","[N/m]","[N/m]"]
Ylabel = ["Local Aerofoil Normal Force", "Local Aerofoil Tangential Force","Local Aerofoil Force in x direction","Local Aerofoil Force in y direction"]

ix=0
for Var,unit in zip(Vars,units):

    fig = plt.figure(figsize=(14,8))

    Var_list = []
    for i in np.arange(1,act_stations+1):
        if i < 10:
            txt = "AB1N00{0}{1}_{2}".format(i,Var,unit)
        elif i >= 10 and i < 100:
            txt = "AB1N0{0}{1}_{2}".format(i,Var,unit)
        elif i >= 100:
            txt = "AB1N{0}{1}_{2}".format(i,Var,unit)


        Var_dist = np.average(df[txt][Tstart_idx:Tend_idx])
        
        Var_list.append(Var_dist)

    plt.plot(x,Var_list,"-ob",markersize=1)

    plt.ylabel("{0} {1}".format(Ylabel[ix],unit),fontsize=16)
    plt.xlabel("Normalized blade radius [-]",fontsize=16)
    plt.title("Averaged over 1000s",fontsize=12)
    plt.tight_layout()
    plt.savefig("{0}.png".format(Var))
    plt.close(fig)

    ix+=1