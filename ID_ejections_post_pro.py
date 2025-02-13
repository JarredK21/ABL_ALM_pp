from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
import time
from multiprocessing import Pool
from scipy import interpolate
from matplotlib.patches import Circle


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


def moments(y):
    mu = np.mean(y)
    std = np.std(y)
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    return mu, std, skewness,kurotsis


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Threshold_Asymmetry_Dataset.nc")

Time_a = np.array(a.variables["Time"])
Time_a = Time_a - 38000

b = Dataset(in_dir+"Asymmetry_Dataset.nc")

Time_b = np.array(b.variables["time"])
Time_b = Time_b - Time_b[0]
start_idx = np.searchsorted(Time_b,200)
Time_b = Time_b[start_idx:]

Thresholds = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.4]

out_dir=in_dir+"Asymmetry_analysis/Threshold_Asymmetry/"

for threshold in Thresholds:

    group_a = a.groups["{}".format(threshold)]

    print(group_a)

    Iy_a = np.array(group_a.variables["Iy_ejection"])
    Iz_a = -np.array(group_a.variables["Iz_ejection"])
    I_a = np.sqrt(np.add(np.square(Iy_a),np.square(Iz_a)))

    plt.rcParams.update({'font.size': 18})

    Iy_b = np.array(b.variables["Iy_low"][start_idx:])
    Iz_b = -np.array(b.variables["Iz_low"][start_idx:])
    I_b = np.sqrt(np.add(np.square(Iy_b),np.square(Iz_b)))

    Iy_P = np.true_divide(abs(Iy_a),abs(Iy_b))
    Iz_P = np.true_divide(abs(Iz_a),abs(Iz_b))
    I_P = np.true_divide(abs(I_a),abs(I_b))

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_a,I_a,"-b",label="Asymmetry due to surges")
    plt.grid()
    plt.xlabel("Time [s]")
    plt.plot(Time_b,I_b,"-r",label="Asymmetry due to low speed areas")
    plt.ylabel("Magnitude Asymmetry vector [$m^4/s$]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"I_low_I_ej_{}.png".format(threshold))
    plt.close()

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(21,12),sharex=True)
    ax1.plot(Time_a,Iy_a,"-b",label="Asymmetry due to surges")
    ax1.grid()
    fig.supxlabel("Time [s]")
    ax1.plot(Time_b,Iy_b,"-r",label="Asymmetry due to low speed areas")
    ax1.set_title("Asymmetry around y axis [$m^4/s$]")
    ax1.legend()
    ax2.plot(Time_a,Iy_P)
    ax2.grid()
    ax2.set_title("Iy (surges) / Iy (low speed areas) [-]")
    ax2.set_ylim([0,1])
    plt.tight_layout()
    plt.savefig(out_dir+"Iy_low_Iy_ej_{}_perc.png".format(threshold))
    plt.close()

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(21,12),sharex=True)
    ax1.plot(Time_a,Iz_a,"-b",label="Asymmetry due to surges")
    ax1.grid()
    fig.supxlabel("Time [s]")
    ax1.plot(Time_b,Iz_b,"-r",label="Asymmetry due to low speed areas")
    ax1.set_title("Asymmetry around z axis [$m^4/s$]")
    ax1.legend()
    ax2.plot(Time_a,Iz_P)
    ax2.grid()
    ax2.set_title("Iz (surges) / Iz (low speed areas) [-]")
    ax2.set_ylim([0,1])
    plt.tight_layout()
    plt.savefig(out_dir+"Iz_low_Iz_ej_{}_perc.png".format(threshold))
    plt.close()

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(21,12),sharex=True)
    ax1.plot(Time_a,I_a,"-b",label="Asymmetry due to surges")
    ax1.grid()
    fig.supxlabel("Time [s]")
    ax1.plot(Time_b,I_b,"-r",label="Asymmetry due to low speed areas")
    ax1.set_title("Magnitude Asymmetry vector [$m^4/s$]")
    ax1.legend()
    ax2.plot(Time_a,I_P)
    ax2.grid()
    ax2.set_title("I (surges) / I (low speed areas) [-]")
    ax2.set_ylim([0,1])
    plt.tight_layout()
    plt.savefig(out_dir+"I_low_I_ej_{}_perc.png".format(threshold))
    plt.close()


    # fig = plt.figure(figsize=(14,8))
    # P,X = probability_dist(Iy_a)
    # plt.plot(X,P,"-b",label="Asymmetry due to surges")
    # P,X = probability_dist(Iy_b)
    # plt.plot(X,P,"-r",label="Asymmetry due to low speed areas")
    # plt.xlabel("Asymmetry around y axis [$m^4/s$]")
    # plt.ylabel("Probability [-]")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(out_dir+"Iy_low_Iy_ej_{}_pdf.png".format(threshold))
    # plt.close()

    # fig = plt.figure(figsize=(14,8))
    # P,X = probability_dist(Iz_a)
    # plt.plot(X,P,"-b",label="Asymmetry due to surges")
    # P,X = probability_dist(Iz_b)
    # plt.plot(X,P,"-r",label="Asymmetry due to low speed areas")
    # plt.xlabel("Asymmetry around z axis [$m^4/s$]")
    # plt.ylabel("Probability [-]")
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(out_dir+"Iz_low_Iz_ej_{}_pdf.png".format(threshold))
    # plt.close()

    fig = plt.figure(figsize=(14,8))
    P,X = probability_dist(I_a)
    plt.plot(X,P,"-b",label="Asymmetry due to surges")
    P,X = probability_dist(I_b)
    plt.plot(X,P,"-r",label="Asymmetry due to low speed areas")
    plt.xlabel("Magnitude Asymmetry vector [$m^4/s$]")
    plt.ylabel("Probability [-]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"I_low_I_ej_{}_pdf.png".format(threshold))
    plt.close()




