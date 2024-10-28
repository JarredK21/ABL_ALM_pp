import matplotlib.pyplot as plt
import numpy as np
import pyFAST.input_output as io
import pandas as pd
from scipy import interpolate
from netCDF4 import Dataset
from multiprocessing import Pool


def rotate_coordinates(it):

    xo = np.array(WT_E.variables["xyz"][it,300,0])
    yo = np.array(WT_E.variables["xyz"][it,300,1])


    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))

    return xs


def moment_calc(it):

    xo = np.array(WT_E.variables["xyz"][it,1:,0])
    yo = np.array(WT_E.variables["xyz"][it,1:,1])


    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))



    My = np.sum(BWeight_ext*xs)/1000

    return My


twist = np.array([1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01,
    1.3308000E+01, 1.3181000E+01, 1.2848000E+01, 1.2192000E+01, 1.1561000E+01, 1.1072000E+01, 1.0792000E+01, 1.0232000E+01, 9.6720000E+00, 9.1100000E+00, 8.5340000E+00,
    7.9320000E+00, 7.3210000E+00, 6.7110000E+00, 6.1220000E+00, 5.5460000E+00, 4.9710000E+00, 4.4010000E+00, 3.8340000E+00, 3.3320000E+00, 2.8900000E+00, 2.5030000E+00,
    2.1160000E+00, 1.7300000E+00, 1.3420000E+00, 9.5400000E-01, 7.6000000E-01, 5.7400000E-01, 4.0400000E-01, 3.1900000E-01, 2.5300000E-01, 2.1600000E-01, 1.7800000E-01,
    1.4000000E-01, 1.0100000E-01, 6.2000000E-02, 2.3000000E-02, 0.0000000E+00])

BlFract = np.array([0.0000000E+00, 3.2500000E-03, 1.9510000E-02, 3.5770000E-02, 5.2030000E-02, 6.8290000E-02, 8.4550000E-02, 1.0081000E-01, 1.1707000E-01, 1.3335000E-01, 1.4959000E-01,
    1.6585000E-01, 1.8211000E-01, 1.9837000E-01, 2.1465000E-01, 2.3089000E-01, 2.4715000E-01, 2.6341000E-01, 2.9595000E-01, 3.2846000E-01, 3.6098000E-01, 3.9350000E-01, 
    4.2602000E-01, 4.5855000E-01, 4.9106000E-01, 5.2358000E-01, 5.5610000E-01, 5.8862000E-01, 6.2115000E-01, 6.5366000E-01, 6.8618000E-01, 7.1870000E-01, 7.5122000E-01,
    7.8376000E-01, 8.1626000E-01, 8.4878000E-01, 8.8130000E-01, 8.9756000E-01, 9.1382000E-01, 9.3008000E-01, 9.3821000E-01, 9.4636000E-01, 9.5447000E-01, 9.6260000E-01,
    9.7073000E-01, 9.7886000E-01, 9.8699000E-01, 9.9512000E-01, 1.0000000E+00])

FlapStiffness = [1.8110000E+10, 1.8110000E+10, 1.9424900E+10, 1.7455900E+10, 1.5287400E+10, 1.0782400E+10, 7.2297200E+09, 6.3095400E+09, 5.5283600E+09, 4.9800600E+09, 4.9368400E+09,
    4.6916600E+09, 3.9494600E+09, 3.3865200E+09, 2.9337400E+09, 2.5689600E+09, 2.3886500E+09, 2.2719900E+09, 2.0500500E+09, 1.8282500E+09, 1.5887100E+09, 1.3619300E+09, 1.1023800E+09,
    8.7580000E+08, 6.8130000E+08, 5.3472000E+08, 4.0890000E+08, 3.1454000E+08, 2.3863000E+08, 1.7588000E+08, 1.2601000E+08, 1.0726000E+08, 9.0880000E+07, 7.6310000E+07, 6.1050000E+07,
    4.9480000E+07, 3.9360000E+07, 3.4670000E+07, 3.0410000E+07, 2.6520000E+07, 2.3840000E+07, 1.9630000E+07, 1.6000000E+07, 1.2830000E+07, 1.0080000E+07, 7.5500000E+06, 4.6000000E+06,
    2.5000000E+05, 1.7000000E+05]

EgdeStiffness = [1.8113600E+10, 1.8113600E+10, 1.9558600E+10, 1.9497800E+10, 1.9788800E+10, 1.4858500E+10, 1.0220600E+10, 9.1447000E+09, 8.0631600E+09, 6.8844400E+09, 7.0091800E+09,
    7.1676800E+09, 7.2716600E+09, 7.0817000E+09, 6.2445300E+09, 5.0489600E+09, 4.9484900E+09, 4.8080200E+09, 4.5014000E+09, 4.2440700E+09, 3.9952800E+09, 3.7507600E+09, 3.4471400E+09,
    3.1390700E+09, 2.7342400E+09, 2.5548700E+09, 2.3340300E+09, 1.8287300E+09, 1.5841000E+09, 1.3233600E+09, 1.1836800E+09, 1.0201600E+09, 7.9781000E+08, 7.0961000E+08, 5.1819000E+08,
    4.5487000E+08, 3.9512000E+08, 3.5372000E+08, 3.0473000E+08, 2.8142000E+08, 2.6171000E+08, 1.5881000E+08, 1.3788000E+08, 1.1879000E+08, 1.0163000E+08, 8.5070000E+07, 6.4260000E+07,
    6.6100000E+06, 5.0100000E+06]

BMassDen = [6.7893500E+02, 6.7893500E+02, 7.7336300E+02, 7.4055000E+02, 7.4004200E+02, 5.9249600E+02, 4.5027500E+02, 4.2405400E+02, 4.0063800E+02, 3.8206200E+02, 3.9965500E+02,
            4.2632100E+02, 4.1682000E+02, 4.0618600E+02, 3.8142000E+02, 3.5282200E+02, 3.4947700E+02, 3.4653800E+02, 3.3933300E+02, 3.3000400E+02, 3.2199000E+02, 3.1382000E+02,
            2.9473400E+02, 2.8712000E+02, 2.6334300E+02, 2.5320700E+02, 2.4166600E+02, 2.2063800E+02, 2.0029300E+02, 1.7940400E+02, 1.6509400E+02, 1.5441100E+02, 1.3893500E+02, 
            1.2955500E+02, 1.0726400E+02, 9.8776000E+01, 9.0248000E+01, 8.3001000E+01, 7.2906000E+01, 6.8772000E+01, 6.6264000E+01, 5.9340000E+01, 5.5914000E+01, 5.2484000E+01,
            4.9114000E+01, 4.5818000E+01, 4.1669000E+01, 1.1453000E+01, 1.0319000E+01]

# out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(BlFract,BMassDen)
# plt.xlabel("Blade fraction [-]")
# plt.ylabel("Blade mass density [kg/m]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"BMassDen.png")
# plt.close()

f = interpolate.interp1d(BlFract,BMassDen)
BlFract_interp = np.linspace(0,1,300)
dr = BlFract[1] - BlFract[0]
BMassDen = f(BlFract_interp)

BWeight = list(BMassDen*dr*63*9.81)
BlFract_new = []
for i in np.arange(0,len(BlFract_interp)-1):
    BlFract_new.append(((BlFract_interp[i+1]-BlFract_interp[i])/2)+BlFract_interp[i])


BWeight_ext = BWeight+BWeight+BWeight

# fig = plt.figure(figsize=(14,8))
# plt.plot(BlFract_new,BWeight)
# plt.xlabel("Blade fraction [-]")
# plt.ylabel("Blade weight distribution [N]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../NREL_5MW_3.4.1/BldWeightDist.png")
# plt.close(fig)


# in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"
# df = pd.read_csv(in_dir+"mean_coordinates.csv")
# xco = np.array(df["x"])

# My = 0
# for i in np.arange(0,len(xco)):
#     My+=BWeight[i]*(xco[i]-xco[0])

# print(My/1000)


in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"
df_E = Dataset(in_dir+"WTG01b.nc")

WT_E = df_E.groups["WTG01"]

Time = np.array(WT_E.variables["time"])
Start_time_idx = np.searchsorted(Time,Time[0]+200)

Time_steps = np.arange(0,len(Time))
#Time = Time[Start_time_idx:]

Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]


ix = 0
xs = []
with Pool() as pool:
    for xs_it in pool.imap(rotate_coordinates,Time_steps):
        xs.append(xs_it)
        print(ix)
        ix+=1

out_dir=in_dir+"Elastic_deformations_analysis/"
plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(14,8))
plt.plot(Time,xs)
plt.xlabel("Time [s]")
plt.ylabel("$x_H$ coordinate blade tip [m]")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"xH_tip_coordinate.png")
plt.close(fig)

idx = np.searchsorted(Time,Time[0]+100)
fig = plt.figure(figsize=(14,8))
plt.plot(Time[:idx],xs[:idx])
plt.xlabel("Time [s]")
plt.ylabel("$x_H$ coordinate blade tip [m]")
plt.title("First 100s initial OpenFAST transients")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"xH_tip_coordinate_100.png")
plt.close(fig)

ix = 0
My = []
with Pool() as pool:
    for My_it in pool.imap(moment_calc,Time_steps):
        My.append(My_it)
        print(ix)
        ix+=1

M_mean = round(np.mean(My),2); M_std = round(np.std(My),2) 
out_dir=in_dir+"Elastic_deformations_analysis/"
plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(14,8))
plt.plot(Time,My)
plt.axhline(y=np.mean(My),linestyle="--",color="k")
plt.xlabel("Time [s]")
plt.ylabel("Estimated $M_{H,y}$ due to weight [kN-m]")
plt.title("Mean = {}kN-m, Standard deviation = {}kN-m".format(M_mean,M_std))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"My_WR.png")
plt.close(fig)



R = 63; omega = (12.1*2*np.pi)/60

rot_speed_a = omega*(R*BlFract)*np.cos(np.radians(twist))

in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

Time = np.array(df["Time_[s]"])
Tstart_idx = np.searchsorted(Time,Time[0]+200)
Time = Time[Tstart_idx:]
Vrel = []
for i in np.arange(1,300):
    print(i)
    if i < 10:
        num = "00{}".format(i)
    elif i < 100:
        num = "0{}".format(i)
    elif i < 1000:
        num  = "{}".format(i)

    txt = "AB1N"+num+"Vrel_[m/s]"

    Vrel.append(np.average(df[txt][Tstart_idx:]))

x = np.linspace(0,1,299)

out_dir="../../NREL_5MW_3.4.1/"
fig = plt.figure(figsize=(14,8))
plt.plot(BlFract,rot_speed_a,"-b",label="Rot speed")
plt.plot(x,Vrel,"-r",label="Relative speed")
plt.xlabel("Blade fraction [-]")
plt.ylabel("Speed airfoil frame of reference [m/s]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"Vrel_rot_speed.png")
plt.close()


fig = plt.figure(figsize=(14,8))
plt.plot(BlFract,twist)
plt.xlabel("Blade fraction")
plt.ylabel("Structual twist [deg]")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"twist.png")
plt.close(fig)

fig = plt.figure(figsize=(14,8))
plt.yscale("log")
plt.plot(BlFract,FlapStiffness,"-b",label="Flapwise stiffness")
plt.plot(BlFract,EgdeStiffness,"-r",label="Edgewise stiffness")
plt.xlabel("Blade fraction")
plt.ylabel("Stiffness [$Nm^2$]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"stiffness.png")
plt.close(fig)