import pyFAST.input_output as io
import numpy as np

a = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

Azimuth = np.array(a["Azimuth_[deg]"])

print(Azimuth[100320-65000])