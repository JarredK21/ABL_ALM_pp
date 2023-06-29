import matplotlib.pyplot as plt
import numpy as np
from decimal import Decimal

Ex = ["Ex2","Ex1","Ex3"]
cases = [47,54,59]
colors = ["red","blue","green"]

fig = plt.figure(figsize=(14,8))

plt.axvline(-100,linestyle="dashed",color="red")
plt.axvline(-100,linestyle="dashed",color="blue")
plt.axvline(-100,linestyle="dashed",color="green")
plt.legend(["47","54","59"])
plt.xlim((0,1))
for j in np.arange(0,len(cases)):

    fname = '../../ALM_sensitivity_analysis/{0}/NREL_5MW_AeroDyn_Blade{1}'.format(Ex[j],cases[j])
    data = np.loadtxt(fname + '.dat', 
        skiprows=6, comments='!')

    Span = data[:,0]
    Span = Span/Span[-1]
    Chord = data[:,5]
    ID = data[:,6]

    fig = plt.figure(figsize=(14,8))
    plt.plot(Span,0.5*Chord)
    plt.axhline(1.0)
    plt.xlabel("Non-dimensionalised Span [-]")
    plt.ylabel("$\epsilon$ [m]")
    plt.show()

    if j == len(cases)-1:
        plt.plot(Span,Chord,color=colors[j])

    ic = 1
    for i in np.arange(0,len(ID)):
        if ID[i] > ic:
            plt.axvline(Span[i],linestyle="dashed",color=colors[j])
            ic+=1



plt.xlabel("Non-dimensional Span [-]")
plt.ylabel("Chord [m]")
plt.show()


