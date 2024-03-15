import numpy as np
import matplotlib.pyplot as plt

direction = "clockwise"

theta_180 = [-2.1235069656598324, -0.01788330627834868, -0.15272079523787366, -0.16297432102583254]
theta_loc = [4.159678341519754, 6.2653020009012375, 6.130464511941713, 6.1202109861537535]
theta_order = [4.159678341519754, 6.1202109861537535, 6.130464511941713, 6.2653020009012375]

#remove if dtheta < one grid cell (radians)
i = 0
while i < len(theta_order)-1:
    if theta_order[i+1] - theta_order[i] < 0.047:
        theta_order.remove(theta_order[i+1])
        theta_idx = theta_loc.index(theta_order[i+1])
        theta_loc.remove(theta_loc[theta_idx])
        theta_180.remove(theta_180[theta_idx])
    else:
        i+=1


print(theta_180)
print(theta_loc)
print(theta_order)