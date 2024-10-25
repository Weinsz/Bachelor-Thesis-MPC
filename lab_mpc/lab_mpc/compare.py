import pandas as pd
import sys
import numpy as np
import math

map = sys.argv[1] if len(sys.argv) > 1 else "monza"
profile = sys.argv[2]
original = pd.read_csv("/home/weins/sim_ws/csv/traj_race_" + map + "_v2.csv", names=['x', 'y', 'theta', 'speed'])
original['accel'] = 0.0
print("Reference CSV:\n", original.head())
new = pd.read_csv("/home/weins/sim_ws/csv/mpc_" + map + "_" + profile + "_controls_out.csv")#.iloc[:, 1]
print("Actual CSV:\n", new.head())
cond = new['speed'] != 0.0
speed_mu = new[cond]['speed'].mean()
res = new[cond] - original[cond]
print("\nActual controls - Original Data controls:")
res['dist'] = ((new.x - original.x)**2 + (new.y - original.y)**2)**0.5
res_dist = new[cond].copy()
res_dist['dist'] = res['dist']
res['energy'] = 3.74 * new['accel'] * res['speed']
res_dist['energy'] = res['energy']
res_dist['theta_map'] = original[cond]['theta']
res_dist.to_csv("/home/weins/sim_ws/csv/mpc_" + map + "_" + profile + "_final_out.csv", index=False)
speed = new[cond]['speed'] - original[cond]['speed']
print(res.head())
print("\nDescribe with infos:")
print(res.describe())

res.energy = res.energy.apply(lambda x: x if x > 0.0 else 0.0)
#print(res[res.speed == 0.0]['dist'])
print("\nActual controls - Original Data controls:\n", res.head(10))
print("\nNum cases of speed's underperformance: {} cases, {:.2%}".format(len(res[speed < 0.0]), len(res[speed < 0.0])/len(res)))
print("Num cases of speed's overperformance: {} cases, {:.2%}".format(len(res[speed > 0.0]), len(res[speed > 0.0])/len(res)))
MSE_tracking = sum(res['dist']**2)/len(res)
print(f"Mean Crosstrack Error: {sum(res['dist'])/len(res):.5f} m")
print(f"Mean Squared Crosstrack Error: {MSE_tracking:.5f} m^2")
print(f"Root Mean Squared Crosstrack Error: {math.sqrt(MSE_tracking):.5f} m")
var = res['dist'].var()
print(f"Variance: {var:.5f} m^2")
print(f"Std: {res['dist'].std():.5f} m")
print(f"Mean Energy: {res['energy'].mean():.5f} W")
print(f"Mean Speed: {speed_mu:.5f} m/s")
theta_corr = abs(new[cond]['theta'])
print(theta_corr.head())
print(f"Mean Theta: {theta_corr.mean()*180.0/math.pi:.3f} gradi\nMedian Theta: {theta_corr.median()*180.0/math.pi:.3f} gradi")
print(new.describe())