from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np

TK: int = 7 #6 #5  # finite time horizon length kinematic
DTK: float = 0.03 # 0.05 #0.1  # time step [s] kinematic
MAX_DSTEER: float = float(np.deg2rad(45.0))  # maximum steering speed [rad/s] (i.e. AckermannDrive.steering_angle_velocity)
MAX_SPEED: float = 15.0  # maximum speed [m/s]
MAX_ACCEL: float = 3.0 #10.0   # maximum acceleration [m/ss]

# Weights for tuning

Rk = [0.001, 110.0] #0.005 110.0 good, #0.01
Rdk = [0.001, 110.0]
#np.diag([0.01, 100.0])
#np.diag([0.1, 300.0])
#np.diag([0.1, 400.0])
#np.diag([0.1, 900.0])
#np.diag([0.1, 1000.0])
Qk = [70.0, 70.0, 5.5, 60.0] # 10
#np.diag([13.5, 13.5, 5.5, 13.0])
#np.diag([30.0, 30.0, 15.0, 30.0])
#np.diag([40.0, 40.0, 20.0, 40.0])
#np.diag([60.0, 60.0, 30.0, 60.0])
#np.diag([90.0, 90.0, 45.0, 90.0])
#np.diag([70.0, 70.0, 50.0, 85.0])

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='lab_mpc',
            executable='mpc',
            name='MPC',
            output='screen',
            emulate_tty=True,
            parameters=[{
                'TK': TK,
                "DTK": DTK,
                "MAX_DSTEER": MAX_DSTEER,
                "MAX_SPEED": MAX_SPEED,
                "MAX_ACCEL": MAX_ACCEL,
                "Rk": Rk,
                "Rdk": Rdk,
                "Qk": Qk,
                "Qfk": Qk,
                'map': '/home/weins/sim_ws/csv/traj_race_spa_v2.csv', # '/home/weins/sim_ws/csv/traj_race_spa_v2.csv',
                "fast_start": True,
            }],
            arguments=[
                "spa",
                "hp"
            ]
        ),
    ])
