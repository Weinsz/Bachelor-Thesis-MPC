from launch import LaunchDescription
from launch_ros.actions import Node
import numpy as np

TK: int = 5 #6 #5  # finite time horizon length kinematic
DTK: float = 0.03 # 0.05 #0.1  # time step [s] kinematic
MAX_DSTEER: float = float(np.deg2rad(90.0))  # maximum steering speed [rad/s] (i.e. AckermannDrive.steering_angle_velocity)
MAX_SPEED: float = 15.0  # maximum speed [m/s]
MAX_ACCEL: float = 3.0 #10.0   # maximum acceleration [m/ss]

# Weights for tuning

Rk = [0.001, 90.0] # Instead, without first wp's speed: ([0.001, 90.0])
#np.diag([0.01, 100.0])
Rdk = [0.1, 600.0] # without first wp's speed: ([0.1, 500.0])
#np.diag([0.01, 100.0])
#np.diag([0.1, 300.0])
#np.diag([0.1, 400.0])
#np.diag([0.1, 500.0])
#np.diag([0.1, 900.0])
#np.diag([0.1, 1000.0])
Qk = [70.0, 70.0, 30.0, 70.0]
Qfk = [70.0, 70.0, 30.0, 70.0]
#np.diag([13.5, 13.5, 5.5, 13.0])
#np.diag([30.0, 30.0, 15.0, 30.0])
#np.diag([40.0, 40.0, 20.0, 40.0])
#np.diag([60.0, 60.0, 30.0, 60.0]) this without first wp's speed
#np.diag([70.0, 70.0, 30.0, 70.0])
#np.diag([70.0, 70.0, 35.0, 70.0])

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
                "Qfk": Qfk,
                'map': '/home/weins/sim_ws/csv/traj_race_spa_v2.csv',
                "fast_start": True,
            }],
            arguments=[
                "spa",
                "fast"
            ]
        ),
    ])
