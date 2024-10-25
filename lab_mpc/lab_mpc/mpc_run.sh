#!/bin/bash
if [ -z "$1" ]; then
	echo "Select a map"
	exit 1
fi

if [ -z "$2" ]; then
	echo "Select slow or fast mode"
	exit 1
fi

konsole --new-tab -e /bin/bash -c \
"cd ~/sim_ws && \
	source install/setup.bash && \
	ros2 launch f1tenth_gym_ros gym_bridge_launch.py" &

echo "waiting for rviz to start..."
sleep 5

konsole --new-tab --hold -e /bin/bash -c \
"cd ~/sim_ws && \
	source install/setup.bash && \
	ros2 launch lab_mpc mpc_${1}_${2}_launch.py"