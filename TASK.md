Create write a bash or python to automatic data collection running with isaacsim in folder scripts/collect_data. 
Run the following each terminal 1-2-3-4 sequentially. Each terminal should wait until previous is ready.
After 4 scanning finish, close all and start new terminals for next run.

The files allow me to input how many run and start run number. 

## Terminal1: kinova Isaacsim
. ~/isaac_ros.sh 

. ~/Projects/kinova_isaacsim/run_kinova_isaac.sh

success load if they show message "[segmentation_service] Ready: /save_object_segmentations; camera frames: ['handeye_camera_color_optical_frame']"

## Terminal2: Kinova moveit
source /opt/ros/humble/setup.bash; source ~/Projects/kinova_ws/install/local_setup.bash

ros2 launch kinova_gen3_7dof_robotiq_2f_85_moveit_config robot.launch.py   robot_ip:=yyy.yyy.yyy.yyy isaac_sim:=true

## Termial3: motion service
source /opt/ros/humble/setup.bash; source ~/Projects/kinova_ws/install/local_setup.bash

ros2 launch robot_interfaces robot_interfaces.launch.py \
  start_gripper_server:=false \
  default_planning_group:=manipulator \
  tracking_tip_links:=end_effector_link \
  velocity_scale:=0.5 \
  acceleration_scale:=0.5 

## Terminal4: multi-view Scan
python3 scripts/path_planning_single/single_view_scan.py --use-sim-time