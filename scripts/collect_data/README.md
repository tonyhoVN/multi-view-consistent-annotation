# Automated Kinova data collection

`collect_kinova_scans.py` runs the four required processes in separate process
groups and waits for each dependency before starting the next:

1. Kinova Isaac Sim, gated by the exact segmentation-service ready message.
2. Kinova MoveIt, gated by `/compute_ik` discovery and one live `/joint_states`
   message.
3. `robot_interfaces`, gated by `/move_cartesian` discovery.
4. `single_view_scan.py`, which runs to completion.

After a scan exits, all four process groups receive `SIGINT` in reverse order.
Processes that do not exit within the shutdown timeout receive `SIGTERM`, then
`SIGKILL`. The next run starts a fresh Isaac scene after `restart_delay`.

## Usage

Run five trials numbered 10 through 14:

```bash
python3 scripts/collect_data/collect_kinova_scans.py \
  --runs 5 \
  --start-run 10
```

The default paths are:

```text
~/Projects/kinova_ws
~/Projects/kinova_isaacsim
~/isaac_ros.sh
```

Terminal 1 clears inherited ROS/colcon Python and library paths before sourcing
`~/isaac_ros.sh`. This prevents ROS Humble's Python 3.10 `rclpy` extension from
being imported by Isaac Sim's Python 3.12 runtime. It then changes to the Isaac
project directory before executing `run_kinova_isaac.sh`, because the simulation
loads project resources relative to its working directory. Terminals 2–4
explicitly source ROS Humble and the Kinova workspace afterward in their own
environments.

Before terminal 3 starts, the supervisor requires both
`joint_state_broadcaster` and `joint_trajectory_controller` to report `active`.
RViz is disabled for unattended collection. During teardown, Ctrl-C is sent to
the ROS launch parent so it can shut down its child nodes in launch order instead
of interrupting every MoveIt plugin process simultaneously.

The default simulated robot IP placeholder is `0.0.0.0`; override it if the
launch configuration requires another value:

```bash
python3 scripts/collect_data/collect_kinova_scans.py \
  --runs 5 --start-run 10 --robot-ip 192.168.1.10
```

Each run receives a prefix-scoped name such as `run_10`. All scan products,
logs, annotations, and evaluation reports for that run live together:

```text
scan_output/run_10/
├── collection_log/
│   ├── isaac.log
│   ├── moveit.log
│   ├── motion.log
│   ├── scan.log
│   └── run.json
├── save_images/
├── save_segment/
├── save_TF/
├── baseline_segment/
│   ├── naive_vlm_zeroshot/
│   └── naive_vlm_multi_shot/
└── manifest.json
```

Migrate legacy runs without overwriting existing canonical artifacts:

```bash
python3 scripts/collect_data/migrate_scan_layout.py 1 24 --dry-run
python3 scripts/collect_data/migrate_scan_layout.py 1 24
```

By default, a failed run is shut down and the next run is attempted. Add
`--stop-on-error` to stop after the first failure. Inspect commands without
starting Isaac Sim or ROS:

```bash
python3 scripts/collect_data/collect_kinova_scans.py \
  --runs 2 --start-run 1 --dry-run
```
