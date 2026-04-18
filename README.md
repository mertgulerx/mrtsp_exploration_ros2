# mrtsp_exploration_ros2

ROS 2 Jazzy frontier exploration package implementing the method from:

- Chunyang Liu et al., "Enhancing autonomous exploration for robotics via real time map optimization and improved frontier costs", Scientific Reports (2025)
- DOI: [10.1038/s41598-025-97231-9](https://doi.org/10.1038/s41598-025-97231-9)

## Overview

This package provides a standalone ROS 2 explorer node that:

- subscribes to `/map` and `/global_costmap/costmap`
- optimizes the map in real time using bilateral filtering and dilation
- extracts frontiers using WFD on the optimized map
- computes frontier costs using the paper's equations
- builds an MRTSP-style cost matrix
- dispatches the first frontier from a greedy MRTSP ordering to Nav2 using `NavigateToPose`

The package is intentionally independent and can be reused in any Nav2-based stack.

## Public ROS Interfaces

- Subscribes:
  - `map_topic` as `nav_msgs/OccupancyGrid`
  - `costmap_topic` as `nav_msgs/OccupancyGrid`
- Uses TF:
  - `global_frame -> robot_base_frame`
- Sends goals with:
  - `nav2_msgs/action/NavigateToPose`
- Optional debug publishers:
  - `/explore/frontiers` as `visualization_msgs/MarkerArray`
  - `/explore/selected_frontier` as `geometry_msgs/PoseStamped`
  - `/explore/optimized_map` as `nav_msgs/OccupancyGrid`

## Formula Interpretation

The paper contains visible typesetting errors in the bilateral filter equations. This package uses the academically corrected forms:

- Spatial kernel:
  - `G_sigma_s(||p-q||) = exp(-((i-x)^2 + (j-y)^2) / (2 sigma_s^2))`
- Range kernel:
  - `G_sigma_r(|I(p)-I(q)|) = exp(-((I(p)-I(q))^2) / (2 sigma_r^2))`
- Bilateral normalization:
  - `W_p = sum_q G_sigma_s G_sigma_r`
- Filtered value:
  - `I_bar(p) = 1/W_p * sum_q G_sigma_s G_sigma_r I(q)`

The path cost and MRTSP terms are implemented directly from the paper:

- `d(V_i, V_j) = max(d_m + d_u, d_n + d_v) - r_s`
- `P(V_i, V_j) = S`
- `M_i,j` and `t_lb(V_i, V_j)` match the corrected latex interpretation used during implementation.

## Parameters

Core parameters are in `config/params.yaml`.

Important ones:

- `map_topic`
- `costmap_topic`
- `local_costmap_topic`
- `navigate_to_pose_action_name`
- `global_frame`
- `robot_base_frame`
- `sigma_s`
- `sigma_r`
- `dilation_kernel_radius_cells`
- `sensor_effective_range_m`
- `weight_distance_wd`
- `weight_gain_ws`
- `max_linear_speed_vmax`
- `max_angular_speed_wmax`
- `occ_threshold`
- `min_frontier_size_cells`
- `publish_debug_topics`
- `goal_preemption_on_frontier_opened`
- `goal_preemption_on_blocked_goal`
- `goal_preemption_min_interval_s`
- `goal_preemption_skip_if_within_m`
- `return_to_start_on_complete`

`goal_preemption_on_frontier_opened` and `goal_preemption_on_blocked_goal` are intentionally
disabled by default so the package keeps paper-like behavior unless the integrator explicitly opts
into online frontier revalidation. When enabled, the node can independently cancel the active goal
if the target frontier has already opened or if the goal point becomes blocked in the available
global/local costmaps.

When a replacement frontier is available, preemption is sent as a replacement
`NavigateToPose` goal (without a mandatory cancel-then-resend round trip). Explicit
cancel is reserved for cases where the active goal becomes invalid and no safe
replacement frontier exists.

When no frontiers remain, the node now announces exploration completion once and stops replanning.
`return_to_start_on_complete` is disabled by default for package neutrality, but can be enabled by
integrators that want the robot to navigate back to its recorded start pose after exploration ends.

## Build

```bash
source /opt/ros/jazzy/setup.bash
colcon build --packages-select mrtsp_exploration_ros2
source install/setup.bash
```

## Run

```bash
ros2 launch mrtsp_exploration_ros2 explore.launch.py
```

Or with an explicit params file:

```bash
ros2 launch mrtsp_exploration_ros2 explore.launch.py \
  params_file:=/path/to/params.yaml
```

## Package Structure

- `mrtsp_exploration_ros2/occupancy_grid.py`: occupancy grid helpers
- `mrtsp_exploration_ros2/map_optimizer.py`: bilateral filtering and dilation
- `mrtsp_exploration_ros2/wfd_frontier.py`: WFD frontier extraction
- `mrtsp_exploration_ros2/frontier_model.py`: frontier dataclasses
- `mrtsp_exploration_ros2/frontier_costs.py`: path/gain/MRTSP cost equations
- `mrtsp_exploration_ros2/mrtsp_solver.py`: greedy frontier ordering
- `mrtsp_exploration_ros2/explorer_node.py`: ROS 2 node and Nav2 action client
- `mrtsp_exploration_ros2/visualization.py`: debug marker generation

