## Project Overview

This project implements a reinforcement learning (RL) approach for automatic quadrilateral mesh generation. The goal is to train an RL agent that can intelligently place vertices to create a high-quality mesh within a given boundary. The project is based on the papers by Pan et al. (2021, 2023), which propose a self-learning finite element extraction system based on reinforcement learning.

## Reinforcement Learning (`rl`)

The core of the project is the `rl` component, which is responsible for training the mesh generation agent. The main script for this is `rl/baselines/RL_Mesh.py`. This script uses the `stable-baselines3` library to implement various reinforcement learning algorithms, including A2C, DDPG, PPO, SAC, and TD3.

### Environment

The environment for the reinforcement learning agent is defined in `rl/boundary_env.py`. The `BoudaryEnv` class inherits from `gym.Env` and represents the mesh generation environment. The agent's goal is to generate a valid mesh for a given boundary.

### State Space

The state space is defined by the `PointEnvironment` class in `general/components.py`. It represents the local environment of a reference point on the boundary. The state is a vector of `2 * (neighbor_num + radius_num)` floating-point numbers, where `neighbor_num` is the number of neighboring vertices and `radius_num` is the number of radius neighbors. The state includes the following information for each neighbor:

-   **Distance**: The distance from the reference point to the neighbor.
-   **Angle**: The angle of the neighbor with respect to the reference point.

### Action Space

The action space is a continuous space with three values: `[rule_type, radius, angle]`.

-   `rule_type`: Determines the type of mesh generation rule to apply. This is a continuous value in the range `[-1, 1]`.
    -   `rule_type <= -0.5`: Apply rule -1, which creates a mesh from the four vertices to the left of the reference point.
    -   `rule_type >= 0.5`: Apply rule 1, which creates a mesh from the four vertices to the right of the reference point.
    -   `-0.5 < rule_type < 0.5`: Apply rule 0, which creates a new vertex and forms a mesh with the three vertices centered on the reference point.
-   `radius`: The radius of the new vertex to be generated, as a multiplier of the base length. This is a continuous value in the range `[-1.5, 1.5]`.
-   `angle`: The angle of the new vertex to be generated, in radians. This is a continuous value in the range `[0, 1.5]`.

The action `a` is transformed into a new point `p_new` using the following transformation:

`x = radius * cos(angle)`
`y = radius * sin(angle)`
`p_new = detransformation([x, y])`

where `detransformation` is a function that converts the point from the local coordinate system of the reference point to the global coordinate system.

### Reward Function

The reward function is defined in the `step` method of the `BoudaryEnv` class. The reward `R` is calculated as follows:

`R = Q_mesh + P_speed`

where:

-   `Q_mesh` is the quality of the generated mesh, calculated as:
    `Q_mesh = sqrt(Q_element * Q_boundary)`
    -   `Q_element` is the element quality, which is a measure of how well-shaped the generated quadrilateral element is. It is calculated using the following formula:
        `Q_element = 1 / (aspect_ratio + ave_error_angle)`
        -   `aspect_ratio` is the ratio of the longest edge to the shortest edge of the quadrilateral.
        -   `ave_error_angle` is the average absolute difference between the angles of the quadrilateral and 90 degrees.
    -   `Q_boundary` is the boundary quality, which is a measure of how well the new element fits with the existing boundary. It is calculated based on the angles and lengths of the new boundary segments. The formula for boundary quality is:
        `Q_boundary = (2 * min(angles)) / pi * sqrt(min(mean_dist, targt_len) / max(mean_dist, targt_len))`
        - `angles` are the new angles formed at the boundary.
        - `mean_dist` is the mean of the lengths of the new boundary segments.
        - `targt_len` is the target length for the new boundary segments.

-   `P_speed` is a penalty for the speed of the mesh generation, calculated as:
    `P_speed = (A_mesh - A_critical) / (A_critical - A_min)`
    -   `A_mesh` is the area of the generated mesh.
    -   `A_critical` is the critical area, which is the maximum desired area for a single element.
    -   `A_min` is the minimum desired area for a single element.

A penalty of -1 is applied for invalid actions, such as generating a mesh that intersects with the boundary or is not a valid quadrilateral.

### Training

The training process is managed by the `mesh_learning` function in `rl/baselines/RL_Mesh.py`. This function takes the RL method, the environment, and the total number of timesteps as input. It then trains the agent and saves the trained model.

The project also implements curriculum learning in the `curriculum_learning` function. This function trains the agent on a sequence of increasingly complex domains.

## Code Structure

The project is organized into the following main directories:

-   `general`: Contains the core data structures and components for mesh generation.
-   `rl`: Contains the reinforcement learning environment and training scripts.
-   `ui`: Contains the user interface for visualizing the mesh generation process.

### Key Files

-   `rl/baselines/RL_Mesh.py`: The main script for training the reinforcement learning agent.
-   `rl/boundary_env.py`: Defines the mesh generation environment.
-   `general/components.py`: Defines the data structures for representing the mesh and its components.
-   `general/mesh.py`: Contains the core mesh generation logic.
-   `ui/gui.py`: The main script for the user interface.

## Data Flow

1.  **Boundary Definition**: The process starts with a boundary defined in a JSON file (e.g., in the `ui/domains` directory).
2.  **RL Training**: The reinforcement learning agent is trained on the defined boundary. The agent learns to generate a valid mesh by interacting with the environment.
3.  **Sample Extraction**: During RL training, samples are extracted from the generated meshes. These samples consist of the state, the action, and the reward.
4.  **Mesh Generation**: The trained reinforcement learning agent is used to generate the final mesh for the given boundary.

## Annotations

-   The `BoudaryEnv` class is a misspelling of `BoundaryEnv`.
-   The code contains several hardcoded paths, which should be replaced with relative paths or configuration variables.
-   The project uses a custom A2C implementation in `rl/baselines/CustomizeA2C.py`, which disables orthogonal initialization.