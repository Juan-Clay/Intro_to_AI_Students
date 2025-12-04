# inhand_env.py (Final Corrected Version)
import os
import numpy as np
import mujoco
import mujoco.viewer as mjv
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R
from simulation import Simulation

MAX_EPISODE_STEPS = 300

class CanRotateEnv(gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(CanRotateEnv, self).__init__()
        
        # Initialize simulation and get object IDs
        self.sim = Simulation(
            scene_path=os.path.join(os.path.dirname(__file__), "scene.xml"),
            output_dir="rl_output"
        )
        self.sim.load()
        self.obj_body_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_BODY, "obj1") #
        self.site_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site") #
        self.sim.ids_by_name(["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"], mujoco.mjtObj.mjOBJ_JOINT, 'arm') #
        self.sim.ids_by_name(["1", "0", "2", "3", "5", "4", "6", "7", "9", "8", "10", "11", "12", "13", "14", "15"], mujoco.mjtObj.mjOBJ_JOINT, 'hand') #
        self.sim.actuators_for_joints('arm') #
        self.sim.actuators_for_joints('hand') #
        self.can_geom_id = mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "obj1")
        self.fingertip_geom_ids = {
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip"),
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "fingertip_2"),
            mujoco.mj_name2id(self.sim.model, mujoco.mjtObj.mjOBJ_GEOM, "thumb_fingertip"),
        }
        
        # Define action and observation spaces
        self.action_space = spaces.Box(low=-0.03, high=0.03, shape=(16,), dtype=np.float32)
        obs_size = len(self.sim.hand_joint_ids) + 7
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)

        self.render_mode = render_mode
        if render_mode == 'human':
            self.viewer = mjv.launch_passive(self.sim.model, self.sim.data) 
        else:
            self.viewer = None
        self.step_count = 0

    def _get_obs(self):
        finger_qpos = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) #
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id] #
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr] #
        object_pose = self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 7] #
        return np.concatenate([finger_qpos, object_pose])

    def _calculate_reward(self):
        # --- Rotation and Survival Rewards ---
        obj_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.sim.model, self.sim.data, mujoco.mjtObj.mjOBJ_BODY, self.obj_body_id, obj_vel, 0)
        angular_velocity_z = obj_vel[2]
        rotation_reward = angular_velocity_z * 10.0
        
        can_pos = self.sim.data.xpos[self.obj_body_id]
        palm_pos = self.sim.data.site_xpos[self.site_id]
        distance_from_palm = np.linalg.norm(can_pos - palm_pos)
        survival_reward = 0.1 - distance_from_palm

        # --- Contact Reward ---
        contact_reward = 0.0
        
        # Use a set to count unique fingers touching the can
        fingers_in_contact = set()

        # Iterate through all contacts in the current simulation step
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check if one of the geometries is the can and the other is a fingertip
            is_geom1_fingertip = geom1 in self.fingertip_geom_ids
            is_geom2_fingertip = geom2 in self.fingertip_geom_ids
            is_geom1_can = geom1 == self.can_geom_id
            is_geom2_can = geom2 == self.can_geom_id

            if (is_geom1_fingertip and is_geom2_can):
                fingers_in_contact.add(geom1)
            elif (is_geom2_fingertip and is_geom1_can):
                fingers_in_contact.add(geom2)
        
        # Give a bonus if three specified fingers are touching the can
        if len(fingers_in_contact) >= 3:
            contact_reward = 5.0  # Large, one-time bonus for achieving the grasp
        elif len(fingers_in_contact) > 0:
            contact_reward = 0.5 * len(fingers_in_contact) # Smaller reward for partial contact
            
        # Combine all reward components
        total_reward = rotation_reward + survival_reward + contact_reward
        return total_reward

    def _is_terminated(self):
        can_z_pos = self.sim.data.xpos[self.obj_body_id][2] #
        palm_z_pos = self.sim.data.site_xpos[self.site_id][2] #
        return can_z_pos < (palm_z_pos - 0.05) or self.step_count > MAX_EPISODE_STEPS

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        
        mujoco.mj_resetData(self.sim.model, self.sim.data)

        target_pos_up = np.array([0.4, 0.0, .5]) #
        target_euler_up = np.array([0, 0, 0]) #
        q_palm_up = self.sim.desired_qpos_from_ik(self.site_id, target_pos_up, target_euler_up) #
        self.sim.set_joint_positions(self.sim.arm_joint_ids, q_palm_up) #
        for i, act_id in enumerate(self.sim.arm_act_ids):
            self.sim.data.ctrl[act_id] = q_palm_up[i] #
        
        mujoco.mj_forward(self.sim.model, self.sim.data)

        palm_surface_pos = self.sim.data.site_xpos[self.site_id].copy() #
        object_start_pos = palm_surface_pos + np.array([0.011, -0.03, 0.075]) #
        obj_jnt_adr = self.sim.model.body_jntadr[self.obj_body_id] #
        obj_qpos_adr = self.sim.model.jnt_qposadr[obj_jnt_adr] #
        self.sim.data.qpos[obj_qpos_adr : obj_qpos_adr + 3] = object_start_pos #
        self.sim.data.qpos[obj_qpos_adr + 3 : obj_qpos_adr + 7] = [1, 0, 0, 0] #

        mujoco.mj_forward(self.sim.model, self.sim.data)

        for _ in range(20):
            mujoco.mj_step(self.sim.model, self.sim.data) #

        q_open_angles = np.array([1.0, 0.3, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.3, 1.0, 1.3, 1.0, 0.8, 1.3, 0.8, 0.5]) #
        self.sim.set_joint_positions(self.sim.hand_joint_ids, q_open_angles) #
        for i, act_id in enumerate(self.sim.hand_act_ids):
            self.sim.data.ctrl[act_id] = q_open_angles[i] #

        mujoco.mj_forward(self.sim.model, self.sim.data)

        if self.render_mode != "headless":
            self.viewer.sync()
        
        return self._get_obs(), {}

    def step(self, action):
        target_angles = np.array([self.sim.data.qpos[self.sim.model.jnt_qposadr[j]] for j in self.sim.hand_joint_ids]) + action
        self.sim.move_gripper_to_angles(target_angles, 0.5) #

        if self.render_mode != "headless":
            self.viewer.sync()

        self.step_count += 1
        
        observation = self._get_obs()
        reward = self._calculate_reward()
        terminated = self._is_terminated()
        truncated = self.step_count >= MAX_EPISODE_STEPS

        return observation, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data)
            
            # Check if the viewer is still active before trying to sync
            try:
                if self.viewer.is_running():
                    self.viewer.sync()
                else:
                    # If the user closed the window, we must handle it
                    self.close() # Properly close the viewer resources
                    self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data) # And re-launch it
            except Exception:
                # This can happen if the viewer was closed abruptly
                self.viewer = mujoco.viewer.launch(self.sim.model, self.sim.data)
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


class DiscreteCanRotateEnv(gym.Env):
    """
    Wrapper around CanRotateEnv that:
      - Uses a discrete state: cube yaw angle binned into N bins
      - Uses a small set of discrete hand actions (16D delta joint angles)
    This is the environment used for the Q-learning + Q-table agent.
    """

    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super().__init__()
        # Underlying continuous environment
        self.base_env = CanRotateEnv(render_mode=render_mode)

        # --- Discrete state: yaw angle bins ---
        self.n_yaw_bins = 24
        self.bin_size_deg = 360.0 / self.n_yaw_bins
        self.observation_space = spaces.Discrete(self.n_yaw_bins)

        # --- Discrete actions: hand motion primitives ---
        self.discrete_actions = self._init_actions()
        self.action_space = spaces.Discrete(len(self.discrete_actions))

        self.render_mode = render_mode

    def _init_actions(self):
        """
        Define a small set of 16D action vectors that move the hand in
        different ways. These are deltas added to the current joint angles.
        """
        # All fingers close / open
        close_all = np.full(16, 0.02, dtype=np.float32)
        open_all = -close_all

        # First half close, second half open (and inverse)
        close_first_open_second = np.concatenate(
            [np.full(8, 0.02, dtype=np.float32),
             np.full(8, -0.02, dtype=np.float32)]
        )
        open_first_close_second = -close_first_open_second

        # Alternating pattern (and inverse)
        alt_pattern = np.array(
            [0.02 if i % 2 == 0 else -0.02 for i in range(16)],
            dtype=np.float32
        )
        alt_pattern_inv = -alt_pattern

        return [
            close_all,
            open_all,
            close_first_open_second,
            open_first_close_second,
            alt_pattern,
            alt_pattern_inv,
        ]

    def _get_yaw_deg(self):
        """
        Compute the object's yaw (Z rotation in degrees) from the MuJoCo quaternion.
        """
        quat_wxyz = self.base_env.sim.data.xquat[self.base_env.obj_body_id]
        # Convert (w, x, y, z) -> (x, y, z, w)
        quat_xyzw = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )
        r = R.from_quat(quat_xyzw)
        euler_deg = r.as_euler("xyz", degrees=True)
        return float(euler_deg[2])

    def _obs_to_state(self):
        """
        Map the continuous yaw angle into a discrete bin index [0, n_yaw_bins-1].
        """
        yaw = self._get_yaw_deg()
        yaw_wrapped = (yaw + 360.0) % 360.0
        bin_idx = int(yaw_wrapped // self.bin_size_deg)
        bin_idx = int(np.clip(bin_idx, 0, self.n_yaw_bins - 1))
        return bin_idx

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        # We ignore the continuous obs and return the discrete state index instead
        state = self._obs_to_state()
        return state, info

    def step(self, action):
        """
        action: integer index into self.discrete_actions
        """
        continuous_action = self.discrete_actions[int(action)]
        obs, reward, terminated, truncated, info = self.base_env.step(continuous_action)
        state = self._obs_to_state()
        return state, reward, terminated, truncated, info

    def render(self):
        # Delegate to the base env's render (mainly for human mode)
        return self.base_env.render()

    def close(self):
        self.base_env.close()