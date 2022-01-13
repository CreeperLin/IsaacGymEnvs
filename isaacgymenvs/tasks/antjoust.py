# Copyright (c) 2018-2021, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from typing import Dict, Any, Tuple

import math
import numpy as np
import os
import torch
from torch import Tensor

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.gymtorch import *

from isaacgymenvs.utils.torch_jit_utils import *
from .base.vec_task import VecTask


class AntJoust(VecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):

        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.num_teams = self.cfg["env"].get("numTeams", 1)
        self.num_agents_team = self.cfg["env"].get("numAgentsPerTeam", 1)
        self.num_agents = self.num_agents_team * self.num_teams
        self.cfg["env"]["numAgents"] = self.num_agents
        self.num_agents_export = self.cfg["env"].get("numAgentsExport", self.num_teams)
        print('actors', self.num_agents)
        num_obs_self = 60
        num_obs_per_rng = 28
        self.max_num_obs_rng = min(self.cfg["env"].get("numObsRangeAgents", 1), self.num_agents - 1)
        num_obs = num_obs_self + num_obs_per_rng * self.max_num_obs_rng
        num_acts = 8
        self.num_space_parts = self.cfg["env"].get("numSpacePartitions", 1)
        self.ma_supported = self.cfg["env"].get("maSupported", True)
        space_mult = self.num_agents // self.num_space_parts // (self.num_agents_export if self.ma_supported else 1)
        self.num_obs_per_agent = num_obs
        self.num_acts_per_agent = num_acts
        self.cfg["env"]["numObservations"] = num_obs * space_mult
        self.cfg["env"]["numActions"] = num_acts * space_mult
        self.reward_sum = self.cfg["env"].get("rewardSum", 'team')
        if self.reward_sum == 'team':
            value_size = self.num_teams
        elif self.reward_sum == 'all':
            value_size = 1
        elif self.reward_sum in [None, 'none']:
            value_size = self.num_agents
        self.value_size = value_size
        self.value_size_export = self.value_size // self.num_agents_export

        self._act_part_pt = 0

        super().__init__(
            config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless
        )

        # self.actions = torch.zeros((self.num_envs * self.num_agents_export, self.num_actions),
        #                            device=self.device, dtype=torch.float)
        self.terminated_buf = torch.zeros((self.num_envs, self.num_agents), device=self.device, dtype=torch.bool)

        print('dof', self.num_dof)
        print('dof limit', self.dof_limits_lower)
        # self.num_dof = self.num_dof * self.num_agents
        if self.viewer is not None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        print('dof state', dof_state_tensor.shape)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        # sensors_per_env = 4
        sensors_per_env = 4 * self.num_agents
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        # self.initial_root_states = self.root_states.clone()
        # self.initial_root_states[:, 7:13] = 0  # set lin_vel and ang_vel to 0
        self.initial_root_states = self.root_states.clone().view(self.num_envs, self.num_agents, -1)
        self.initial_root_states[:, :, 7:13] = 0  # set lin_vel and ang_vel to 0

        print('root', self.root_states.shape)  # [n_env * n_agt, 13]

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        # self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        # self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof * self.num_agents, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof * self.num_agents, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(
            self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
            torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos)
        )
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        # self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        # self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        # self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx),
                               device=self.device).repeat((self.num_envs * self.num_agents, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs * self.num_agents, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs * self.num_agents, 1))

        # self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx),
        #    device=self.device).repeat((self.num_envs, self.num_agents, 1))
        # self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, self.num_agents, 1))
        # self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, self.num_agents, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        # self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, self.num_agents, 1))
        # self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        # self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs, self.num_agents)
        self.prev_potentials = self.potentials.clone()

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        self.obs_buf = torch.zeros(
            # (self.num_envs * self.num_agents_export, self.num_obs), device=self.device, dtype=torch.float)
            (self.num_envs * self.num_agents, self.num_obs_per_agent), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        if self.value_size_export == 1:
            rew_size = self.num_envs * self.num_agents_export
        else:
            rew_size = (self.num_envs * self.num_agents_export, self.value_size_export)
        self.rew_buf = torch.zeros(
            rew_size, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
            # (self.num_envs), device=self.device, dtype=torch.long)
            (self.num_envs * self.num_agents_export), device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            # (self.num_envs), device=self.device, dtype=torch.long)
            (self.num_envs * self.num_agents_export), device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        # actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)
        actions = torch.zeros(
            (self.num_envs * self.num_agents_export, self.num_actions), device=self.rl_device, dtype=torch.float
        )

        return actions

    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # cam_pos = [20.0, 25.0, 3.0]
        # cam_target = [10.0, 15.0, 0.0]
        cam_pos = [0, 0, 3.0]
        cam_target = [1.0, 1.5, 0.0]

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis != gymapi.UP_AXIS_Z:
                cam_pos = [cam_pos[0], cam_pos[2], cam_pos[1]]
                cam_target = [cam_target[0], cam_target[2], cam_target[1]]

            self.gym.viewer_camera_look_at(
                self.viewer, None, gymapi.Vec3(*cam_pos), gymapi.Vec3(*cam_target))

    def create_sim(self):
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, 'z')
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        print(f'num envs {self.num_envs} env spacing {self.cfg["env"]["envSpacing"]}')
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_ant.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.angular_damping = 0.0

        ant_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        # self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        # self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        self.num_dof = self.gym.get_asset_dof_count(ant_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset) * self.num_agents

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(ant_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        self.joint_gears = to_torch(motor_efforts, device=self.device).repeat(self.num_agents)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))

        self.start_rotation = torch.tensor(
            [start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(ant_asset)
        body_names = [self.gym.get_asset_rigid_body_name(ant_asset, i) for i in range(self.num_bodies)]
        extremity_names = [s for s in body_names if "foot" in s]
        self.extremities_index = torch.zeros(len(extremity_names), dtype=torch.long, device=self.device)

        # create force sensors attached to the "feet"
        extremity_indices = [self.gym.find_asset_rigid_body_index(ant_asset, name) for name in extremity_names]
        sensor_pose = gymapi.Transform()
        for body_idx in extremity_indices:
            self.gym.create_asset_force_sensor(ant_asset, body_idx, sensor_pose)

        self.ant_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        colors = [
            (0.97, 0.38, 0.06),
            (0.38, 0.06, 0.97),
            (0.06, 0.97, 0.38),
        ]

        # pos = 10 + torch.rand(self.num_envs, self.num_agents, 3)
        # pos *= torch.sign(torch.randn_like(pos))
        # pos = torch.ones(self.num_envs, self.num_agents, 3)
        init_pos_radius = self.num_agents * 1.
        pos = torch.arange(0, self.num_agents) * 2. * math.pi / self.num_agents
        pos = pos.repeat(3, self.num_envs).T.view(self.num_envs, self.num_agents, -1)
        pos[:, :, 0].cos_()
        pos[:, :, 1].sin_()
        pos.mul_(init_pos_radius)
        pos += torch.randn(self.num_envs, self.num_agents, 3)
        pos[:, :, self.up_axis_idx] = 0
        # print(pos)

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            for k in range(self.num_agents):
                start_pose = gymapi.Transform()
                start_pose.p = gymapi.Vec3(*get_axis_params(0.44, self.up_axis_idx))
                start_pose.p += gymapi.Vec3(*pos[i][k].tolist())
                # ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "ant", i, 1, 0)
                ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "ant", i, 0, 0)
                # ant_handle = self.gym.create_actor(env_ptr, ant_asset, start_pose, "ant", 0, 1, 0)
                self.ant_handles.append(ant_handle)

                for j in range(self.num_bodies):
                    self.gym.set_rigid_body_color(
                        env_ptr, ant_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(*colors[k // self.num_agents_team]))

            self.envs.append(env_ptr)

        dof_prop = self.gym.get_actor_dof_properties(env_ptr, ant_handle)
        for _ in range(self.num_agents):
            for j in range(self.num_dof):
                if dof_prop['lower'][j] > dof_prop['upper'][j]:
                    self.dof_limits_lower.append(dof_prop['upper'][j])
                    self.dof_limits_upper.append(dof_prop['lower'][j])
                else:
                    self.dof_limits_lower.append(dof_prop['lower'][j])
                    self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        for i in range(len(extremity_names)):
            self.extremities_index[i] = self.gym.find_actor_rigid_body_handle(
                self.envs[0], self.ant_handles[0], extremity_names[i])

    def compute_reward(self):
        rew_buf, reset_buf, terminated_buf = compute_ant_reward(
            self.obs_buf,
            self.reset_buf,
            self.terminated_buf,
            self.progress_buf,
            self.actions,
            self.up_weight,
            self.heading_weight,
            self.potentials,
            self.prev_potentials,
            self.actions_cost_scale,
            self.energy_cost_scale,
            self.joints_at_limit_cost_scale,
            self.termination_height,
            self.death_cost,
            self.max_episode_length,
            self.num_envs,
            self.num_agents,
            self.num_teams,
            self.value_size,
        )
        self.reset_buf[:], self.terminated_buf[:] = reset_buf, terminated_buf
        if self.num_space_parts > 1:
            rew_buf = rew_buf[:, self._act_part_pt]
        elif not self.ma_supported:
            rew_buf = torch.sum(rew_buf, dim=-1)
        self.rew_buf[:] = rew_buf.view(self.rew_buf.size())

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        # print("Feet forces and torques: ", self.vec_sensor_tensor[0, :])
        # print(self.vec_sensor_tensor.shape)

        obs_buf, potentials, prev_potentials, up_vec, heading_vec = compute_ant_observations(
            self.obs_buf,
            self.terminated_buf,
            self.root_states,
            self.targets,
            self.potentials,
            self.inv_start_rot,
            self.dof_pos,
            self.dof_vel,
            self.dof_limits_lower,
            self.dof_limits_upper,
            self.dof_vel_scale,
            self.vec_sensor_tensor,
            self.actions,
            self.dt,
            self.contact_force_scale,
            self.basis_vec0,
            self.basis_vec1,
            self.up_axis_idx,
            self.num_envs,
            self.num_agents,
            self.num_teams,
            self.max_num_obs_rng,
        )
        self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] =\
            potentials, prev_potentials, up_vec, heading_vec
        if self.num_space_parts > 1:
            obs_buf = obs_buf.view(obs_buf.shape[0], self.num_space_parts, -1)[:, self._act_part_pt, :]
        self.obs_buf[:] = obs_buf

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        # velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)
        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof * self.num_agents), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof * self.num_agents), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions,
                                             self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        ent_ids = (env_ids * self.num_agents).repeat(self.num_agents).view(self.num_agents, -1)
        ent_ids += torch.arange(0, self.num_agents).view(-1, 1).to(device=self.device)
        ent_ids_int32 = ent_ids.flatten().to(dtype=torch.int32)
        # print(env_ids)
        # print(ent_ids_int32)

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(ent_ids_int32), len(ent_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(ent_ids_int32), len(ent_ids_int32))

        # print(self.initial_root_states.shape)
        # sel = slice(env_ids*self.num_agents, (env_ids+1)*self.num_agents)
        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, :, 0:3]
        to_target[:, :, 2] = 0.0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf.view(self.num_envs, -1)[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.terminated_buf[env_ids] = 0
        # print('reset', env_ids, self.terminated_buf)

    def terminate_idx(self, agent_ids):
        pass

    def pre_physics_step(self, actions):
        actions = actions.clone().to(self.device)
        if self.num_space_parts > 1:
            self.actions.zero_()
            self.actions.view(self.actions.shape[0], self.num_space_parts, -1)[:, self._act_part_pt, :] = actions
        else:
            self.actions = actions
        self.actions.view(self.num_envs, self.num_agents, -1)[self.terminated_buf] = 0
        forces = self.actions.view(self.num_envs, -1) * self.joint_gears * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)

    def post_physics_step(self):
        self.progress_buf[
            torch.logical_not(torch.all(self.terminated_buf.view(self.num_envs, self.value_size, -1), dim=-1)).flatten()
        ] += 1
        # self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward()

        if self.num_space_parts > 1:
            self._act_part_pt += 1
            if self._act_part_pt == self.num_space_parts:
                self._act_part_pt = 0

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            points = []
            colors = []
            poses = self.root_states[:, 0:3].cpu().numpy()
            h_vecs = self.heading_vec.cpu().numpy()
            u_vecs = self.up_vec.cpu().numpy()
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                for j in range(self.num_agents):
                    pose = poses[i*self.num_agents+j]
                    h_vec = h_vecs[i*self.num_agents+j]
                    u_vec = u_vecs[i*self.num_agents+j]
                    glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                    points.append([
                        glob_pos.x, glob_pos.y, glob_pos.z,
                        glob_pos.x + 4 * h_vec[0], glob_pos.y + 4 * h_vec[1], glob_pos.z + 4 * h_vec[2]
                    ])
                    colors.append([0.97, 0.1, 0.06])
                    points.append([
                        glob_pos.x, glob_pos.y, glob_pos.z,
                        glob_pos.x + 4 * u_vec[0], glob_pos.y + 4 * u_vec[1], glob_pos.z + 4 * u_vec[2]
                    ])
                    colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * self.num_agents * 2, points, colors)

    # def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
    #     """Step the physics of the environment.

    #     Args:
    #         actions: actions to apply
    #     Returns:
    #         Observations, rewards, resets, info
    #         Observations are dict of observations (currently only one member called 'obs')
    #     """

    #     # randomize actions
    #     if self.dr_randomizations.get('actions', None):
    #         actions = self.dr_randomizations['actions']['noise_lambda'](actions)

    #     action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
    #     # apply actions
    #     self.pre_physics_step(action_tensor)

    #     # step physics and render each frame
    #     for i in range(self.control_freq_inv):
    #         self.render()
    #         self.gym.simulate(self.sim)

    #     # to fix!
    #     if self.device == 'cpu':
    #         self.gym.fetch_results(self.sim, True)

    #     # fill time out buffer
    #     self.timeout_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.timeout_buf), torch.zeros_like(self.timeout_buf))
    #     # compute observations, rewards, resets, ...
    #     self.post_physics_step()

    #     # randomize observations
    #     if self.dr_randomizations.get('observations', None):
    #         self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

    #     self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)

    #     self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    #     # asymmetric actor-critic
    #     if self.num_states > 0:
    #         self.obs_dict["states"] = self.get_state()

    #     return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras

    def get_obs_export(self):
        return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)\
            .view(self.num_envs * self.num_agents_export, self.num_obs)\
            .to(self.rl_device)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs_dict, rew_buf, reset_buf, extras = super().step(actions)
        if self.num_agents_export > 1:
            obs_dict['obs'] = obs_dict['obs'].view(self.num_envs * self.num_agents_export, self.num_obs)
            reset_buf = reset_buf.repeat(self.num_agents_export)
        return obs_dict, rew_buf, reset_buf, extras

    def reset(self) -> torch.Tensor:
        """Reset the environment.
        Returns:
            Observation dictionary
        """
        zero_actions = self.zero_actions()

        # step the simulator
        self.step(zero_actions)

        self.obs_dict["obs"] = self.get_obs_export()

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()

        return self.obs_dict

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_ant_reward(
    obs_buf: Tensor,
    reset_buf: Tensor,
    terminated_buf: Tensor,
    progress_buf: Tensor,
    actions: Tensor,
    up_weight: float,
    heading_weight: float,
    potentials: Tensor,
    prev_potentials: Tensor,
    actions_cost_scale: float,
    energy_cost_scale: float,
    joints_at_limit_cost_scale: float,
    termination_height: float,
    death_cost: float,
    max_episode_length: float,
    num_envs: int,
    num_agents: int,
    num_teams: int,
    value_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:

    # reward from direction headed
    obs_buf = obs_buf.view(num_envs, num_agents, -1)
    z_pos = obs_buf[:, :, 0]
    torso = obs_buf[:, :, 10]
    heading_proj = obs_buf[:, :, 11]
    dof_pos = obs_buf[:, :, 12:20]
    dof_vel = obs_buf[:, :, 20:28]
    heading_weight_tensor = torch.ones_like(heading_proj) * heading_weight
    heading_reward = torch.where(heading_proj > 0.8, heading_weight_tensor, heading_weight * heading_proj / 0.8)
    # print('heading_reward', heading_reward.shape)

    # aligning up axis of ant and environment
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(torso > 0.93, up_reward + up_weight, up_reward)
    # print('up_reward', up_reward.shape)

    # energy penalty for movement
    actions = actions.view(num_envs, num_agents, -1)
    actions_cost = torch.sum(actions ** 2, dim=-1)
    electricity_cost = torch.sum(torch.abs(actions * dof_vel.reshape(num_envs, num_agents, -1)), dim=-1)
    dof_at_limit_cost = torch.sum(dof_pos > 0.99, dim=-1)
    # print('actions_cost', actions_cost.shape)
    # print('electricity_cost', electricity_cost.shape)
    # print('dof_at_limit_cost', dof_at_limit_cost.shape)

    # reward for duration of staying alive
    alive_reward = torch.ones_like(potentials) * 0.5
    progress_reward = potentials - prev_potentials
    # print('progress_reward', progress_reward.shape)
    # print('alive_reward', alive_reward.shape)

    # reward for multi-agents
    # ma_alive_reward =

    total_reward = \
        progress_reward +\
        alive_reward +\
        up_reward +\
        heading_reward +\
        (-actions_cost_scale) * actions_cost +\
        (-energy_cost_scale) * electricity_cost +\
        (-dof_at_limit_cost) * joints_at_limit_cost_scale

    # adjust reward for fallen agents
    terminated = z_pos < termination_height
    total_reward = torch.where(terminated, torch.ones_like(total_reward) * death_cost, total_reward)

    total_reward = torch.where(terminated_buf, torch.zeros_like(total_reward), total_reward)
    # print(total_reward.shape)
    total_reward = torch.sum(total_reward.view(num_envs, value_size, -1), dim=-1).squeeze()

    terminated = torch.logical_or(terminated, terminated_buf)

    # reset agents
    reset_ones = torch.ones_like(reset_buf)
    # reset = torch.where(torch.sum(terminated, dim=-1), reset_ones, reset_buf)
    # no team aggregate for resets
    # reset = torch.where(torch.all(terminated, dim=-1), reset_ones, reset_buf)
    reset = torch.where(
        torch.any(torch.all(terminated.view(num_envs, value_size, -1), dim=-1), dim=-1), reset_ones, reset_buf
    )
    reset = torch.where(
        torch.all(progress_buf.view(num_envs, -1) >= max_episode_length - 1, dim=-1), reset_ones, reset
    )

    return total_reward, reset, terminated


@torch.jit.script
def compute_ant_observations(
    obs_buf: Tensor,
    terminated_buf: Tensor,
    root_states: Tensor,
    targets: Tensor,
    potentials: Tensor,
    inv_start_rot: Tensor,
    dof_pos: Tensor,
    dof_vel: Tensor,
    dof_limits_lower: Tensor,
    dof_limits_upper: Tensor,
    dof_vel_scale: float,
    sensor_force_torques: Tensor,
    actions: Tensor,
    dt: float,
    contact_force_scale: float,
    basis_vec0: Tensor,
    basis_vec1: Tensor,
    up_axis_idx: int,
    num_envs: int,
    num_agents: int,
    num_teams: int,
    max_num_obs_rng: int,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:

    # constants
    max_num_obs = min(max_num_obs_rng + 1, num_agents)

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]
    targets = targets.view(-1, targets.shape[-1])

    to_target = targets - torso_position
    to_target[:, 2] = 0.0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt
    potentials = potentials.view(-1, num_agents)

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    n_agents = vel_loc.shape[0]

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs(8), num_dofs(8), 24, num_dofs(8)
    # for v in (
    #     torso_position[:, up_axis_idx].view(-1, 1),
    #     vel_loc,
    #     angvel_loc,
    #     yaw.unsqueeze(-1),
    #     roll.unsqueeze(-1),
    #     angle_to_target.unsqueeze(-1),
    #     up_proj.unsqueeze(-1),
    #     heading_proj.unsqueeze(-1),
    #     dof_pos_scaled,
    #     dof_vel * dof_vel_scale,
    #     sensor_force_torques.view(-1, 24) * contact_force_scale,
    #     actions
    # ):
    #     print(v.shape)
    obs = torch.cat((
        torso_position[:, up_axis_idx].view(-1, 1),
        vel_loc,
        angvel_loc,
        yaw.unsqueeze(-1),
        roll.unsqueeze(-1),
        angle_to_target.unsqueeze(-1),
        up_proj.unsqueeze(-1),
        heading_proj.unsqueeze(-1),
        dof_pos_scaled.view(n_agents, -1),
        (dof_vel * dof_vel_scale).view(n_agents, -1),
        sensor_force_torques.view(-1, 24) * contact_force_scale,
        actions.view(n_agents, -1)
    ), dim=-1)
    # obs shape: (E*N, S)

    obs[terminated_buf.flatten(), :] = 0

    # interactive observation

    # ANN (all nearest neighbors) search

    # LSH / projection

    # naive search
    if num_agents > 1:
        pos = torso_position.view(num_envs, num_agents, -1)   # (E, N, 3)
        dist = torch.cdist(pos, pos)    # (E, N, N)
        pos_t = pos.T.unsqueeze(-1)     # (E, 3, N, 1)
        rel_pos = torch.cdist(pos_t, pos_t, p=1).permute(0, 2, 3, 1)    # (E, N, N, 3)
        val, ind = torch.topk(dist, max_num_obs, largest=False)     # (E, N, M+1)
        ind = ind[:, :, 1:]     # (E, N, M)
        rng_obs = obs[ind.view(-1, ind.shape[-1])][:, :, :28]   # (E * N, M, S_r)
        obs = torch.cat((obs, rng_obs.reshape(rng_obs.shape[0], -1)), dim=-1)   # (E * N, S + M * S_r)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec
