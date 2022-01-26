
from typing import Tuple

import math
import numpy as np
import torch
from torch import Tensor
import xml.etree.ElementTree as ET

from isaacgym import gymutil, gymtorch, gymapi
from isaacgym.torch_utils import tensor_clamp, torch_rand_float, to_torch
from utils.torch_jit_utils import quat_axis
from .base.ma_vec_task import MultiAgentVecTask,\
    reset_any_team_all_terminated, reset_max_episode_length,\
    obs_all_nearest_neighbors, obs_get_by_env_index, obs_same_team_index, obs_rel_pos_by_env_index,\
    reward_agg_sum, reward_reweight_team,\
    terminated_buf_update,\
    start_pos_circle


class BallJoust(MultiAgentVecTask):

    def __init__(self, cfg, sim_device, graphics_device_id, headless):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["maxEpisodeLength"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        num_obs = 13
        n_agts = cfg["env"].get("numAgentsPerTeam", 1) * cfg["env"].get("numTeams", 1)
        num_rng = min(n_agts - 1, 3)
        num_obs += (num_obs + 1 + 3) * num_rng
        self.num_rng = num_rng

        # Actions:
        num_acts = 4

        self.cfg["env"]["numObservations"] = num_obs
        self.cfg["env"]["numActions"] = num_acts

        self.thrust_action_speed_scale = 400

        super().__init__(
            config=self.cfg, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless
        )

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.initial_root_states = self.root_states.clone()

        max_thrust = 8
        self.thrust_lower_limits = torch.zeros(1, device=self.device, dtype=torch.float32)
        self.thrust_upper_limits = max_thrust * torch.ones(1, device=self.device, dtype=torch.float32)

        # control tensors
        self.thrusts = torch.zeros((self.num_agts, ), dtype=torch.float32, device=self.device, requires_grad=False)
        self.forces = torch.zeros((self.num_agts, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.all_actor_indices = torch.arange(self.num_agts, dtype=torch.int32, device=self.device)

        if self.viewer:
            cam_pos = gymapi.Vec3(1.0, 1.0, 2.5)
            cam_target = gymapi.Vec3(2.2, 2.0, 1.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

            # need rigid body states for visualizing thrusts
            self.rb_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
            self.rb_states = gymtorch.wrap_tensor(self.rb_state_tensor).view(self.num_agts, 13)
            self.rb_positions = self.rb_states[..., 0:3]
            self.rb_quats = self.rb_states[..., 3:7]

    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self.dt = self.sim_params.dt
        self._create_balljoust_asset()
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_balljoust_asset(self):
        chassis_radius = 0.1

        root = ET.Element('mujoco')
        root.attrib["model"] = "Ballcopter"
        compiler = ET.SubElement(root, "compiler")
        compiler.attrib["angle"] = "degree"
        compiler.attrib["coordinate"] = "local"
        compiler.attrib["inertiafromgeom"] = "true"
        worldbody = ET.SubElement(root, "worldbody")

        chassis = ET.SubElement(worldbody, "body")
        chassis.attrib["name"] = "chassis"
        chassis.attrib["pos"] = "%g %g %g" % (0, 0, 0)
        chassis_geom = ET.SubElement(chassis, "geom")
        # chassis_geom.attrib["type"] = "cylinder"
        # chassis_geom.attrib["size"] = "%g %g" % (chassis_radius, 0.5 * chassis_thickness)
        chassis_geom.attrib["type"] = "sphere"
        chassis_geom.attrib["size"] = "%g" % (chassis_radius)
        chassis_geom.attrib["pos"] = "0 0 0"
        chassis_geom.attrib["density"] = "50"
        chassis_joint = ET.SubElement(chassis, "joint")
        chassis_joint.attrib["name"] = "root_joint"
        chassis_joint.attrib["type"] = "free"

        gymutil._indent_xml(root)
        ET.ElementTree(root).write("ball.xml")

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "."
        asset_file = "ball.xml"

        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = False
        asset_options.angular_damping = 0.0
        asset_options.max_angular_velocity = 4 * math.pi
        asset_options.slices_per_cylinder = 40
        asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        pos = start_pos_circle(
            self.num_envs, self.num_agents, gymapi.UP_AXIS_Z, radius_coef=0.2, randomize_coef=0
        )

        for i in range(self.num_envs):
            # create env instance
            env = self.gym.create_env(self.sim, lower, upper, num_per_row)
            self.add_env(env)

            for k in range(self.num_agents):
                pose = gymapi.Transform()
                pose.p += gymapi.Vec3(*pos[i][k].tolist())
                pose.p.z = 1.0
                actor_handle = self.gym.create_actor(env, asset, pose, "balljoust", i, 0, 0)
                # pretty colors
                team = self.get_team_id(k)
                chassis_color = gymapi.Vec3(0.8, 0.6, 0.2) + self.team_colors[team]
                self.gym.set_rigid_body_color(env, actor_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, chassis_color)
                self.add_actor(actor_handle)

    def reset_idx(self, env_ids):
        agt_ids = self.get_reset_agent_ids(env_ids)
        num_resets = len(agt_ids)

        actor_indices = self.all_actor_indices[agt_ids].flatten()

        self.root_states[agt_ids] = self.initial_root_states[agt_ids]
        self.root_states[agt_ids, 0] += torch_rand_float(-0.2, 0.2, (num_resets, 1), self.device).flatten()
        self.root_states[agt_ids, 1] += torch_rand_float(-0.2, 0.2, (num_resets, 1), self.device).flatten()
        self.root_states[agt_ids, 2] += torch_rand_float(-0.2, 0.2, (num_resets, 1), self.device).flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, self.root_state_tensor, gymtorch.unwrap_tensor(actor_indices), num_resets
        )

        super().reset_idx(env_ids)

    def pre_physics_step(self, _actions):
        # for env_id, actor_id in torch.nonzero(self.terminated_buf, as_tuple=False).tolist():
            # print(actor_id)
        # resets
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)
        agt_ids = self.get_reset_agent_ids(reset_env_ids)

        self.terminate_agents()

        actions = _actions.to(self.device)
        actions = actions.view(self.num_agts, -1)
        actions = self.clear_terminated(actions)

        force_shifts = self.dt * 8 * math.pi * actions[:, 0:3]

        self.thrusts += self.dt * self.thrust_action_speed_scale * actions[:, -1]
        self.thrusts[:] = tensor_clamp(self.thrusts, self.thrust_lower_limits, self.thrust_upper_limits)
        self.thrusts = self.clear_terminated(self.thrusts)

        force_shifted = torch.nn.functional.normalize((self.forces + force_shifts), p=2.0, dim=-1)
        self.forces = force_shifted * self.thrusts.unsqueeze(-1)

        # clear actions for reset envs
        self.thrusts[agt_ids] = 0.0
        self.forces[agt_ids] = 0.0

        # apply actions
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.forces), None, gymapi.LOCAL_SPACE)

    def post_physics_step(self):

        self.update_progress()

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

        self.compute_observations()
        self.compute_reward()

        # debug viz
        if self.viewer and self.debug_viz:
            # compute start and end positions for visualizing thrust lines
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            rotor_indices = torch.LongTensor([2, 4, 6, 8])
            quats = self.rb_quats[:, rotor_indices]
            dirs = -quat_axis(quats.view(self.num_agts * 4, 4), 2).view(self.num_agts, 4, 3)
            starts = self.rb_positions[:, rotor_indices] + self.rotor_env_offsets
            ends = starts + 0.1 * self.thrusts.view(self.num_agts, 4, 1) * dirs

            # submit debug line geometry
            verts = torch.stack([starts, ends], dim=2).cpu().numpy()
            colors = np.zeros((self.num_agts * 4, 3), dtype=np.float32)
            colors[..., 0] = 1.0
            self.gym.clear_lines(self.viewer)
            self.gym.add_lines(self.viewer, None, self.num_agts * 4, verts, colors)

    def compute_observations(self):
        self.obs_buf[:] = compute_balljoust_observations(
            self.obs_buf,
            self.root_states,
            self.terminated_buf,
            self.num_envs,
            self.num_agents,
            self.num_rng,
            self.num_teams,
        )
        return self.obs_buf

    def compute_reward(self):
        self.rew_buf[:], self.reset_buf[:], self.terminated_buf[:] = compute_balljoust_reward(
            self.root_states,
            self.reset_buf,
            self.terminated_buf,
            self.progress_buf,
            self.reward_weight,
            self.max_episode_length,
            self.num_envs,
            self.num_agents,
            self.num_teams,
            self.value_size,
        )


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_balljoust_observations(
    obs_buf: Tensor,
    root_states: Tensor,
    terminated_buf: Tensor,
    num_envs: int,
    num_agents: int,
    num_rng: int,
    num_teams: int,
) -> Tensor:
    root_positions = root_states[..., 0:3]
    root_quats = root_states[..., 3:7]
    root_linvels = root_states[..., 7:10]
    root_angvels = root_states[..., 10:13]
    obs_buf = torch.zeros_like(obs_buf)
    obs_buf[..., 0:3] = root_positions
    obs_buf[..., 3:7] = root_quats
    obs_buf[..., 7:10] = root_linvels / 2
    obs_buf[..., 10:13] = root_angvels / math.pi

    if num_agents > 1:
        ind = obs_all_nearest_neighbors(root_positions, num_envs, num_agents, num_rng)
        rel_obs_obs = obs_get_by_env_index(obs_buf[..., :13], ind, num_envs, num_agents)
        rel_pos_obs = obs_rel_pos_by_env_index(root_positions, ind, num_envs, num_agents)
        team_obs = obs_same_team_index(ind, num_envs, num_agents, num_teams).float()
        obs_buf[..., 13:] = torch.cat((
            team_obs.view(obs_buf.shape[0], -1),
            rel_pos_obs.view(obs_buf.shape[0], -1),
            rel_obs_obs.view(obs_buf.shape[0], -1),
        ), dim=-1)

    obs_buf[terminated_buf.flatten(), :] = 0

    return obs_buf


@torch.jit.script
def compute_balljoust_reward(
    root_states: Tensor,
    reset_buf: Tensor,
    terminated_buf: Tensor,
    progress_buf: Tensor,
    reward_weight: Tensor,
    max_episode_length: int,
    num_envs: int,
    num_agents: int,
    num_teams: int,
    value_size: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    root_positions = root_states[..., 0:3]
    z_pos = root_states[..., 2]
    # distance to target
    target_dist = torch.sqrt(root_positions[..., 0] * root_positions[..., 0] +
                             root_positions[..., 1] * root_positions[..., 1] +
                             (1 - root_positions[..., 2]) * (1 - root_positions[..., 2]))
    pos_reward = 1.0 / (1.0 + target_dist * target_dist)
    root_dist = torch.norm(root_positions, dim=-1)

    # uprightness
    # root_quats = root_states[..., 3:7]
    # ups = quat_axis(root_quats, 2)
    # tiltage = torch.abs(1 - ups[..., 2])
    # up_reward = 1.0 / (1.0 + tiltage * tiltage)

    # spinning
    # root_angvels = root_states[..., 10:13]
    # spinnage = torch.abs(root_angvels[..., 2])
    # spinnage_reward = 1.0 / (1.0 + spinnage * spinnage)

    alive_reward = torch.ones_like(z_pos) * 0.5

    # combined reward
    # uprigness and spinning only matter when close to the target
        # + z_pos * 0.5 \
        # + spinnage_reward * 0.5 \
    reward = 0 \
        + alive_reward \
        + pos_reward \
        # + pos_reward * (up_reward + spinnage_reward)

    # resets due to misbehavior
    ones = torch.ones_like(z_pos, dtype=torch.bool)
    terminated = torch.zeros_like(z_pos, dtype=torch.bool)
    # terminated = torch.where(target_dist > 4.0, ones, terminated)
    terminated = torch.where(root_dist > 4.0, ones, terminated)
    terminated = torch.where(root_positions[..., 2] < 0.3, ones, terminated)

    if num_agents > 1:
        pos = root_positions.view(num_envs, num_agents, -1)
        z_pos = z_pos.view(num_envs, num_agents)
        dist = torch.cdist(pos, pos)    # [E, N, N]
        val, ind = [r[:, :, 1] for r in torch.topk(dist, 2, largest=False)]     # [E, N]
        env_idx = torch.arange(0, num_envs).view(num_envs, 1)
        target_alive = torch.logical_not(terminated_buf.view(num_envs, num_agents)[env_idx, ind])
        precond = torch.logical_and(target_alive, val < 0.25)
        tgt_z_pos = z_pos[env_idx, ind]
        jousted = torch.logical_and(precond, tgt_z_pos - z_pos > 0.1)
        jouster = torch.logical_and(precond, z_pos - tgt_z_pos > 0.1)
        terminated = torch.where(jousted.flatten(), ones, terminated)
        joust_score = 1000
        same_team = obs_same_team_index(ind.unsqueeze(-1), num_envs, num_agents, num_teams).squeeze()
        ek_jouster = torch.logical_and(torch.logical_not(same_team), jouster)
        reward[ek_jouster.flatten()] += joust_score
        # tk_jouster = torch.logical_and(torch.logical_not(same_team), jouster)
        # reward[tk_jouster.flatten()] -= joust_score
        # reward[jousted] -= joust_score

    reward = torch.where(terminated, torch.ones_like(reward) * (-2.0), reward)

    reward = torch.where(terminated_buf.flatten(), torch.zeros_like(reward), reward)

    reward = reward_agg_sum(reward, num_envs, value_size)

    # reward = reward_reweight_team(reward, reward_weight)

    reset = reset_any_team_all_terminated(reset_buf, terminated, num_envs, num_teams)
    # resets due to episode length
    # reset = torch.where(progress_buf >= max_episode_length - 1, ones, terminated)
    reset = reset_max_episode_length(reset, progress_buf, num_envs, max_episode_length)

    terminated_buf = terminated_buf_update(terminated_buf, terminated, num_envs)

    return reward, reset, terminated_buf
