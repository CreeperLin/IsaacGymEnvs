from typing import Dict, Any, Tuple
import sys
import math
from isaacgym import gymtorch
from isaacgym import gymapi
import torch
from .vec_task import VecTask
from torch import Tensor


@torch.jit.script
def obs_nearest_neighbors(
    pos: Tensor,
    num_envs: int,
    num_agents: int,
    max_num_obs: int,
) -> Tensor:
    pos = pos.view(num_envs, num_agents, -1)   # [E, N, 3]
    dist = torch.cdist(pos, pos)    # [E, N, N]
    _, ind = torch.topk(dist, max_num_obs, largest=False)     # [E, N, M+1]
    ind = ind[:, :, 1:]     # [E, N, M]
    return ind


@torch.jit.script
def reset_any_team_all_terminated(
    reset: Tensor,
    terminated: Tensor,
    num_envs: int,
    value_size: int,
):
    reset_ones = torch.ones_like(reset)
    return torch.where(
        torch.any(torch.all(terminated.view(num_envs, value_size, -1), dim=-1), dim=-1), reset_ones, reset
    )


@torch.jit.script
def reset_max_episode_length(
    reset: Tensor,
    progress_buf: Tensor,
    num_envs: int,
    max_episode_length: int,
):
    reset_ones = torch.ones_like(reset)
    return torch.where(
        torch.all(progress_buf.view(num_envs, -1) >= max_episode_length - 1, dim=-1), reset_ones, reset
    )


@torch.jit.script
def reward_sum_team(
    reward: Tensor,
    num_envs: int,
    value_size: int,
):
    return torch.sum(reward.view(num_envs, value_size, -1), dim=-1).flatten()


@torch.jit.script
def terminated_buf_update(
    terminated_buf: Tensor,
    terminated: Tensor,
    num_envs: int,
):
    return torch.logical_or(terminated_buf, terminated.view(num_envs, -1))


def start_pose_radian(
    num_envs,
    num_agents,
    up_axis_idx=gymapi.UP_AXIS_Z,
    radius_coef=0.75,
    randomize_coef=1.,
):
    init_pos_radius = num_agents * radius_coef
    pos = torch.arange(0, num_agents) * 2. * math.pi / num_agents
    pos = pos.repeat(3, num_envs).T.view(num_envs, num_agents, -1)
    if up_axis_idx is gymapi.UP_AXIS_Z:
        x_axis, y_axis, up_axis = [0, 1, 2]
    else:
        raise NotImplementedError
    pos[:, :, x_axis].cos_()
    pos[:, :, y_axis].sin_()
    pos.mul_(init_pos_radius)
    if randomize_coef:
        pos += randomize_coef * torch.randn(num_envs, num_agents, 3)
    pos[:, :, up_axis] = 0
    return pos


class MultiAgentVecTask(VecTask):

    def __init__(self, config, sim_device, graphics_device_id, headless):
        self.num_teams = config["env"].get("numTeams", 1)
        self.num_agents_team = config["env"].get("numAgentsPerTeam", 1)
        self.num_agents = self.num_agents_team * self.num_teams
        config["env"]["numAgents"] = self.num_agents
        self.num_agents_export = config["env"].get("numAgentsExport", self.num_teams)

        self.ma_supported = config["env"].get("maSupported", True)
        space_mult = self.num_agents // (self.num_agents_export if self.ma_supported else 1)
        self.num_obs_per_agent = config["env"]["numObservations"]
        self.num_acts_per_agent = config["env"]["numActions"]
        config["env"]["numObservations"] = self.num_obs_per_agent * space_mult
        config["env"]["numActions"] = self.num_acts_per_agent * space_mult
        self.reward_sum = config["env"].get("rewardSum", 'team')
        zero_sum = config["env"].get("rewardZeroSum", True)
        if self.reward_sum == 'team':
            value_size = self.num_teams
        elif self.reward_sum == 'all':
            value_size = 1
        elif self.reward_sum in [None, 'none']:
            value_size = self.num_agents
        self.value_size = value_size
        self.value_size_export = self.value_size // self.num_agents_export
        team_colors = [
            (0.97, 0.38, 0.06),
            (0.38, 0.06, 0.97),
            (0.06, 0.97, 0.38),
        ]
        self.team_colors = [gymapi.Vec3(*t) for t in team_colors]
        self.num_agts = config["env"]["numEnvs"] * self.num_agents

        self.viewer = None
        self.enable_viewer_sync = config["env"].get("rewardZeroSum", True)
        self.viewer_render_collision = False
        self.actor_handles = []
        self.env_handles = []

        super().__init__(
            config=config, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless
        )

        reward_weight = torch.ones((value_size, value_size), device=self.device, dtype=torch.float)
        if zero_sum and value_size > 1:
            reward_weight *= (-1 / (value_size-1))
            reward_weight[[torch.arange(0, value_size)] * 2] = 1.
        self.reward_weight = reward_weight
        self.terminated_buf = torch.zeros((self.num_envs, self.num_agents), device=self.device, dtype=torch.bool)

        save_replay_steps = config["env"].get("saveReplaySteps", 0)
        self.save_replay = save_replay_steps > 0
        if self.save_replay:
            self.replay_pt = -1
            self.replay_device = 'cpu'
            self.replay_actions = torch.zeros(
                (save_replay_steps, self.self.num_envs, self.num_agents), device=self.replay_device, dtype=torch.float
            )

    def add_actor(self, actor_handle):
        self.actor_handles.append(actor_handle)

    def add_env(self, env_handle):
        self.env_handles.append(env_handle)

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

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    def get_team_id(self, agt_id):
        return agt_id // self.num_agents_team

    def get_reset_agent_ids(self, env_ids):
        ent_ids = (env_ids * self.num_agents).repeat(self.num_agents).view(self.num_agents, -1)
        ent_ids += torch.arange(0, self.num_agents).view(-1, 1).to(device=self.device)
        return ent_ids.flatten().to(dtype=torch.long)

    def reset_idx(self, env_ids):
        self.progress_buf.view(self.num_envs, -1)[env_ids] = 0
        self.reset_buf.view(self.num_envs, -1)[env_ids] = 0
        self.terminated_buf.view(self.num_envs, -1)[env_ids] = 0

    def get_obs_export(self):
        return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs)\
            .view(self.num_envs * self.num_agents_export, self.num_obs)\
            .to(self.rl_device)

    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if self.save_replay:
            self.replay_pt += 1
            self.replay_actions[self.replay_pt] = actions.to(self.replay_device, non_blocking=True)
        obs_dict, rew_buf, reset_buf, extras = super().step(actions)
        if self.num_agents_export > 1:
            obs_dict['obs'] = obs_dict['obs'].view(self.num_envs * self.num_agents_export, self.num_obs)
            reset_buf = reset_buf.repeat(self.num_agents_export)
            # reset_buf = torch.all(self.terminated_buf.view(self.num_envs, self.num_agents_export, -1), dim=-1)
        return obs_dict, rew_buf, reset_buf, extras

    def update_progress(self):
        self.progress_buf[
            torch.logical_not(torch.all(self.terminated_buf.view(self.num_envs, self.value_size, -1), dim=-1)).flatten()
        ] += 1

    def clear_terminated_actions(self, actions):
        actions.view(self.num_envs, self.num_agents, -1)[self.terminated_buf] = 0
        return actions

    def terminate_agents(self):
        pass

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

    def set_viewer(self):
        """Create the viewer."""
        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless is False:
            cam_props = gymapi.CameraProperties()
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, cam_props)
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def render(self):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, self.viewer_render_collision)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

            else:
                self.gym.poll_viewer_events(self.viewer)