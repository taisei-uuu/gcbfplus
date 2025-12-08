import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import NamedTuple, Tuple, Optional

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, AgentState, Array, Cost, Done, Info, Reward, State
from ..utils.utils import merge01, jax_vmap
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class DoubleIntegrator(MultiAgentEnv):
    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: AgentState
        goal: State
        obstacle: Obstacle
        # フォーメーション割り当ての状態
        formation_assignment: Optional[Array] = None  # 現在の割り当て [n_followers]
        formation_assignment_age: Optional[Array] = None  # 割り当ての経過ステップ数 [n_followers]

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.5],
        "n_obs": 8,
        "m": 0.1,  # mass
        "formation_mode": False, #フォーメーションモード
        "formation_offsets": None, #フォーメーションオフセット
        "formation_flexible_assignment": False,  # 柔軟な割り当てを有効化
        "formation_min_distance": 0.1,  # フォーメーション達成とみなす最小距離
        "formation_assignment_cooldown": 30,  # 割り当て変更のクールダウン期間（ステップ数）
        "formation_assignment_min_diff": 0.05,  # 再割り当てを検討する最小距離差
        "fixed_config": None,  # 固定シナリオ設定
        "kp_bs": 1.0,  # Backstepping position gain
        "kv_bs": 2.0,  # Backstepping velocity gain
    }

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None
    ):
        super(DoubleIntegrator, self).__init__(num_agents, area_size, max_step, max_travel, dt, params)
        A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32)
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        self._A = A * self._dt + np.eye(self.state_dim)
        self._B = (
            np.array([[0.0, 0.0], [0.0, 0.0], [1.0 / self._params["m"], 0.0], [0.0, 1.0 / self._params["m"]]])
            * self._dt
        )
        self._Q = np.eye(self.state_dim) * 5
        self._R = np.eye(self.action_dim)
        self._K = jnp.array(lqr(self._A, self._B, self._Q, self._R))
        self.create_obstacles = jax_vmap(Rectangle.create)

    def _get_formation_radius(self) -> float:
        """
        フォーメーションオフセットの最大半径を返す
        """
        if not self._params.get("formation_mode", False):
            return 0.0
        
        offsets = self._params.get("formation_offsets", None)
        if offsets is None:
            return 0.0
            
        # オフセットのノルムの最大値を計算
        max_radius = 0.0
        for offset in offsets:
            r = np.linalg.norm(offset)
            if r > max_radius:
                max_radius = r
        
        return float(max_radius)

    @property
    def state_dim(self) -> int:
        return 4  # x, y, vx, vy

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, vx_rel, vy_rel

    @property
    def action_dim(self) -> int:
        return 2  # fx, fy

    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0

        # 固定シナリオモードのチェック
        fixed_config = self._params.get("fixed_config", None)
        
        if fixed_config is not None:
            # --- 固定設定を使用 ---
            # Obstacles
            obs_conf = fixed_config["obstacles"]
            # obs_conf is expected to be a list of dicts or a structured object, 
            # here we assume simple array structure for easy JAX handling or pre-processed dict
            # user-provided dict: {"pos": [[x,y], ...], "len": [[w,h], ...], "theta": [t, ...]}
            obs_pos = jnp.array(obs_conf["pos"])
            obs_len = jnp.array(obs_conf["len"])
            obs_theta = jnp.array(obs_conf["theta"])
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
            
            # Agents
            agent_conf = fixed_config["agents"]
            states = jnp.array(agent_conf["start"])
            goals = jnp.array(agent_conf["goal"])
            
            # エージェント数が合っているか確認 (Optional)
            # assert states.shape[0] == self.num_agents
        else:
            # --- 通常のランダム生成 ---
            # randomly generate obstacles
            n_rng_obs = self._params["n_obs"]
            assert n_rng_obs >= 0
            obstacle_key, key = jr.split(key, 2)
            obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=self.area_size)
            length_key, key = jr.split(key, 2)
            obs_len = jr.uniform(
                length_key,
                (n_rng_obs, 2),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, key = jr.split(key, 2)
            obs_theta = jr.uniform(theta_key, (n_rng_obs,), minval=0, maxval=2 * np.pi)
            obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)
    
            # randomly generate agent and goal
            states, goals = get_node_goal_rng(
                key, self.area_size, 2, obstacles, self.num_agents, 4 * self.params["car_radius"], self.max_travel,
                # 初期位置を限定
                formation_mode=self._params.get("formation_mode", False),
                formation_start_radius=self._get_formation_radius() 
            )

        # add zero velocity
        states = jnp.concatenate([states, jnp.zeros((self.num_agents, 2))], axis=1)
        goals = jnp.concatenate([goals, jnp.zeros((self.num_agents, 2))], axis=1)

        #フォーメーションモードの場合
        formation_assignment = None
        formation_assignment_age = None
        if self._params.get("formation_mode", False):
            formation_offsets = self._params["formation_offsets"]
            if formation_offsets is not None:
                leader_pos = states[0, :2]  # リーダーの位置 [x, y]
                
                if self._params.get("formation_flexible_assignment", False):
                    # 柔軟な割り当てモード: 初期割り当てを計算
                    follower_positions = states[1:, :2]
                    initial_assignment, initial_age = self.assign_formation_offsets(
                        leader_pos,
                        follower_positions,
                        formation_offsets,
                        previous_assignment=None,  # 初期状態なので前回の割り当てなし
                        assignment_age=None,
                        cooldown_steps=self._params.get("formation_assignment_cooldown", 30),
                        min_distance_diff=self._params.get("formation_assignment_min_diff", 0.05),
                    )
                    formation_assignment = initial_assignment
                    formation_assignment_age = initial_age
                    
                    # 割り当てられたオフセットに基づいてゴールを設定
                    for i in range(len(initial_assignment)):
                        offset = formation_offsets[initial_assignment[i]]
                        goals = goals.at[i + 1, :2].set(leader_pos + offset)
                else:
                    # 既存の固定割り当てモード
                    for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                        offset = jnp.array(formation_offsets[i-1])
                        goals = goals.at[i, :2].set(leader_pos + offset)

        env_states = self.EnvState(states, goals, obstacles, formation_assignment, formation_assignment_age)

        return self.get_graph(env_states)

    def agent_accel(self, action: Action) -> Action:
        return action / self._params["m"]

    def agent_step_exact(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        # [x, y, vx, vy]
        assert agent_states.shape == (self.num_agents, self.state_dim)
        n_accel = self.agent_accel(action)
        n_pos_new = agent_states[:, :2] + agent_states[:, 2:] * self.dt + n_accel * self.dt**2 / 2
        n_vel_new = agent_states[:, 2:] + n_accel * self.dt
        n_state_agent_new = jnp.concatenate([n_pos_new, n_vel_new], axis=1)
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return n_state_agent_new

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        # [x, y, vx, vy]
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = self.agent_xdot(agent_states, action)
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def agent_xdot(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        n_accel = self.agent_accel(action)
        x_dot = jnp.concatenate([agent_states[:, 2:], n_accel], axis=1)
        assert x_dot.shape == (self.num_agents, self.state_dim)
        return x_dot

    def assign_formation_offsets(
        self, 
        leader_pos: Array, 
        follower_positions: Array, 
        formation_offsets: Array,
        previous_assignment: Optional[Array] = None,  # 前回の割り当て
        assignment_age: int=0,  # 各割り当ての経過ステップ数
        cooldown_steps: int = 15,  # クールダウン期間（ステップ数）
        min_distance_diff: float = 0.05,  # 再割り当てを検討する最小距離差
    ) -> Tuple[Array, int]:
        """
        フォロワーの現在位置に基づいて、最適なオフセット割り当てを決定する。
        チャタリングを防ぐため、ヒステリシス（クールダウンタイム）を実装。
        
        Args:
            leader_pos: リーダーの位置 [x, y]
            follower_positions: フォロワーの現在位置 [n_followers, 2]
            formation_offsets: 利用可能なオフセット [n_offsets, 2]
            previous_assignment: 前回の割り当て [n_followers] (オプション)
            assignment_age: 各割り当ての経過ステップ数 (オプション)
            cooldown_steps: 再割り当てを禁止する期間（ステップ数）
            min_distance_diff: 再割り当てを検討する最小の総距離差
        
        Returns:
            (assignment, new_assignment_age): 割り当てと更新された経過ステップ数
        """
        n_followers = follower_positions.shape[0] #0行目=followerの数
        n_offsets = formation_offsets.shape[0] #0行目=オフセットの数
        
        # 初期化
        if previous_assignment is None:
            previous_assignment = jnp.arange(n_followers, dtype=jnp.int32)
        if assignment_age is None:
            assignment_age = 0
        
        # 各フォロワーと各オフセットの組み合わせについて、目標位置を計算
        target_positions = leader_pos[None, :] + formation_offsets[:, None, :]  # [n_offsets, n_followers, 2]
        target_positions = jnp.transpose(target_positions, (1, 0, 2))  # [n_followers, n_offsets, 2]
        
        # 各フォロワーから各目標位置への距離を計算
        distances = jnp.linalg.norm(
            follower_positions[:, None, :] - target_positions, 
            axis=-1
        )  # [n_followers, n_offsets]
        
        # 新しい割り当てを計算（クールダウンを考慮しない）
        assignment = jnp.zeros(n_followers, dtype=jnp.int32)
        used_offsets = jnp.zeros(n_offsets, dtype=jnp.bool_)
        
        def assign_one(carry, follower_idx):
            assignment, used_offsets, distances = carry
            # 未使用のオフセットの中で、このフォロワーに最も近いものを選択
            masked_distances = jnp.where(
                used_offsets, #modified
                jnp.inf,
                distances[follower_idx, :]
            )
            best_offset_idx = jnp.argmin(masked_distances)
            assignment = assignment.at[follower_idx].set(best_offset_idx)
            used_offsets = used_offsets.at[best_offset_idx].set(True)
            return (assignment, used_offsets, distances), None
        
        (new_assignment, _, _), _ = jax.lax.scan(
            assign_one,
            (assignment, used_offsets, distances),
            jnp.arange(n_followers)
        )
        
        # クールダウン期間中の割り当てを維持
        # 前回の割り当てからの総距離差を計算
        current_distances = jnp.take_along_axis(
            distances,
            previous_assignment[:, None],
            axis=1
        ).squeeze(1)
        current_total_distance = jnp.sum(current_distances) # 総距離差
        
        new_distances = jnp.take_along_axis(
            distances,
            new_assignment[:, None],
            axis=1
        ).squeeze(1)
        new_total_distance = jnp.sum(new_distances)
        
        distance_improvement = current_total_distance - new_total_distance  # 改善量
        
        # クールダウン期間中、または改善が小さい場合は前回の割り当てを維持
        in_cooldown = assignment_age < cooldown_steps
        small_improvement = distance_improvement < min_distance_diff
        
        should_keep_previous = jnp.logical_or(in_cooldown, small_improvement)
        
        final_assignment = jnp.where(
            should_keep_previous,
            previous_assignment,
            new_assignment
        )
        
        # 割り当てが変更された場合はageをリセット、そうでなければインクリメント
        assignment_changed = jnp.any(final_assignment != previous_assignment)
        new_assignment_age = jnp.where(
            assignment_changed,
            0,
            assignment_age + 1
        )
        
        return final_assignment, new_assignment_age

    def step(
        self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)

        # フォーメーションモードの場合、フォロワーのゴールを更新
        formation_assignment = None
        formation_assignment_age = None
        if self._params.get("formation_mode", False):
            formation_offsets = self._params["formation_offsets"]
            if formation_offsets is not None:
                leader_pos = next_agent_states[0, :2]  # リーダーの新しい位置
                
                if self._params.get("formation_flexible_assignment", False):
                    # 柔軟な割り当てモード（チャタリング対策付き）
                    follower_indices = jnp.arange(1, self.num_agents)
                    follower_positions = next_agent_states[1:, :2]
                    
                    # 前回の割り当て状態を取得
                    prev_assignment = graph.env_states.formation_assignment
                    prev_age = graph.env_states.formation_assignment_age
                    
                    offset_indices, new_age = self.assign_formation_offsets(
                        leader_pos,
                        follower_positions,
                        formation_offsets,
                        previous_assignment=prev_assignment,
                        assignment_age=prev_age,
                        cooldown_steps=self._params.get("formation_assignment_cooldown", 30),
                        min_distance_diff=self._params.get("formation_assignment_min_diff", 0.05),
                    )
                    formation_assignment = offset_indices
                    formation_assignment_age = new_age
                    
                    # 割り当てられたオフセットに基づいてゴールを設定
                    for i in range(len(offset_indices)):
                        offset = formation_offsets[offset_indices[i]]
                        goal_states = goal_states.at[i + 1, :2].set(leader_pos + offset)
                else:
                    # 既存の固定割り当てモード
                    for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                        offset = jnp.array(formation_offsets[i-1])
                        goal_states = goal_states.at[i, :2].set(leader_pos + offset)
            
            # リーダーの速度をフォロワーの目標速度として設定
            leader_vel = next_agent_states[0, 2:]
            goal_states = goal_states.at[1:, 2:].set(leader_vel)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(
            next_agent_states,
            goal_states,
            obstacles,
            formation_assignment,
            formation_assignment_age,
        )

        info = {}
        if get_eval_info:
            # collision between agents and obstacles
            agent_pos = agent_states[:, :2]
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])

        return self.get_graph(next_state), reward, cost, done, info

    def get_cost(self, graph: EnvGraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states[:, :2]
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost

    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: State) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent[:, :2]
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] + 1)
        state_diff = state.agent[:, None, :] - state.agent[None, :, :]
        agent_agent_mask = jnp.less(dist, self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(state_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_pos = agent_pos[i, :] - lidar_data[id_hits, :2]
            lidar_feats = state.agent[i, :] - lidar_data[id_hits, :]
            lidar_dist = jnp.linalg.norm(lidar_pos, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.concatenate([state[:, 2:], jnp.zeros((state.shape[0], 2))], axis=1)
        g = jnp.concatenate([jnp.zeros((2, 2)), jnp.eye(2) / self._params['m']], axis=0)
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2

        edge_feats = state[graph.receivers] - state[graph.senders]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)

        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(DoubleIntegrator.GOAL)
        node_type = node_type.at[-n_hits:].set(DoubleIntegrator.OBS)

        get_lidar_vmap = jax_vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent[:, :2]))
        lidar_data = jnp.concatenate([lidar_data, jnp.zeros_like(lidar_data)], axis=-1)
        edge_blocks = self.edge_blocks(state, lidar_data)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.array([-jnp.inf, -jnp.inf, -0.5, -0.5])
        upper_lim = jnp.array([jnp.inf, jnp.inf, 0.5, 0.5])
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        
        # Backstepping Control
        # x_i: agent[:, :2], v_i: agent[:, 2:]
        # p_i^d: goal[:, :2], v_i^d: goal[:, 2:]
        # a_i^d is assumed to be 0
        
        x_i = agent[:, :2]
        v_i = agent[:, 2:]
        p_d = goal[:, :2]
        v_d = goal[:, 2:]
        
        kp = self._params["kp_bs"]
        kv = self._params["kv_bs"]
        
        # Step 1: Position Error
        e_p = x_i - p_d
        
        # Virtual Velocity Reference
        # v_{i,ref} = -k_p * e_p + v_d
        v_ref = -kp * e_p + v_d
        
        # Step 2: Velocity Error
        # e_v = v_i - v_{i,ref}
        e_v = v_i - v_ref
        
        # Control Input
        # u_i = -k_v * e_v - e_p
        # (Assuming a_d = 0)
        u_bs = -kv * e_v - e_p
        
        return self.clip_action(u_bs)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)

        # フォーメーションモードの場合、フォロワーのゴールを更新
        # 注意: forward_graph()はグラフの前進計算のみを行うため、割り当て状態の更新は行わない
        # 割り当て状態はstep()でのみ更新される
        if self._params.get("formation_mode", False):
            formation_offsets = self._params["formation_offsets"]
            if formation_offsets is not None:
                leader_pos = next_agent_states[0, :2]
                
                if self._params.get("formation_flexible_assignment", False):
                    # 柔軟な割り当てモード: 前回の割り当てを使用（状態は更新しない）
                    prev_assignment = graph.env_states.formation_assignment
                    if prev_assignment is not None:
                        # 前回の割り当てを使用してゴールを設定
                        for i in range(len(prev_assignment)):
                            offset = formation_offsets[prev_assignment[i]]
                            goal_states = goal_states.at[i + 1, :2].set(leader_pos + offset)
                    else:
                        # 初期状態: 固定割り当てを使用
                        for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                            offset = jnp.array(formation_offsets[i-1])
                            goal_states = goal_states.at[i, :2].set(leader_pos + offset)
                else:
                    # 既存の固定割り当てモード
                    for i in range(1, min(len(formation_offsets) + 1, self.num_agents)):
                        offset = jnp.array(formation_offsets[i-1])
                        goal_states = goal_states.at[i, :2].set(leader_pos + offset)
            
            # リーダーの速度をフォロワーの目標速度として設定
            leader_vel = next_agent_states[0, 2:]
            goal_states = goal_states.at[1:, 2:].set(leader_vel)
                    
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)
        return next_graph

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 4)

        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 2)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]

        # agents are colliding
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        agent_dist = agent_dist + jnp.eye(agent_dist.shape[1]) * (self._params["car_radius"] * 2 + 1)
        unsafe_agent = jnp.less(agent_dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        # unsafe direction
        agent_warn_dist = 3 * self._params["car_radius"]
        obs_warn_dist = 2 * self._params["car_radius"]
        obs_pos = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)[:, :2]
        obs_pos_diff = obs_pos[None, :, :] - agent_pos[:, None, :]
        obs_dist = jnp.linalg.norm(obs_pos_diff, axis=-1)
        pos_diff = jnp.concatenate([agent_pos_diff, obs_pos_diff], axis=1)
        warn_zone = jnp.concatenate([jnp.less(agent_dist, agent_warn_dist), jnp.less(obs_dist, obs_warn_dist)], axis=1)
        pos_vec = (pos_diff / (jnp.linalg.norm(pos_diff, axis=2, keepdims=True) + 0.0001))
        speed_agent = jnp.linalg.norm(agent_state[:, 2:], axis=1, keepdims=True)
        heading_vec0 = (agent_state[:, 2:] / (speed_agent + 0.0001))[:, None, :]
        heading_vec = heading_vec0.repeat(pos_vec.shape[1], axis=1)
        inner_prod = jnp.sum(pos_vec * heading_vec, axis=2)
        unsafe_theta_agent = jnp.arctan2(self._params['car_radius'] * 2,
                                         jnp.sqrt(agent_dist**2 - 4 * self._params['car_radius']**2))
        unsafe_theta_obs = jnp.arctan2(self._params['car_radius'],
                                       jnp.sqrt(obs_dist**2 - self._params['car_radius']**2))
        unsafe_theta = jnp.concatenate([unsafe_theta_agent, unsafe_theta_obs], axis=1)
        lidar_mask = jnp.ones((self._params["n_rays"],))
        lidar_mask = jax.scipy.linalg.block_diag(*[lidar_mask] * self.num_agents)
        valid_mask = jnp.concatenate([jnp.ones((self.num_agents, self.num_agents)), lidar_mask], axis=-1)
        warn_zone = jnp.logical_and(warn_zone, valid_mask)
        unsafe_dir = jnp.max(jnp.logical_and(warn_zone, jnp.greater(inner_prod, jnp.cos(unsafe_theta))), axis=1)

        return jnp.logical_or(collision_mask, unsafe_dir)  # | unsafe_stop

    def collision_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        collision_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return collision_mask

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.env_states.goal[:, :2]
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach
