#waterworld_base.py
import math
from collections import defaultdict
import gymnasium
import numpy as np
import pygame
import pymunk
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.spatial import distance as ssd
import random

from pettingzoo.sisl.waterworld.waterworld_models import (
    Evaders,
    Obstacle,
    Poisons,
    Predator,
    Prey,
)

FPS = 15

DEFAULT_PERF = {
    'foraging_rate': 0.0,
    'escape_rate':   0.0,
    'hunting_rate':  0.0,
    'DESR':          0.0,
    'MDSD':          0.0,
}


class ObservationBasedRewardCalculator:
    """基于观察信息计算奖励，更高效更简洁"""
    
    def __init__(self, n_sensors=30):
        self.n_sensors = n_sensors
        
        self.obs_indices = {
            'obstacle_dist': (0, n_sensors),
            'barrier_dist': (n_sensors, 2*n_sensors),
            'food_dist': (2*n_sensors, 3*n_sensors),
            'food_vel': (3*n_sensors, 4*n_sensors),
            'poison_dist': (4*n_sensors, 5*n_sensors),
            'poison_vel': (5*n_sensors, 6*n_sensors),
            'agent_dist': (6*n_sensors, 7*n_sensors),
            'agent_vel': (7*n_sensors, 8*n_sensors),
            'agent_type': (8*n_sensors, 9*n_sensors),
            'agent_id': (9*n_sensors, 10*n_sensors),
            'food_contact': 10*n_sensors,
            'poison_contact': 10*n_sensors + 1,
        }
        
        self.predator_params = {
            'catch_reward': 4.0,
            'distance_scale': 0.0,
            'sight_scale': 0.0,
            'approach_scale': 0.0,
            'survival_reward': 0.0,
        }
        
        self.prey_params = {
            'food_reward': 3.0,
            'poison_reward': 3.0,
            'contact_reward': 0.03,
            'caught_penalty': -3.0,
            'escape_reward': 0.05,
            'survival_reward': 0.00,
        }
        
        self.prev_observations = {}
        self.prev_flags = defaultdict(lambda: {'caught': 0, 'food': 0, 'poison': 0})
        
    def extract_obs_features(self, observation):
        """从观察中提取关键特征"""
        features = {}
        for key, indices in self.obs_indices.items():
            if isinstance(indices, tuple):
                start, end = indices
                features[key] = observation[start:end]
            else:
                features[key] = observation[indices]
        return features
    
    def calculate_predator_reward_from_obs(self, agent_id, observation, catch_indicator=0, catch_active=False):
        """基于观察计算Predator奖励
        
        Args:
            catch_indicator: 本帧是否发生捕获（用于奖励计算）
            catch_active: 是否仍在捕获状态中（用于 prev_flags 更新）
        """
        features = self.extract_obs_features(observation)
        total_reward = 0.0
        comp = {}

        # 1) 捕获：只在 0→1 上升沿给一次奖
        caught_edge = 1 if (catch_indicator > 0 and self.prev_flags[agent_id]['caught'] == 0) else 0
        catch_reward = self.predator_params['catch_reward'] * caught_edge
        total_reward += catch_reward
        comp['catch'] = float(catch_reward)

        # 2) 距离塑形
        agent_distances = features['agent_dist']
        agent_types = features['agent_type']
        prey_mask = (agent_types == 1.0) & (agent_distances < 1.0)
        if np.any(prey_mask):
            min_prey_distance = float(np.min(agent_distances[prey_mask]))
            distance_reward = self.predator_params['distance_scale'] * max(0.0, 1.0 - min_prey_distance)
            distance_reward = float(np.clip(distance_reward, 0.0, 0.2))
            total_reward += distance_reward
            comp['dist'] = distance_reward
        else:
            comp['dist'] = 0.0

        # 3) 视野塑形
        prey_in_sight = int(np.sum((agent_distances < 1.0) & (agent_types == 1.0)))
        sight_reward = self.predator_params['sight_scale'] * prey_in_sight
        sight_reward = float(np.clip(sight_reward, 0.0, 0.2))
        total_reward += sight_reward
        comp['sight'] = sight_reward

        # 4) 接近差分
        if agent_id in self.prev_observations:
            prev_features = self.extract_obs_features(self.prev_observations[agent_id])
            prev_dist = prev_features['agent_dist']
            prev_type = prev_features['agent_type']
            prev_prey_mask = (prev_type == 1.0) & (prev_dist < 1.0)
            curr_prey_mask = prey_mask
            if np.any(prev_prey_mask) and np.any(curr_prey_mask):
                prev_min = float(np.min(prev_dist[prev_prey_mask]))
                curr_min = float(np.min(agent_distances[curr_prey_mask]))
                d_change = max(0.0, prev_min - curr_min)
                d_change = min(d_change, 0.2)
                approach_reward = self.predator_params['approach_scale'] * d_change
                approach_reward = float(np.clip(approach_reward, 0.0, 0.2))
                total_reward += approach_reward
                comp['approach'] = approach_reward
            else:
                comp['approach'] = 0.0
        else:
            comp['approach'] = 0.0

        self.prev_observations[agent_id] = observation.copy()
        total_reward = float(np.clip(total_reward, -2.0, 4.0))

        self._last_components = getattr(self, "_last_components", {})
        self._last_components[agent_id] = comp

        return total_reward

    def calculate_prey_reward_from_obs(self, agent_id, observation, 
                                       food_contact=0, food_touching=False,
                                       poison_contact=0, poison_touching=False,
                                       caught_indicator=0):
        """基于观察计算Prey奖励（修复版）
        
        Args:
            food_contact: 本帧是否成功吃到食物（food_indicator）
            food_touching: 是否正在接触食物（food_touched_indicator > 0）
            poison_contact: 本帧是否接触毒物
            poison_touching: 是否正在接触毒物
            caught_indicator: 是否被捕获
        """
        features = self.extract_obs_features(observation)
        total_reward = 0.0
        comp = {}

        # 🔍 诊断5：函数入口
        if food_contact > 0 or poison_contact > 0:
            pass
            # print(f"🔬 [Agent {agent_id}] calculate_prey_reward_from_obs CALLED")
            # print(f"   INPUT PARAMS:")
            # print(f"      food_contact: {food_contact}")
            # print(f"      food_touching: {food_touching}")
            # print(f"      poison_contact: {poison_contact}")
            # print(f"      poison_touching: {poison_touching}")
            # print(f"      caught_indicator: {caught_indicator}")
            # print(f"   CURRENT STATE:")
            # print(f"      prev_flags[food]: {self.prev_flags[agent_id]['food']}")
            # print(f"      prev_flags[poison]: {self.prev_flags[agent_id]['poison']}")

        # 0) 被捕获：直接返回惩罚
        if caught_indicator > 0:
            pen = float(self.prey_params['caught_penalty'])
            comp = {'caught_pen': pen}
            self._last_components = getattr(self, "_last_components", {})
            self._last_components[agent_id] = comp
            # if food_contact > 0 or poison_contact > 0:
            #     # print(f"   💀 CAUGHT! Returning penalty: {pen}")
            #     # print(f"{'*'*60}\n")
            return pen

        # 🔥 关键修复：边沿检测基于 touching 状态（持续接触），而不是 contact（瞬时事件）
        # 从"不在接触"到"在接触"才触发奖励
        # ✅ 确认边沿检测使用的是touching状态
        food_edge = 1 if (food_touching and self.prev_flags[agent_id]['food'] == 0) else 0
        poison_edge = 1 if (poison_touching and self.prev_flags[agent_id]['poison'] == 0) else 0

        # ✅ 只在边沿触发时给奖励
        food_r = float(self.prey_params['food_reward'] * food_edge)
        poi_r = float(self.prey_params['poison_reward'] * poison_edge)
        total_reward += (food_r + poi_r)
        comp['food'] = food_r
        comp['poison'] = poi_r
        
        # # 🔍 诊断7：边沿奖励
        # if food_r > 0 or poi_r > 0:
        #     print(f"   💰 EDGE REWARDS:")
        #     print(f"      food_reward = {self.prey_params['food_reward']} * {food_edge} = {food_r}")
        #     print(f"      poison_reward = {self.prey_params['poison_reward']} * {poison_edge} = {poi_r}")
        #     print(f"      total so far: {total_reward}")

        # 2) 触碰中的小额塑形（可选）
        if food_contact > 0 or poison_contact > 0:
            cr = float(self.prey_params['contact_reward'])
            total_reward += cr
            comp['contact'] = cr
            # if food_contact > 0 or poison_contact > 0:
            #     print(f"   📍 Contact reward: {cr}")
        else:
            comp['contact'] = 0.0

        # 3) 逃离塑形
        agent_distances = features['agent_dist']
        agent_types = features['agent_type']
        predator_mask = (agent_types == 0.0) & (agent_distances < 1.0)
        if np.any(predator_mask):
            min_predator_distance = float(np.min(agent_distances[predator_mask]))
            escape_reward = float(self.prey_params['escape_reward'] * min_predator_distance)
            escape_reward = float(np.clip(escape_reward, 0.0, 0.2))
            total_reward += escape_reward
            comp['escape'] = escape_reward
        else:
            comp['escape'] = 0.0

        self.prev_observations[agent_id] = observation.copy()
        total_reward = float(np.clip(total_reward, -2.0, 2.0))

        self._last_components = getattr(self, "_last_components", {})
        self._last_components[agent_id] = comp

        # # 🔍 诊断8：最终结果
        # if food_contact > 0 or poison_contact > 0:
        #     print(f"   ✅ FINAL REWARD: {total_reward:.4f}")
        #     print(f"   Components: {comp}")
        #     print(f"{'*'*60}\n")

        return total_reward


class ForagingEscapeCalculator:
    """精简版性能 + 事件级会话指标（DESR/MDSD）"""
    def __init__(self, n_sensors=30, danger_zone=0.3, pred_health_frac=0.2, pred_inact_frames=150):
        self.n_sensors = n_sensors
        self.DANGER_ZONE = danger_zone
        self.PRED_HEALTH_FRAC = pred_health_frac
        self.PRED_INACT_FRAMES = pred_inact_frames

        self.stats = {}
        self.prev_states = {}
        self.in_session = defaultdict(lambda: False)
        self.session_len = defaultdict(lambda: 0)
        self.session_lens = defaultdict(list)
        self.session_survived = defaultdict(list)
        self.total_sessions = defaultdict(int)
        self.init_health = {}
        self.last_catch_frame = defaultdict(lambda: None)

    def process_step(self, agent_id, observation, step_info):
        if agent_id not in self.stats:
            self.stats[agent_id] = {
                'food_discovery': 0, 'food_success': 0,
                'threat_encounter': 0, 'escape_success': 0,
                'hunt_start': 0, 'hunt_success': 0, 'hunt_fail': 0
            }
            self.prev_states[agent_id] = {'food': False, 'predator': False, 'prey': False, 'danger': False}

        current = self._detect_current_state(observation)
        prev = self.prev_states[agent_id]
        agent_type = step_info.get('agent_type')
        alive = step_info.get('is_alive', True)
        frame = int(step_info.get('current_frame', 0))
        H = int(step_info.get('max_cycles', 1))

        if agent_type == 'predator':
            if agent_id not in self.init_health:
                self.init_health[agent_id] = max(1e-6, float(step_info.get('current_health', 100.0)))
            if bool(step_info.get('predator_catch', False)):
                self.last_catch_frame[agent_id] = frame

        if agent_type == 'prey':
            if not prev['food'] and current['food']:
                self.stats[agent_id]['food_discovery'] += 1
            if step_info.get('food_caught', False):
                self.stats[agent_id]['food_success'] += 1
            if not prev['predator'] and current['predator']:
                self.stats[agent_id]['threat_encounter'] += 1
            if prev['danger'] and not current['danger']:
                self.stats[agent_id]['escape_success'] += 1
        elif agent_type == 'predator':
            if not prev['prey'] and current['prey']:
                self.stats[agent_id]['hunt_start'] += 1
            if prev['prey'] and not current['prey'] and not step_info.get('predator_catch', False):
                self.stats[agent_id]['hunt_fail'] += 1
            if step_info.get('predator_catch', False):
                self.stats[agent_id]['hunt_success'] += 1

        hazard = self._compute_hazard_flag(agent_id, agent_type, current, step_info, frame)

        if (not self.in_session[agent_id]) and hazard:
            self.in_session[agent_id] = True
            self.session_len[agent_id] = 0
            self.total_sessions[agent_id] += 1

        if self.in_session[agent_id]:
            self.session_len[agent_id] += 1
            leaving = (not hazard)
            dying = (not alive)
            ending = (frame + 1 == H)

            if leaving or dying or ending:
                self.session_lens[agent_id].append(int(self.session_len[agent_id]))
                survived = (leaving or ending) and alive
                self.session_survived[agent_id].append(bool(survived))
                self.in_session[agent_id] = False
                self.session_len[agent_id] = 0

        self.prev_states[agent_id] = current

        out = self._calculate_speed_metrics(agent_id, agent_type)
        out.update(self._calculate_event_metrics(agent_id))
        return out

    def _detect_current_state(self, obs):
        food_detected = np.any(obs[60:90] < 0.8)
        agent_dists = obs[180:210]
        agent_types = obs[240:270]
        predator_mask = (agent_types == 0.0) & (agent_dists < 1.0)
        predator_detected = np.any(predator_mask)
        in_danger = False
        if predator_detected:
            in_danger = np.min(agent_dists[predator_mask]) < self.DANGER_ZONE
        prey_mask = (agent_types == 1.0) & (agent_dists < 1.0)
        prey_detected = np.any(prey_mask)
        return {
            'food': food_detected,
            'predator': predator_detected,
            'prey': prey_detected,
            'danger': in_danger
        }

    def _compute_hazard_flag(self, agent_id, agent_type, current, step_info, frame):
        if agent_type == 'prey':
            return bool(current['danger'])
        health = float(step_info.get('current_health', np.inf))
        thr = self._pred_health_threshold(agent_id)
        low_health = (health < thr)
        since = self._frames_since_last_catch(agent_id, frame, step_info.get('predator_catch', False))
        inactive = (since is not None and since > self.PRED_INACT_FRAMES)
        return bool(low_health or inactive)

    def _pred_health_threshold(self, agent_id):
        base = self.init_health.get(agent_id, 100.0)
        return max(1e-6, self.PRED_HEALTH_FRAC * base)

    def _frames_since_last_catch(self, agent_id, frame, catch_now):
        if catch_now:
            return 0
        last = self.last_catch_frame.get(agent_id, None)
        if last is None:
            return None
        return max(0, frame - last)

    def _calculate_speed_metrics(self, agent_id, agent_type):
        s = self.stats[agent_id]
        if agent_type == 'prey':
            return {
                'foraging_rate': s['food_success'] / max(1, s['food_discovery']),
                'escape_rate':   s['escape_success'] / max(1, s['threat_encounter'])
            }
        elif agent_type == 'predator':
            total_hunts = s['hunt_success'] + s['hunt_fail']
            return {'hunting_rate': s['hunt_success'] / max(1, total_hunts)}
        return {}

    def _calculate_event_metrics(self, agent_id):
        total = max(1, self.total_sessions[agent_id])
        desr = (sum(self.session_survived[agent_id]) / total)
        if len(self.session_lens[agent_id]) > 0:
            mdsd = float(np.median(self.session_lens[agent_id]))
        else:
            mdsd = 0.0
        return {'DESR': desr, 'MDSD': mdsd}

    def get_event_metrics(self, agent_id):
        return self._calculate_event_metrics(agent_id)

    def reset_all_stats(self):
        self.stats.clear()
        self.prev_states.clear()
        self.in_session.clear()
        self.session_len.clear()
        self.session_lens.clear()
        self.session_survived.clear()
        self.total_sessions.clear()
        self.init_health.clear()
        self.last_catch_frame.clear()


class WaterworldBase:
    def __init__(
        self,
        n_agents=None,
        n_predators=2,
        n_preys=3,
        agent_algorithms=None,
        static_food=True,
        static_poison=True,
        n_evaders=10,
        n_poisons=20,
        n_obstacles=1,
        n_coop=1,
        n_sensors=30,
        sensor_range=0.2,
        radius=0.015,
        obstacle_radius=0.1,
        predator_catch_reward=5.0,
        same_algo_reward=3.0,
        initial_health=100.0,
        obstacle_coord=[(0.5, 0.5)],
        agent_max_accel=0.5,
        agent_speed=0.2,
        evader_speed=0.1,
        poison_speed=0.1,
        poison_reward=10.0,
        food_reward=10.0,
        encounter_reward=0.01,
        thrust_penalty=-0.1,
        local_ratio=1.0,
        speed_features=True,
        max_cycles=500,
        predator_speed=None,    # ← 新增这行
        prey_speed=None,        # ← 新增这行
        render_mode=None,
        FPS=FPS,
    ):
        self.pixel_scale = 30 * 25 * 1
        self.clock = pygame.time.Clock()
        self.FPS = FPS

        self.handlers = []
        self.predator_catch_reward = predator_catch_reward
        self.same_algo_reward = same_algo_reward
        self.initial_health = initial_health
        self.evader_handlers = {}
        self.poison_handlers = {}
        self.n_coop = n_coop
        self.n_evaders = n_evaders
        self.n_obstacles = n_obstacles
        self.n_poisons = n_poisons
        self.n_predators = n_predators
        self.n_preys = n_preys
        self.agent_algorithms = agent_algorithms
        self.dead_agents = set()
        self.removed_agents = set()
        self.static_food = static_food
        self.static_poison = static_poison
        self.evaders_to_rebuild = set()
        self.poisons_to_rebuild = set()

        if n_agents is not None:
            self.num_agents = n_agents
            self.n_predators = n_agents // 2
            self.n_preys = n_agents - self.n_predators
        else:
            self.n_predators = n_predators
            self.n_preys = n_preys
            self.num_agents = n_predators + n_preys

        if agent_algorithms:
            unique_algorithms = sorted(set(agent_algorithms))
            self.algo_name_to_id = {name: idx for idx, name in enumerate(unique_algorithms)}
            self.agent_algo_ids = [self.algo_name_to_id[algo] for algo in agent_algorithms]
        else:
            self.algo_name_to_id = {"default": 0}
            self.agent_algo_ids = [0] * self.num_agents

        self.n_sensors = n_sensors
        self.base_radius = radius * self.pixel_scale
        self.obstacle_radius = obstacle_radius * self.pixel_scale  # ← 添加这行
        self.sensor_range = sensor_range * self.pixel_scale
        self.predator_speed = (predator_speed if predator_speed is not None else agent_speed) * self.pixel_scale
        self.prey_speed = (prey_speed if prey_speed is not None else agent_speed) * self.pixel_scale        
        self.agent_speed = agent_speed * self.pixel_scale
        self.evader_speed = evader_speed * self.pixel_scale
        self.poison_speed = poison_speed * self.pixel_scale
        self.speed_features = speed_features
        self.agent_max_accel = agent_max_accel
        self.encounter_reward = encounter_reward
        self.food_reward = food_reward
        self.local_ratio = local_ratio
        self.poison_reward = poison_reward
        self.thrust_penalty = thrust_penalty
        self.max_cycles = max_cycles

        self.control_rewards = [0 for _ in range(self.num_agents)]
        self.behavior_rewards = [0 for _ in range(self.num_agents)]
        self.collision_rewards = [0 for _ in range(self.num_agents)]
        self.last_dones = [bool(False) for _ in range(self.num_agents)]
        self.last_obs = [None for _ in range(self.num_agents)]
        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]
        self.step_movement_penalties = [0 for _ in range(self.num_agents)]
        self.reward_calculator = ObservationBasedRewardCalculator(n_sensors=self.n_sensors)
        self.cached_performance_metrics = {}

        if obstacle_coord is not None and len(obstacle_coord) != self.n_obstacles:
            raise ValueError("obstacle_coord does not have same length as n_obstacles")
        else:
            self.initial_obstacle_coord = obstacle_coord

        self.render_mode = render_mode
        self.screen = None
        self.frames = 0
        self.performance_calculator = ForagingEscapeCalculator(n_sensors=self.n_sensors)
        self.get_spaces()
        self._seed()

    def get_agent_type(self, agent_id):
        if agent_id < self.n_predators:
            return 'predator'
        else:
            return 'prey'

    def _generate_food_speed(self, max_speed):
        if self.static_food:
            return 0.0, 0.0
        else:
            return self._generate_speed(max_speed)

    def _generate_poison_speed(self, max_speed):
        if self.static_poison:
            return 0.0, 0.0
        else:
            return self._generate_speed(max_speed)

    def remove_agent_from_space(self, agent_id):
        if agent_id in self.removed_agents:
            return
        agent = self.agents[agent_id]
        if agent.body in self.space.bodies:
            self.space.remove(agent.body, agent.shape)
        self.remove_agent_collision_handlers(agent_id)
        self.removed_agents.add(agent_id)

    def _safe_perf(self, agent_id):
        out = dict(DEFAULT_PERF)
        d = self.cached_performance_metrics.get(agent_id, {})
        for k in out.keys():
            out[k] = float(d.get(k, 0.0))
        return out

    def remove_agent_collision_handlers(self, agent_id):
        agent = self.agents[agent_id]
        collision_type = agent.shape.collision_type
        handlers_to_remove = []
        for handler in self.handlers:
            if (hasattr(handler, 'type_a') and handler.type_a == collision_type) or \
               (hasattr(handler, 'type_b') and handler.type_b == collision_type):
                handlers_to_remove.append(handler)
        for handler in handlers_to_remove:
            try:
                self.space.remove(handler)
                self.handlers.remove(handler)
            except:
                pass

    def is_predator(self, agent_id):
        return agent_id < self.n_predators

    def is_prey(self, agent_id):
        return agent_id >= self.n_predators

    def get_spaces(self):
        if self.speed_features:
            obs_dim = 8 * self.n_sensors + 2 + 2 * self.n_sensors
        else:
            obs_dim = 5 * self.n_sensors + 2 + 2 * self.n_sensors

        obs_space = spaces.Box(
            low=np.float32(-np.inf),
            high=np.float32(np.inf),
            shape=(obs_dim,),
            dtype=np.float32,
        )

        act_space = spaces.Box(
            low=np.float32(-1.0),
            high=np.float32(1.0),
            shape=(2,),
            dtype=np.float32,
        )

        self.observation_space = [obs_space for i in range(self.num_agents)]
        self.action_space = [act_space for i in range(self.num_agents)]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def add_obj(self):
        self.agents = []
        self.evaders = []
        self.poisons = []
        self.obstacles = []

        for i in range(self.num_agents):
            x, y = self._generate_coord(self.base_radius)
            algo = self.agent_algorithms[i] if self.agent_algorithms else "default"

            if i < self.n_predators:
                agent = Predator(
                    x, y,
                    self.agent_max_accel,
                    self.predator_speed,  # ← 修改这行（原来是 self.agent_speed）

                    radius=self.base_radius,
                    collision_type=i + 1,
                    n_sensors=self.n_sensors,
                    sensor_range=self.sensor_range,
                    speed_features=self.speed_features,
                    pixel_scale=self.pixel_scale,  # ← 新增：传递场景尺度
                    algorithm=algo,
                )
            else:
                agent = Prey(
                    x, y,
                    self.agent_max_accel,
                    self.prey_speed,  # ← 修改这行（原来是 self.agent_speed）
                    radius=self.base_radius,
                    collision_type=i + 1,
                    n_sensors=self.n_sensors,
                    sensor_range=self.sensor_range,
                    speed_features=self.speed_features,
                    pixel_scale=self.pixel_scale,  # ← 新增：传递场景尺度
                    algorithm=algo,
                )
            
            self.agents.append(agent)
            agent.shape.health = self.initial_health

        for i in range(self.n_evaders):
            x, y = self._generate_coord(0.5 * self.base_radius)
            vx, vy = self._generate_food_speed(self.evader_speed)
            self.evaders.append(
                Evaders(
                    x, y, vx, vy,
                    radius=0.5 * self.base_radius,
                    collision_type=i + 1000,
                    max_speed=self.evader_speed,
                    is_static=self.static_food,
                    pixel_scale=self.pixel_scale,  # ← 新增
                )
            )

        for i in range(self.n_poisons):
            x, y = self._generate_coord(self.base_radius)
            vx, vy = self._generate_poison_speed(self.poison_speed)
            self.poisons.append(
                Poisons(
                    x, y, vx, vy,
                    radius=self.base_radius,
                    collision_type=i + 2000,
                    max_speed=self.poison_speed,
                    is_static=self.static_poison,
                    pixel_scale=self.pixel_scale,  # ← 新增
                )
            )

        for _ in range(self.n_obstacles):
            self.obstacles.append(
                Obstacle(
                    self.pixel_scale / 2,
                    self.pixel_scale / 2,
                    radius=self.obstacle_radius,
                    pixel_scale=self.pixel_scale,  # ← 新增
                )
            )

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def convert_coordinates(self, value, option="position"):
        if option == "position":
            return int(value[0]), self.pixel_scale - int(value[1])
        if option == "velocity":
            return value[0], -value[1]

    def _generate_coord(self, radius):
        coord = self.np_random.random(2) * self.pixel_scale
        for obstacle in self.obstacles:
            x, y = obstacle.body.position
            while (ssd.cdist(coord[None, :], np.array([[x, y]])) <= radius * 2 + obstacle.radius):
                coord = self.np_random.random(2) * self.pixel_scale
        return coord

    def _generate_speed(self, speed):
        _speed = (self.np_random.random(2) - 0.5) * 2 * speed
        return _speed[0], _speed[1]

    def add(self):
        self.space = pymunk.Space()
        for i, agent in enumerate(self.agents):
            if i not in self.removed_agents:
                 agent.add(self.space)
        for obj_list in [self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.add(self.space)
    def add_bounding_box(self):
        # 边界坐标：左(L)、下(B)、右(R)、上(T)
        L, B = 0, 0
        R, T = self.pixel_scale, self.pixel_scale
        
        # 墙体厚度：随场景缩放（建议 0.3~0.5% 的场景尺寸）
        # 或者固定 3~5 像素也可以
        thickness = max(3, int(self.pixel_scale * 0.02))
        
        self.barriers = []
        
        # 四条边的起止点（逆时针：下→右→上→左）
        segments = [
            ((L, B), (R, B)),  # 底边
            ((R, B), (R, T)),  # 右边
            ((R, T), (L, T)),  # 顶边
            ((L, T), (L, B)),  # 左边
        ]
        
        for start, end in segments:
            seg = pymunk.Segment(
                self.space.static_body, 
                start, 
                end, 
                thickness  # ✅ 关键：用很薄的半径代替原来的 100
            )
            seg.elasticity = 0.999
            # seg.friction = 0.5  # 可选：防止高速在边缘打滑
            self.space.add(seg)
            self.barriers.append(seg)

    def draw(self):
        for i, agent in enumerate(self.agents):
            if i not in self.removed_agents:
                agent.draw(self.screen, self.convert_coordinates)
        for obj_list in [self.evaders, self.poisons, self.obstacles]:
            for obj in obj_list:
                obj.draw(self.screen, self.convert_coordinates)

    def add_handlers(self):
        self.handlers = []
        self.evader_handlers = {}
        self.poison_handlers = {}

        for idx, evader in enumerate(self.evaders):
            self._register_evader_handlers(evader, idx)

        for idx, poison in enumerate(self.poisons):
            self._register_poison_handlers(poison, idx)

        for i in range(len(self.agents)):
            for j in range(i + 1, len(self.agents)):
                handler = self.space.add_collision_handler(
                    self.agents[i].shape.collision_type,
                    self.agents[j].shape.collision_type,
                )
                handler.begin = self.agent_agent_begin_callback
                self.handlers.append(handler)
    
    def get_step_info(self, agent_id):
        if agent_id >= len(self.agents) or agent_id < 0:
            return {}
        
        agent = self.agents[agent_id]
        death_cause = None
        if agent_id in self.dead_agents:
            if agent.shape.prey_caught_indicator > 0:
                death_cause = "caught"
            elif agent.shape.poison_indicator > 0:
                death_cause = "poison"
            elif agent.shape.health <= 0:
                death_cause = "health_depleted"
        
        algorithm = "default"
        if self.agent_algorithms and agent_id < len(self.agent_algorithms):
            algorithm = self.agent_algorithms[agent_id]
        
        step_info = {
            "current_health": float(agent.shape.health),
            "is_alive": agent_id not in self.dead_agents,
            "max_cycles": int(self.max_cycles),
            "current_frame": int(self.frames),
            "episode_progress": float(self.frames / self.max_cycles),
            "food_caught": bool(agent.shape.food_indicator > 0),
            "food_touched": bool(agent.shape.food_touched_indicator > 0),
            "poison_contacted": bool(agent.shape.poison_indicator > 0),
            "predator_catch": bool(agent.shape.predator_catch_indicator > 0),
            "prey_caught": bool(agent.shape.prey_caught_indicator > 0),
            "death_cause": death_cause,
            "agent_type": self.get_agent_type(agent_id),
            "algorithm": algorithm
        }
        step_info["performance_metrics"] = self._safe_perf(agent_id)
        return step_info

    def reset(self):
        self.performance_calculator.reset_all_stats()
        self.reward_calculator.prev_observations.clear()
        self.reward_calculator.prev_flags.clear()  # ✅ 新增：清空 prev_flags
        self.cached_performance_metrics.clear()
        self.dead_agents.clear()
        self.removed_agents.clear()
        self.evaders_to_rebuild = set()
        self.poisons_to_rebuild = set()
        self.handlers = []
        self.evader_handlers = {}
        self.poison_handlers = {}
        self.add_obj()
        self.frames = 0

        if self.initial_obstacle_coord is None:
            for i, obstacle in enumerate(self.obstacles):
                obstacle_position = (
                    self.np_random.random((self.n_obstacles, 2)) * self.pixel_scale
                )
                obstacle.body.position = (
                    obstacle_position[0, 0],
                    obstacle_position[0, 1],
                )
        else:
            for i, obstacle in enumerate(self.obstacles):
                obstacle.body.position = (
                    self.initial_obstacle_coord[i][0] * self.pixel_scale,
                    self.initial_obstacle_coord[i][1] * self.pixel_scale,
                )

        self.add()
        self.add_handlers()
        self.add_bounding_box()

        obs_list = self.observe_list()

        self.last_rewards = [np.float64(0) for _ in range(self.num_agents)]
        self.control_rewards = [0 for _ in range(self.num_agents)]
        self.behavior_rewards = [0 for _ in range(self.num_agents)]
        self.collision_rewards = [0 for _ in range(self.num_agents)]
        self.last_dones = [bool(False) for _ in range(self.num_agents)]
        self.last_obs = obs_list
        for agent in self.agents:
            agent.shape.health = self.initial_health

        return obs_list[0]

    def step(self, action, agent_id, is_last):
        # ✅ 关键修复：死亡 agent 立即返回
        if agent_id in self.dead_agents:
            return self.observe(agent_id)
        
        # ✅ 行动前检查
        p = self.agents[agent_id]
        if p.shape.health <= 0:
            self.last_dones[agent_id] = True
            self.dead_agents.add(agent_id)
            self.remove_agent_from_space(agent_id)
            return self.observe(agent_id)
        
        action = np.asarray(action) * self.agent_max_accel
        action = action.reshape(2)

        thrust = np.linalg.norm(action)
        if thrust > self.agent_max_accel:
            action = action * (self.agent_max_accel / thrust)

        p = self.agents[agent_id]

        _velocity = np.clip(
            p.body.velocity + action * self.pixel_scale,
            -self.agent_speed,
            self.agent_speed,
        )

        p.reset_velocity(_velocity[0], _velocity[1])

        accel_penalty = self.thrust_penalty * math.sqrt((action**2).sum())

        self.control_rewards = (
            (accel_penalty / self.num_agents)
            * np.ones(self.num_agents)
            * (1 - self.local_ratio)
        )
        
        self.control_rewards[agent_id] += accel_penalty * self.local_ratio
        movement_penalty = accel_penalty * self.local_ratio
        
        self.step_movement_penalties[agent_id] = movement_penalty

        if is_last:
            self.space.step(1 / self.FPS)
            self._process_rebuild_queues()
            obs_list = self.observe_list()
            self.last_obs = obs_list
            self.cached_performance_metrics = {}
            
            # ✅ 第1步：先缓存 step_info（此时indicators还未清零）
            cached_step_infos = {}
            for id in range(self.num_agents):
                cached_step_infos[id] = self.get_step_info(id)
            
            # ✅ 第2步：处理每个agent
            for id in range(self.num_agents):
                # 2.1 计算性能指标
                if id not in self.dead_agents:
                    perf = self.performance_calculator.process_step(
                        agent_id=id,
                        observation=obs_list[id],
                        step_info=cached_step_infos[id]
                    )
                    self.cached_performance_metrics[id] = perf
                else:
                    self.cached_performance_metrics[id] = dict(DEFAULT_PERF)
                
                p = self.agents[id]
                observation = obs_list[id]
                
                # 2.2 更新健康值（使用indicators）
                health_before = p.shape.health
                
                if p.shape.prey_caught_indicator > 0:
                    p.shape.health = 0
                elif p.shape.health > 0:
                    base_decay = self.thrust_penalty * 0.5 if self.is_predator(id) else self.thrust_penalty * 0.8
                    p.shape.health += base_decay
                    
                    if p.shape.predator_catch_indicator > 0 and self.is_predator(id):
                        p.shape.health += self.predator_catch_reward
                    
                    if (p.shape.food_indicator > 0 or p.shape.poison_indicator > 0) and self.is_prey(id):
                        p.shape.health += self.food_reward
                
                # 2.3 计算奖励（indicators还保留，健康值已更新）
                if self.is_predator(id):
                    self.behavior_rewards[id] = self.reward_calculator.calculate_predator_reward_from_obs(
                        agent_id=id,
                        observation=observation,
                        catch_indicator=p.shape.predator_catch_indicator,
                        catch_active=p.shape.predator_catch_indicator > 0
                    )
                else:
                    self.behavior_rewards[id] = self.reward_calculator.calculate_prey_reward_from_obs(
                        agent_id=id,
                        observation=observation,
                        food_contact=p.shape.food_indicator,
                        food_touching=p.shape.food_touched_indicator > 0,
                        poison_contact=p.shape.poison_indicator,
                        poison_touching=p.shape.poison_indicator > 0,
                        caught_indicator=p.shape.prey_caught_indicator
                    )
                
                self.collision_rewards[id] = 0
                
                # 2.4 更新prev_flags（基于touching状态）
                if self.is_predator(id):
                    self.reward_calculator.prev_flags[id]['caught'] = 1 if p.shape.predator_catch_indicator > 0 else 0
                else:
                    self.reward_calculator.prev_flags[id]['food'] = 1 if p.shape.food_touched_indicator > 0 else 0
                    self.reward_calculator.prev_flags[id]['poison'] = 1 if p.shape.poison_indicator > 0 else 0
                
                # 2.5 清零indicators
                p.shape.food_indicator = 0
                p.shape.poison_indicator = 0
                p.shape.predator_catch_indicator = 0
                p.shape.prey_caught_indicator = 0
                p.shape.same_algo_meet_indicator = 0
                p.shape.diff_algo_meet_indicator = 0
                
                # 2.6 死亡检查
                if p.shape.health <= 0:
                    self.last_dones[id] = True
                    self.dead_agents.add(id)
                    self.remove_agent_from_space(id)
                else:
                    if not self.last_dones[id]:
                        self.last_dones[id] = False
            
            # ✅ 第3步：保存cached_step_infos供后续使用
            self._cached_step_infos = cached_step_infos
            
            # ✅ 第4步：裁剪奖励
            for id in range(self.num_agents):
                self.behavior_rewards[id] = np.clip(self.behavior_rewards[id], -2.0, 4.0)
                self.collision_rewards[id] = np.clip(self.collision_rewards[id], -2.0, 4.0)
            
            # ✅ 第5步：计算最终奖励
            rewards = np.array(self.behavior_rewards) + np.array(self.control_rewards) + np.array(self.collision_rewards)
            local_reward = rewards
            global_reward = local_reward.mean()
            
            self.last_rewards = local_reward * self.local_ratio + global_reward * (1 - self.local_ratio)
            self.frames += 1

        return self.observe(agent_id)
    
    def observe(self, agent_id):
        return np.array(self.last_obs[agent_id], dtype=np.float32)

    def observe_list(self):
        observe_list = []

        for i, agent in enumerate(self.agents):
            obstacle_distances = []
            evader_distances = []
            evader_velocities = []
            poison_distances = []
            poison_velocities = []
            _agent_distances = []
            _agent_velocities = []
            _agent_types = []
            _agent_ids = []

            for obstacle in self.obstacles:
                obstacle_distance, _ = agent.get_sensor_reading(
                    obstacle.body.position, obstacle.radius, obstacle.body.velocity, 0.0
                )
                obstacle_distances.append(obstacle_distance)

            obstacle_sensor_vals = self.get_sensor_readings(
                obstacle_distances, agent.sensor_range
            )

            barrier_distances = agent.get_sensor_barrier_readings()

            for evader in self.evaders:
                evader_distance, evader_velocity = agent.get_sensor_reading(
                    evader.body.position,
                    evader.radius,
                    evader.body.velocity,
                    self.evader_speed,
                )
                evader_distances.append(evader_distance)
                evader_velocities.append(evader_velocity)

            (
                evader_sensor_distance_vals,
                evader_sensor_velocity_vals,
            ) = self.get_sensor_readings(
                evader_distances,
                agent.sensor_range,
                velocites=evader_velocities,
            )

            for poison in self.poisons:
                poison_distance, poison_velocity = agent.get_sensor_reading(
                    poison.body.position,
                    poison.radius,
                    poison.body.velocity,
                    self.poison_speed,
                )
                poison_distances.append(poison_distance)
                poison_velocities.append(poison_velocity)

            (
                poison_sensor_distance_vals,
                poison_sensor_velocity_vals,
            ) = self.get_sensor_readings(
                poison_distances,
                agent.sensor_range,
                velocites=poison_velocities,
            )

            if self.num_agents > 1:
                for j, _agent in enumerate(self.agents):
                    if i == j:
                        continue

                    _agent_distance, _agent_velocity, _agent_type, _agent_id = agent.get_sensor_reading(
                        _agent.body.position,
                        _agent.radius,
                        _agent.body.velocity,
                        self.agent_speed,
                        object_type=0 if j < self.n_predators else 1,
                        object_id=j / self.num_agents
                    )
                    _agent_distances.append(_agent_distance)
                    _agent_velocities.append(_agent_velocity)
                    _agent_types.append(_agent_type)
                    _agent_ids.append(_agent_id)

                (
                    _agent_sensor_distance_vals,
                    _agent_sensor_velocity_vals,
                    _agent_sensor_type_vals,
                    _agent_sensor_id_vals,
                ) = self.get_sensor_readings(
                    _agent_distances,
                    agent.sensor_range,
                    velocites=_agent_velocities,
                    types=_agent_types,
                    ids=_agent_ids,
                )
            else:
                _agent_sensor_distance_vals = np.zeros(self.n_sensors)
                _agent_sensor_velocity_vals = np.zeros(self.n_sensors)
                _agent_sensor_type_vals = np.full(self.n_sensors, -1.0)
                _agent_sensor_id_vals = np.full(self.n_sensors, -1.0)

            if agent.shape.food_touched_indicator >= 1:
                food_obs = 1
            else:
                food_obs = 0

            if agent.shape.poison_indicator >= 1:
                poison_obs = 1
            else:
                poison_obs = 0

            if self.speed_features:
                agent_observation = np.concatenate(
                    [
                        obstacle_sensor_vals,
                        barrier_distances,
                        evader_sensor_distance_vals,
                        evader_sensor_velocity_vals,
                        poison_sensor_distance_vals,
                        poison_sensor_velocity_vals,
                        _agent_sensor_distance_vals,
                        _agent_sensor_velocity_vals,
                        _agent_sensor_type_vals,
                        _agent_sensor_id_vals,
                        np.array([food_obs]),
                        np.array([poison_obs]),
                    ]
                )
            else:
                agent_observation = np.concatenate(
                    [
                        obstacle_sensor_vals,
                        barrier_distances,
                        evader_sensor_distance_vals,
                        poison_sensor_distance_vals,
                        _agent_sensor_distance_vals,
                        _agent_sensor_type_vals,
                        _agent_sensor_id_vals,
                        np.array([food_obs]),
                        np.array([poison_obs]),
                    ]
                )

            observe_list.append(agent_observation)

        return observe_list

    def get_sensor_readings(self, positions, sensor_range, velocites=None, types=None, ids=None):
        distance_vals = np.concatenate(positions, axis=1)
        min_idx = np.argmin(distance_vals, axis=1)
        sensor_distance_vals = np.amin(distance_vals, axis=1)

        if velocites is not None:
            velocity_vals = np.concatenate(velocites, axis=1)
            sensor_velocity_vals = velocity_vals[np.arange(self.n_sensors), min_idx]
            if types is not None and ids is not None:
                type_vals = np.column_stack(types)
                id_vals = np.column_stack(ids)
                sensor_type_vals = type_vals[np.arange(self.n_sensors), min_idx]
                sensor_id_vals = id_vals[np.arange(self.n_sensors), min_idx]
                return sensor_distance_vals, sensor_velocity_vals, sensor_type_vals, sensor_id_vals
            else:
                return sensor_distance_vals, sensor_velocity_vals

        return sensor_distance_vals

    def _clear_object_handlers(self, obj_type, obj_id):
        if obj_type == 'evader':
            handlers = self.evader_handlers.get(obj_id, [])
        elif obj_type == 'poison':
            handlers = self.poison_handlers.get(obj_id, [])
        else:
            return
        
        for handler in handlers:
            try:
                if handler in self.handlers:
                    self.handlers.remove(handler)
            except:
                pass
        
        if obj_type == 'evader':
            self.evader_handlers[obj_id] = []
        elif obj_type == 'poison':
            self.poison_handlers[obj_id] = []

    def _register_evader_handlers(self, evader, evader_idx):
        """为食物注册碰撞处理器"""
        handlers_for_this_evader = []
        
        # 为所有存活的 Agent 注册处理器
        for i, agent in enumerate(self.agents):
            if i not in self.removed_agents:
                handler = self.space.add_collision_handler(
                    agent.shape.collision_type, 
                    evader.shape.collision_type
                )
                
                if self.is_prey(i):
                    # Prey 可以吃食物
                    handler.begin = self.agent_evader_begin_callback
                    handler.separate = self.agent_evader_separate_callback
                else:
                    # Predator 穿过食物（不发生物理碰撞）
                    handler.begin = self.return_false_begin_callback
                
                self.handlers.append(handler)
                handlers_for_this_evader.append(handler)
        
        # 食物与毒物之间不碰撞
        for poison in self.poisons:
            handler = self.space.add_collision_handler(
                poison.shape.collision_type, 
                evader.shape.collision_type
            )
            handler.begin = self.return_false_begin_callback
            self.handlers.append(handler)
            handlers_for_this_evader.append(handler)
        
        # 食物之间不碰撞
        for other_evader in self.evaders:
            if other_evader.shape != evader.shape:
                handler = self.space.add_collision_handler(
                    evader.shape.collision_type,
                    other_evader.shape.collision_type
                )
                handler.begin = self.return_false_begin_callback
                self.handlers.append(handler)
                handlers_for_this_evader.append(handler)
        
        self.evader_handlers[evader_idx] = handlers_for_this_evader
   
    def _register_poison_handlers(self, poison, poison_idx):
        """为毒物注册碰撞处理器"""
        handlers_for_this_poison = []
        
        # 为所有存活的 Agent 注册处理器
        for i, agent in enumerate(self.agents):
            if i not in self.removed_agents:
                handler = self.space.add_collision_handler(
                    agent.shape.collision_type,
                    poison.shape.collision_type
                )
                
                if self.is_prey(i):
                    # Prey 会被毒物影响
                    handler.begin = self.agent_poison_begin_callback
                else:
                    # Predator 穿过毒物（不发生物理碰撞）
                    handler.begin = self.return_false_begin_callback
                
                self.handlers.append(handler)
                handlers_for_this_poison.append(handler)
        
        # 毒物之间不碰撞
        for other_poison in self.poisons:
            if other_poison.shape != poison.shape:
                handler = self.space.add_collision_handler(
                    poison.shape.collision_type,
                    other_poison.shape.collision_type
                )
                handler.begin = self.return_false_begin_callback
                self.handlers.append(handler)
                handlers_for_this_poison.append(handler)
        
        self.poison_handlers[poison_idx] = handlers_for_this_poison   
   
    def _rebuild_evader(self, evader_shape):
        evader = evader_shape.parent_evader
        evader_idx = evader_shape.collision_type - 1000
        if evader_idx < 0 or evader_idx >= len(self.evaders):
            return
        self._clear_object_handlers('evader', evader_idx)
        x, y = self._generate_coord(evader.radius)
        vx, vy = self._generate_food_speed(evader.shape.max_speed)
        evader.rebuild_in_place(self.space, x, y, vx, vy)
        self._register_evader_handlers(evader, evader_idx)

    def _rebuild_poison(self, poison_shape):
        poison = poison_shape.parent_poison
        poison_idx = poison_shape.collision_type - 2000
        if poison_idx < 0 or poison_idx >= len(self.poisons):
            return
        self._clear_object_handlers('poison', poison_idx)
        x, y = self._generate_coord(poison.radius)
        vx, vy = self._generate_poison_speed(poison.shape.max_speed)
        poison.rebuild_in_place(self.space, x, y, vx, vy)
        self._register_poison_handlers(poison, poison_idx)

    def _process_rebuild_queues(self):
        for evader_idx in list(self.evaders_to_rebuild):
            if 0 <= evader_idx < len(self.evaders):
                evader = self.evaders[evader_idx]
                if evader.body in self.space.bodies:
                    self._clear_object_handlers('evader', evader_idx)
                    x, y = self._generate_coord(evader.radius)
                    vx, vy = self._generate_food_speed(evader.shape.max_speed)
                    evader.rebuild_in_place(self.space, x, y, vx, vy)
                    evader.shape.counter = 0
                    self._register_evader_handlers(evader, evader_idx)
        
        for poison_idx in list(self.poisons_to_rebuild):
            if 0 <= poison_idx < len(self.poisons):
                poison = self.poisons[poison_idx]
                if poison.body in self.space.bodies:
                    self._clear_object_handlers('poison', poison_idx)
                    x, y = self._generate_coord(poison.radius)
                    vx, vy = self._generate_poison_speed(poison.shape.max_speed)
                    poison.rebuild_in_place(self.space, x, y, vx, vy)
                    self._register_poison_handlers(poison, poison_idx)
        
        self.evaders_to_rebuild.clear()
        self.poisons_to_rebuild.clear()

    def agent_evader_begin_callback(self, arbiter, space, data):
        agent_shape, evader_shape = arbiter.shapes
        agent_id = agent_shape.collision_type - 1
        
        # print(f"\n{'='*60}")
        # print(f"🥕 [Frame {self.frames}] COLLISION START: Agent {agent_id}")
        # print(f"   counter BEFORE: {evader_shape.counter}")
        # print(f"   food_touched_indicator BEFORE: {agent_shape.food_touched_indicator}")
        # print(f"   food_indicator BEFORE: {agent_shape.food_indicator}")
        
        evader_shape.counter += 1
        agent_shape.food_touched_indicator += 1

        if evader_shape.counter >= self.n_coop:
            agent_shape.food_indicator = 1
        #     print(f"   ✅ FOOD_INDICATOR SET TO 1 (n_coop={self.n_coop} reached)")
        # else:
        #     print(f"   ⏳ Waiting for n_coop ({evader_shape.counter}/{self.n_coop})")
        
        # print(f"   counter AFTER: {evader_shape.counter}")
        # print(f"   food_touched_indicator AFTER: {agent_shape.food_touched_indicator}")
        # print(f"   food_indicator AFTER: {agent_shape.food_indicator}")
        # print(f"{'='*60}\n")
        
        return False

    def agent_evader_separate_callback(self, arbiter, space, data):
        agent_shape, evader_shape = arbiter.shapes
        agent_id = agent_shape.collision_type - 1

        # print(f"\n{'='*60}")
        # print(f"⬅️ [Frame {self.frames}] COLLISION END: Agent {agent_id}")
        # print(f"   counter BEFORE: {evader_shape.counter}, n_coop: {self.n_coop}")

        if evader_shape.counter < self.n_coop:
            evader_shape.counter -= 1
            # print(f"   ⬇️ Agent left food (counter now: {evader_shape.counter})")
        else:
            evader_idx = evader_shape.collision_type - 1000
            self.evaders_to_rebuild.add(evader_idx)
            evader_shape.counter = 0
            # print(f"   🎉 Food CONSUMED! Resetting counter, rebuilding evader {evader_idx}")

        agent_shape.food_touched_indicator -= 1
        # print(f"   food_touched_indicator now: {agent_shape.food_touched_indicator}")
        # print(f"{'='*60}\n")
        
        return False

    def agent_poison_begin_callback(self, arbiter, space, data):
        agent_shape, poison_shape = arbiter.shapes
        agent_id = agent_shape.collision_type - 1

        # print(f"\n{'='*60}")
        # print(f"☠️ [Frame {self.frames}] POISON COLLISION: Agent {agent_id}")
        # print(f"   poison_indicator BEFORE: {agent_shape.poison_indicator}")
        
        agent_shape.poison_indicator += 1
        
        # print(f"   poison_indicator AFTER: {agent_shape.poison_indicator}")

        poison_idx = poison_shape.collision_type - 2000
        self.poisons_to_rebuild.add(poison_idx)
        # print(f"   Rebuilding poison {poison_idx}")
        # print(f"{'='*60}\n")

        return False

    def agent_agent_begin_callback(self, arbiter, space, data):
        agent_shape1, agent_shape2 = arbiter.shapes
        
        agent1_id = agent_shape1.collision_type - 1
        agent2_id = agent_shape2.collision_type - 1
        
        agent1_is_predator = self.is_predator(agent1_id)
        agent2_is_predator = self.is_predator(agent2_id)
        
        if agent1_is_predator and not agent2_is_predator:
            agent_shape1.predator_catch_indicator = 1
            agent_shape2.prey_caught_indicator = 1
            
        elif not agent1_is_predator and agent2_is_predator:
            agent_shape1.prey_caught_indicator = 1
            agent_shape2.predator_catch_indicator = 1
        
        return False

    def return_false_begin_callback(self, arbiter, space, data):
        return False

    def get_last_dones(self):
        return dict(zip(list(range(self.num_agents)), [bool(x) for x in self.last_dones]))

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.pixel_scale, self.pixel_scale)
                )
                pygame.display.set_caption("Waterworld")
            else:
                self.screen = pygame.Surface((self.pixel_scale, self.pixel_scale))

        self.screen.fill((255, 255, 255))
        self.draw()
        self.clock.tick(self.FPS)

        observation = pygame.surfarray.pixels3d(self.screen)
        new_observation = np.copy(observation)
        del observation

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return (
            np.transpose(new_observation, axes=(1, 0, 2))
            if self.render_mode == "rgb_array"
            else None
        )