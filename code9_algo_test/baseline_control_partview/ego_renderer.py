import pygame
import numpy as np
import math
from collections import defaultdict

class EgoRenderer:
    """独立的 Ego 视图渲染器 - 只基于观测信息显示传感器世界"""
    
    def __init__(self, n_sensors=30, window_size=900, danger_zone=0.3):
        """
        参数:
            n_sensors: 传感器数量（对应观测向量的扇区数）
            window_size: 窗口大小（正方形）
            danger_zone: 危险区域半径（归一化值，0-1）
        """
        self.n_sensors = n_sensors
        self.window_size = window_size
        self.danger_zone = danger_zone
        
        # 初始化 Pygame
        if not pygame.get_init():
            pygame.init()
        
        # 创建独立窗口
        self.screen = pygame.Surface((window_size, window_size))
        self.display = pygame.display.set_mode((window_size, window_size))
        pygame.display.set_caption("Ego View - Sensor Radar")
        
        # 🔥 强制启用键盘输入捕获（修复 Linux 问题）
        pygame.event.set_allowed([pygame.QUIT, pygame.KEYDOWN, pygame.KEYUP])
        pygame.key.set_repeat(0)  # 禁用按键重复
        
        # 尝试获取窗口焦点
        try:
            pygame.display.iconify()  # 最小化
            pygame.time.wait(50)
            pygame.display.toggle_fullscreen()  # 退出全屏（如果有）
            pygame.display.toggle_fullscreen()  # 再次切换以刷新
        except:
            pass
        
        # 字体
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_large = pygame.font.Font(None, 36)
        
        # 计算传感器角度（均匀分布 0 到 2π）
        self.sensor_angles = np.linspace(0, 2 * np.pi, n_sensors + 1)[:-1]
        
        # 雷达盘参数
        self.center = (window_size // 2, window_size // 2)
        self.max_radius = window_size // 2 - 100  # 留出边距给 HUD
        
        # 多环半径（从内到外）
        self.ring_radii = {
            'agents': self.max_radius * 0.85,      # 最外环：其他 agent
            'food': self.max_radius * 0.68,        # 食物
            'poison': self.max_radius * 0.51,      # 毒物
            'barrier': self.max_radius * 0.34,     # 边界
            'obstacle': self.max_radius * 0.17,    # 最内环：障碍物
        }
        
        # 颜色配置
        self.colors = {
            'background': (20, 25, 35),
            'grid': (50, 55, 65),
            'obstacle': (120, 120, 120),
            'barrier': (80, 80, 100),
            'food': (50, 200, 50),
            'poison': (220, 50, 50),
            'predator': (200, 30, 30),
            'prey': (50, 120, 220),
            'danger_zone': (255, 50, 50),
            'text': (220, 220, 220),
            'text_dim': (150, 150, 150),
            'health_high': (50, 200, 50),
            'health_mid': (220, 220, 50),
            'health_low': (220, 50, 50),
        }
        
        # 观测空间索引（基于你的环境定义）
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
        
        # 图层开关状态
        self.show_layers = {
            'obstacle': True,
            'barrier': True,
            'food': True,
            'poison': True,
            'agents': True,
            'velocity': True,
            'danger_zone': True,
            'strategy_pointer': True,
        }
        
        self.clock = pygame.time.Clock()
    
    def extract_obs_features(self, observation):
        """从观测向量中提取各部分特征"""
        features = {}
        for key, indices in self.obs_indices.items():
            if isinstance(indices, tuple):
                start, end = indices
                features[key] = observation[start:end]
            else:
                features[key] = observation[indices]
        return features
    
    def render(self, observation, info):
        """主渲染函数"""
        # 清空屏幕
        self.screen.fill(self.colors['background'])
        
        # 提取观测特征
        features = self.extract_obs_features(observation)
        
        # 绘制网格和标尺
        self.draw_grid()
        
        # 绘制危险区域（仅 prey）
        agent_type = info.get('agent_type', 'unknown')
        if agent_type == 'prey' and self.show_layers['danger_zone']:
            self.draw_danger_zone()
        
        # 绘制各层雷达数据
        if self.show_layers['obstacle']:
            self.draw_radar_ring(features['obstacle_dist'], 
                                self.ring_radii['obstacle'], 
                                self.colors['obstacle'], 'Obstacle')
        
        if self.show_layers['barrier']:
            self.draw_radar_ring(features['barrier_dist'], 
                                self.ring_radii['barrier'], 
                                self.colors['barrier'], 'Barrier')
        
        if self.show_layers['food']:
            self.draw_radar_ring(features['food_dist'], 
                                self.ring_radii['food'], 
                                self.colors['food'], 'Food',
                                velocity=features['food_vel'] if self.show_layers['velocity'] else None)
        
        if self.show_layers['poison']:
            self.draw_radar_ring(features['poison_dist'], 
                                self.ring_radii['poison'], 
                                self.colors['poison'], 'Poison',
                                velocity=features['poison_vel'] if self.show_layers['velocity'] else None)
        
        if self.show_layers['agents']:
            self.draw_agent_ring(features['agent_dist'], 
                                features['agent_vel'] if self.show_layers['velocity'] else None,
                                features['agent_type'],
                                features['agent_id'])
        
        # 绘制策略指针
        if self.show_layers['strategy_pointer']:
            self.draw_strategy_pointer(features, agent_type)
        
        # 绘制 HUD
        self.draw_hud(info, features)
        
        # 更新显示
        self.display.blit(self.screen, (0, 0))
        pygame.display.flip()
        # 不要在这里调用 pygame.event.pump()，让控制器统一处理事件
    
    def draw_grid(self):
        """绘制极坐标网格和方位标记"""
        center_x, center_y = self.center
        
        # 绘制同心圆
        for radius in [r for r in self.ring_radii.values()]:
            pygame.draw.circle(self.screen, self.colors['grid'], 
                             self.center, int(radius), 1)
        
        # 绘制方位线（每 30° 一条）
        for i in range(12):
            angle = i * np.pi / 6
            end_x = center_x + self.max_radius * np.cos(angle - np.pi/2)
            end_y = center_y + self.max_radius * np.sin(angle - np.pi/2)
            pygame.draw.line(self.screen, self.colors['grid'], 
                           self.center, (int(end_x), int(end_y)), 1)
        
        # 绘制方位标记（N, E, S, W）
        directions = [
            (0, 'N', (0, -1)),
            (90, 'E', (1, 0)),
            (180, 'S', (0, 1)),
            (270, 'W', (-1, 0))
        ]
        
        for angle_deg, label, offset in directions:
            angle = np.radians(angle_deg - 90)  # 调整为数学坐标系
            text_radius = self.max_radius + 20
            pos_x = center_x + text_radius * np.cos(angle)
            pos_y = center_y + text_radius * np.sin(angle)
            text = self.font_medium.render(label, True, self.colors['text'])
            text_rect = text.get_rect(center=(int(pos_x), int(pos_y)))
            self.screen.blit(text, text_rect)
    
    def draw_danger_zone(self):
        """绘制危险区域圈（prey 专用）"""
        danger_radius = int(self.max_radius * self.danger_zone)
        pygame.draw.circle(self.screen, self.colors['danger_zone'], 
                         self.center, danger_radius, 2)
        
        # 标签
        text = self.font_small.render("DANGER ZONE", True, self.colors['danger_zone'])
        text_rect = text.get_rect(center=(self.center[0], self.center[1] + danger_radius + 15))
        self.screen.blit(text, text_rect)
    
    def draw_radar_ring(self, distances, ring_radius, color, label, velocity=None):
        """
        绘制一个雷达环
        
        参数:
            distances: 各扇区的距离值（归一化，0-1）
            ring_radius: 该环的半径
            color: 颜色
            label: 标签
            velocity: 可选的速度值（用于绘制箭头）
        """
        center_x, center_y = self.center
        
        for i, distance in enumerate(distances):
            if distance >= 0.99:  # 未检测到
                continue
            
            angle = self.sensor_angles[i] - np.pi/2  # 转换为屏幕坐标（0度朝上）
            
            # 计算条的长度（距离越近，条越长）
            bar_length = ring_radius * (1 - distance) * 0.95
            
            if bar_length < 2:  # 太短不绘制
                continue
            
            # 起点（在环上）
            start_x = center_x + ring_radius * np.cos(angle) * 0.1
            start_y = center_y + ring_radius * np.sin(angle) * 0.1
            
            # 终点
            end_x = center_x + (ring_radius * 0.1 + bar_length) * np.cos(angle)
            end_y = center_y + (ring_radius * 0.1 + bar_length) * np.sin(angle)
            
            # 绘制条
            pygame.draw.line(self.screen, color, 
                           (int(start_x), int(start_y)), 
                           (int(end_x), int(end_y)), 3)
            
            # 绘制速度箭头
            if velocity is not None and abs(velocity[i]) > 0.1:
                self.draw_velocity_arrow(end_x, end_y, angle, velocity[i], color)
    
    def draw_velocity_arrow(self, x, y, angle, velocity, color):
        """绘制速度箭头（向内=接近，向外=远离）"""
        # 箭头长度与速度成比例
        arrow_length = min(abs(velocity) * 15, 20)
        
        # 方向：velocity > 0 表示接近（向内），< 0 表示远离（向外）
        direction = 1 if velocity < 0 else -1
        
        # 箭头终点
        arrow_end_x = x + direction * arrow_length * np.cos(angle)
        arrow_end_y = y + direction * arrow_length * np.sin(angle)
        
        # 绘制箭头线
        pygame.draw.line(self.screen, color, 
                       (int(x), int(y)), 
                       (int(arrow_end_x), int(arrow_end_y)), 2)
        
        # 绘制箭头头部
        arrow_angle = angle + (np.pi if direction > 0 else 0)
        head_size = 5
        head_angles = [arrow_angle + np.pi/6, arrow_angle - np.pi/6]
        
        for head_angle in head_angles:
            head_x = arrow_end_x + head_size * np.cos(head_angle)
            head_y = arrow_end_y + head_size * np.sin(head_angle)
            pygame.draw.line(self.screen, color, 
                           (int(arrow_end_x), int(arrow_end_y)), 
                           (int(head_x), int(head_y)), 2)
    
    def draw_agent_ring(self, distances, velocities, types, ids):
        """绘制其他 agent 环（包含类型和 ID 标注）"""
        center_x, center_y = self.center
        ring_radius = self.ring_radii['agents']
        
        for i, distance in enumerate(distances):
            if distance >= 0.99:  # 未检测到
                continue
            
            agent_type = types[i]
            if agent_type < 0:  # 无效类型
                continue
            
            # 根据类型选择颜色
            color = self.colors['predator'] if agent_type == 0 else self.colors['prey']
            
            angle = self.sensor_angles[i] - np.pi/2
            
            # 计算条的长度
            bar_length = ring_radius * (1 - distance) * 0.95
            
            if bar_length < 2:
                continue
            
            # 绘制条
            start_x = center_x + ring_radius * np.cos(angle) * 0.1
            start_y = center_y + ring_radius * np.sin(angle) * 0.1
            end_x = center_x + (ring_radius * 0.1 + bar_length) * np.cos(angle)
            end_y = center_y + (ring_radius * 0.1 + bar_length) * np.sin(angle)
            
            pygame.draw.line(self.screen, color, 
                           (int(start_x), int(start_y)), 
                           (int(end_x), int(end_y)), 4)
            
            # 绘制速度箭头
            if velocities is not None and abs(velocities[i]) > 0.1:
                self.draw_velocity_arrow(end_x, end_y, angle, velocities[i], color)
            
            # 绘制类型和 ID 标签
            agent_id = int(ids[i])
            type_symbol = 'P' if agent_type == 0 else 'R'
            label = f"{type_symbol}{agent_id}"
            
            text = self.font_small.render(label, True, color)
            text_rect = text.get_rect(center=(int(end_x), int(end_y)))
            self.screen.blit(text, text_rect)
    
    def draw_strategy_pointer(self, features, agent_type):
        """绘制策略提示指针（追击/逃逸）"""
        center_x, center_y = self.center
        
        agent_distances = features['agent_dist']
        agent_velocities = features['agent_vel']
        agent_types = features['agent_type']
        
        # 找到目标类型的 agent
        if agent_type == 'predator':
            # Predator 追击 Prey
            target_mask = (agent_types == 1.0) & (agent_distances < 1.0)
            pointer_color = self.colors['predator']
            pointer_label = "HUNT"
        else:
            # Prey 逃离 Predator
            target_mask = (agent_types == 0.0) & (agent_distances < 1.0)
            pointer_color = self.colors['prey']
            pointer_label = "ESCAPE"
        
        if not np.any(target_mask):
            return
        
        # 计算策略向量（加权平均）
        weights = np.zeros(self.n_sensors)
        for i in range(self.n_sensors):
            if target_mask[i]:
                proximity = 1 - agent_distances[i]
                # Predator: 考虑接近速度；Prey: 只考虑距离
                if agent_type == 'predator':
                    approach_bonus = max(0, -agent_velocities[i])  # 负速度=接近
                    weights[i] = proximity * (1 + approach_bonus)
                else:
                    weights[i] = proximity
        
        if np.sum(weights) < 0.01:
            return
        
        # 归一化权重
        weights /= np.sum(weights)
        
        # 计算策略方向（向量加权和）
        strategy_x = 0
        strategy_y = 0
        for i in range(self.n_sensors):
            if weights[i] > 0:
                angle = self.sensor_angles[i] - np.pi/2
                strategy_x += weights[i] * np.cos(angle)
                strategy_y += weights[i] * np.sin(angle)
        
        # Prey 的策略是反向逃离
        if agent_type == 'prey':
            strategy_x = -strategy_x
            strategy_y = -strategy_y
        
        # 绘制策略指针
        magnitude = np.sqrt(strategy_x**2 + strategy_y**2)
        if magnitude > 0.01:
            pointer_length = 60
            end_x = center_x + (strategy_x / magnitude) * pointer_length
            end_y = center_y + (strategy_y / magnitude) * pointer_length
            
            # 绘制粗箭头
            pygame.draw.line(self.screen, pointer_color, 
                           self.center, (int(end_x), int(end_y)), 5)
            
            # 箭头头部
            arrow_angle = np.arctan2(strategy_y, strategy_x)
            head_size = 15
            head_angles = [arrow_angle + 2.5, arrow_angle - 2.5]
            
            for head_angle in head_angles:
                head_x = end_x + head_size * np.cos(head_angle)
                head_y = end_y + head_size * np.sin(head_angle)
                pygame.draw.line(self.screen, pointer_color, 
                               (int(end_x), int(end_y)), 
                               (int(head_x), int(head_y)), 5)
            
            # 标签
            text = self.font_medium.render(pointer_label, True, pointer_color)
            text_rect = text.get_rect(center=(int(end_x), int(end_y) - 20))
            self.screen.blit(text, text_rect)
    
    def draw_hud(self, info, features):
        """绘制 HUD 信息面板"""
        hud_x = 20
        hud_y = 20
        line_height = 25
        
        # 标题
        agent_type = info.get('agent_type', 'Unknown')
        title_text = f"AGENT TYPE: {agent_type.upper()}"
        title = self.font_large.render(title_text, True, self.colors['text'])
        self.screen.blit(title, (hud_x, hud_y))
        hud_y += 40
        
        # 健康值条
        health = info.get('current_health', 100)
        max_health = 100  # 假设初始健康值
        health_ratio = max(0, min(1, health / max_health))
        
        bar_width = 200
        bar_height = 20
        
        # 背景
        pygame.draw.rect(self.screen, (60, 60, 60), 
                        (hud_x, hud_y, bar_width, bar_height))
        
        # 健康条
        health_width = int(bar_width * health_ratio)
        if health_ratio > 0.7:
            health_color = self.colors['health_high']
        elif health_ratio > 0.3:
            health_color = self.colors['health_mid']
        else:
            health_color = self.colors['health_low']
        
        if health_width > 0:
            pygame.draw.rect(self.screen, health_color, 
                            (hud_x, hud_y, health_width, bar_height))
        
        # 边框
        pygame.draw.rect(self.screen, self.colors['text'], 
                        (hud_x, hud_y, bar_width, bar_height), 2)
        
        # 健康值文本
        health_text = f"Health: {health:.1f} / {max_health}"
        text = self.font_small.render(health_text, True, self.colors['text'])
        self.screen.blit(text, (hud_x + bar_width + 10, hud_y))
        
        hud_y += 35
        
        # 帧数和存活状态
        frame = info.get('current_frame', 0)
        max_cycles = info.get('max_cycles', 500)
        is_alive = info.get('is_alive', True)
        
        frame_text = f"Frame: {frame} / {max_cycles}"
        text = self.font_small.render(frame_text, True, self.colors['text_dim'])
        self.screen.blit(text, (hud_x, hud_y))
        hud_y += line_height
        
        alive_color = self.colors['health_high'] if is_alive else self.colors['health_low']
        alive_text = f"Status: {'ALIVE' if is_alive else 'DEAD'}"
        text = self.font_small.render(alive_text, True, alive_color)
        self.screen.blit(text, (hud_x, hud_y))
        hud_y += line_height + 10
        
        # 性能指标
        metrics = info.get('performance_metrics', {})
        
        if agent_type == 'prey':
            foraging = metrics.get('foraging_rate', 0)
            escape = metrics.get('escape_rate', 0)
            
            text = self.font_small.render(f"Foraging Rate: {foraging:.2%}", 
                                         True, self.colors['text_dim'])
            self.screen.blit(text, (hud_x, hud_y))
            hud_y += line_height
            
            text = self.font_small.render(f"Escape Rate: {escape:.2%}", 
                                         True, self.colors['text_dim'])
            self.screen.blit(text, (hud_x, hud_y))
            hud_y += line_height
        
        elif agent_type == 'predator':
            hunting = metrics.get('hunting_rate', 0)
            
            text = self.font_small.render(f"Hunting Rate: {hunting:.2%}", 
                                         True, self.colors['text_dim'])
            self.screen.blit(text, (hud_x, hud_y))
            hud_y += line_height
        
        # 会话指标
        desr = metrics.get('DESR', 0)
        mdsd = metrics.get('MDSD', 0)
        
        text = self.font_small.render(f"DESR (Survival): {desr:.2%}", 
                                     True, self.colors['text_dim'])
        self.screen.blit(text, (hud_x, hud_y))
        hud_y += line_height
        
        text = self.font_small.render(f"MDSD (Duration): {mdsd:.1f} frames", 
                                     True, self.colors['text_dim'])
        self.screen.blit(text, (hud_x, hud_y))
        hud_y += line_height + 10
        
        # 即时事件
        food_contact = features['food_contact'] > 0
        poison_contact = features['poison_contact'] > 0
        
        if food_contact:
            text = self.font_small.render("🟢 FOOD CONTACT", True, self.colors['food'])
            self.screen.blit(text, (hud_x, hud_y))
            hud_y += line_height
        
        if poison_contact:
            text = self.font_small.render("🔴 POISON CONTACT", True, self.colors['poison'])
            self.screen.blit(text, (hud_x, hud_y))
            hud_y += line_height
        
        # 控制提示（右下角）
        help_x = self.window_size - 250
        help_y = self.window_size - 120
        
        help_texts = [
            "Controls:",
            "W/A/S/D or Arrows: Move",
            "SPACE: Stop",
            "ESC: Quit",
        ]
        
        for text_str in help_texts:
            text = self.font_small.render(text_str, True, self.colors['text_dim'])
            self.screen.blit(text, (help_x, help_y))
            help_y += 20
    
    def close(self):
        """关闭渲染器"""
        pygame.quit()