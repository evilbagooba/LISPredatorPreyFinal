#watrerworld_models.py
import numpy as np
import pygame
import pymunk
from gymnasium import spaces

import pygame
import math
import pymunk

class SafeIconRenderer:
    """碰撞安全的图标渲染器
    
    关键原则：
    1. 物理碰撞体保持简单的圆形
    2. 视觉渲染可以是复杂的图标
    3. 两者完全分离，互不影响
    """
    
    @staticmethod
    def draw_wolf_predator(screen, pos, radius, algorithm="default", health=100, max_health=100):
        """绘制狼形捕食者（视觉效果）"""
        x, y = pos
        
        # 🔥 重要：物理碰撞体是圆形，但视觉上绘制狼形
        
        # 根据算法确定颜色主题
        color_themes = {
            "default": (120, 80, 40),      # 棕色
            "aggressive": (140, 60, 30),   # 深棕红
            "smart": (100, 70, 50),        # 灰棕色
            "cooperative": (130, 100, 60), # 浅棕色
        }
        base_color = color_themes.get(algorithm, (120, 80, 40))
        
        # 健康状态影响颜色（受伤时变暗）
        health_ratio = health / max_health
        color = tuple(int(c * (0.5 + 0.5 * health_ratio)) for c in base_color)
        
        # === 绘制狼的视觉效果 ===
        
        # 1. 主体（圆形，与物理碰撞体对应）
        pygame.draw.circle(screen, color, (int(x), int(y)), int(radius))
        
        # 2. 耳朵（在圆形基础上添加）
        ear_size = radius * 0.4
        ear1_pos = (x - radius*0.4, y - radius*0.7)
        ear2_pos = (x + radius*0.4, y - radius*0.7)
        
        # 外耳
        ear1_points = [
            (ear1_pos[0], ear1_pos[1]),
            (ear1_pos[0] - ear_size*0.6, ear1_pos[1] - ear_size),
            (ear1_pos[0] + ear_size*0.2, ear1_pos[1] - ear_size*0.5)
        ]
        ear2_points = [
            (ear2_pos[0], ear2_pos[1]),
            (ear2_pos[0] + ear_size*0.6, ear2_pos[1] - ear_size),
            (ear2_pos[0] - ear_size*0.2, ear2_pos[1] - ear_size*0.5)
        ]
        pygame.draw.polygon(screen, color, ear1_points)
        pygame.draw.polygon(screen, color, ear2_points)
        
        # 内耳（粉色）
        inner_ear1 = [
            (ear1_pos[0] - ear_size*0.1, ear1_pos[1] - ear_size*0.2),
            (ear1_pos[0] - ear_size*0.4, ear1_pos[1] - ear_size*0.7),
            (ear1_pos[0] + ear_size*0.1, ear1_pos[1] - ear_size*0.4)
        ]
        inner_ear2 = [
            (ear2_pos[0] + ear_size*0.1, ear2_pos[1] - ear_size*0.2),
            (ear2_pos[0] + ear_size*0.4, ear2_pos[1] - ear_size*0.7),
            (ear2_pos[0] - ear_size*0.1, ear2_pos[1] - ear_size*0.4)
        ]
        pygame.draw.polygon(screen, (200, 150, 150), inner_ear1)
        pygame.draw.polygon(screen, (200, 150, 150), inner_ear2)
        
        # 3. 鼻子/嘴部
        snout_center = (x, y + radius*0.4)
        snout_width = radius * 0.6
        snout_height = radius * 0.4
        snout_rect = pygame.Rect(
            snout_center[0] - snout_width//2, 
            snout_center[1] - snout_height//2,
            snout_width, 
            snout_height
        )
        darker_color = tuple(max(0, c - 30) for c in color)
        pygame.draw.ellipse(screen, darker_color, snout_rect)
        
        # 4. 眼睛 - 根据健康状态发光
        eye_size = radius // 5
        eye1_pos = (int(x - radius*0.3), int(y - radius*0.2))
        eye2_pos = (int(x + radius*0.3), int(y - radius*0.2))
        
        # 眼睛发光效果（健康时更亮）
        glow_intensity = int(100 + 155 * health_ratio)
        eye_color = (glow_intensity, 50, 50)
        
        pygame.draw.circle(screen, (255, 200, 200), eye1_pos, eye_size)
        pygame.draw.circle(screen, eye_color, eye1_pos, eye_size//2)
        pygame.draw.circle(screen, (255, 200, 200), eye2_pos, eye_size)
        pygame.draw.circle(screen, eye_color, eye2_pos, eye_size//2)
        
        # 5. 鼻子
        nose_pos = (int(x), int(y + radius*0.5))
        pygame.draw.circle(screen, (50, 50, 50), nose_pos, max(2, radius//8))
        
        # 6. 边缘描边（增强视觉效果）
        pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), int(radius), 2)

    @staticmethod
    def draw_rabbit_prey(screen, pos, radius, algorithm="default", health=100, max_health=100):
        """绘制兔子猎物（视觉效果）"""
        x, y = pos
        
        # 根据算法确定颜色主题
        color_themes = {
            "default": (200, 180, 150),    # 浅棕色
            "evasive": (220, 200, 170),    # 更浅的棕色
            "smart": (180, 160, 140),      # 灰棕色
            "cooperative": (240, 220, 190), # 奶白色
        }
        base_color = color_themes.get(algorithm, (200, 180, 150))
        
        # 健康状态影响颜色
        health_ratio = health / max_health
        color = tuple(int(c * (0.6 + 0.4 * health_ratio)) for c in base_color)
        
        # === 绘制兔子的视觉效果 ===
        
        # 1. 主体（圆形，与物理碰撞体对应）
        pygame.draw.circle(screen, color, (int(x), int(y)), int(radius))
        
        # 2. 长耳朵
        ear_length = radius * 1.2
        ear_width = radius * 0.3
        
        # 左耳
        ear1_points = [
            (x - radius*0.3, y - radius*0.5),
            (x - radius*0.6, y - radius*0.5 - ear_length),
            (x - radius*0.1, y - radius*0.5 - ear_length*0.8),
            (x - radius*0.1, y - radius*0.7)
        ]
        # 右耳
        ear2_points = [
            (x + radius*0.3, y - radius*0.5),
            (x + radius*0.6, y - radius*0.5 - ear_length),
            (x + radius*0.1, y - radius*0.5 - ear_length*0.8),
            (x + radius*0.1, y - radius*0.7)
        ]
        
        pygame.draw.polygon(screen, color, ear1_points)
        pygame.draw.polygon(screen, color, ear2_points)
        
        # 耳朵内部（粉色）
        inner_ear1 = [
            (x - radius*0.25, y - radius*0.6),
            (x - radius*0.45, y - radius*0.6 - ear_length*0.7),
            (x - radius*0.15, y - radius*0.6 - ear_length*0.6)
        ]
        inner_ear2 = [
            (x + radius*0.25, y - radius*0.6),
            (x + radius*0.45, y - radius*0.6 - ear_length*0.7),
            (x + radius*0.15, y - radius*0.6 - ear_length*0.6)
        ]
        
        pink_color = (255, 200, 200) if health_ratio > 0.5 else (200, 150, 150)
        pygame.draw.polygon(screen, pink_color, inner_ear1)
        pygame.draw.polygon(screen, pink_color, inner_ear2)
        
        # 3. 大眼睛（无辜的表情）
        eye_size = radius // 3
        eye1_pos = (int(x - radius*0.25), int(y - radius*0.1))
        eye2_pos = (int(x + radius*0.25), int(y - radius*0.1))
        
        # 眼白
        pygame.draw.circle(screen, (255, 255, 255), eye1_pos, eye_size)
        pygame.draw.circle(screen, (255, 255, 255), eye2_pos, eye_size)
        
        # 瞳孔（健康时更黑，受伤时发红）
        pupil_color = (0, 0, 0) if health_ratio > 0.3 else (100, 0, 0)
        pygame.draw.circle(screen, pupil_color, eye1_pos, eye_size//2)
        pygame.draw.circle(screen, pupil_color, eye2_pos, eye_size//2)
        
        # 高光
        highlight_pos1 = (eye1_pos[0] - eye_size//4, eye1_pos[1] - eye_size//4)
        highlight_pos2 = (eye2_pos[0] - eye_size//4, eye2_pos[1] - eye_size//4)
        pygame.draw.circle(screen, (255, 255, 255), highlight_pos1, max(1, eye_size//6))
        pygame.draw.circle(screen, (255, 255, 255), highlight_pos2, max(1, eye_size//6))
        
        # 4. 小鼻子
        nose_pos = (int(x), int(y + radius*0.2))
        nose_color = (255, 150, 150) if health_ratio > 0.5 else (200, 100, 100)
        pygame.draw.circle(screen, nose_color, nose_pos, max(2, radius//8))
        
        # 5. 嘴巴（小小的）
        mouth_start = (int(x - radius*0.1), int(y + radius*0.35))
        mouth_end = (int(x + radius*0.1), int(y + radius*0.35))
        pygame.draw.line(screen, (100, 100, 100), mouth_start, mouth_end, 1)
        
        # 6. 边缘描边
        pygame.draw.circle(screen, (0, 0, 0), (int(x), int(y)), int(radius), 1)

    @staticmethod
    def draw_food_particle(screen, pos, radius):
        """绘制食物粒子 - 胡萝卜形状"""
        x, y = pos
        
        # 胡萝卜主体（橙色三角形）
        carrot_points = [
            (x, y - radius),           # 顶部
            (x - radius*0.7, y + radius*0.5),  # 左下
            (x + radius*0.7, y + radius*0.5)   # 右下
        ]
        pygame.draw.polygon(screen, (255, 140, 0), carrot_points)
        
        # 胡萝卜叶子（绿色）
        leaf_points = [
            (x - radius*0.2, y - radius),
            (x - radius*0.4, y - radius*1.3),
            (x, y - radius*1.1),
            (x + radius*0.4, y - radius*1.3),
            (x + radius*0.2, y - radius)
        ]
        pygame.draw.polygon(screen, (0, 200, 0), leaf_points)
        
        # 胡萝卜纹理线
        for i in range(2):
            line_y = y - radius*0.3 + i * radius*0.4
            pygame.draw.line(screen, (200, 100, 0), 
                           (x - radius*0.4, line_y), 
                           (x + radius*0.4, line_y), 1)
        
        # 边框
        pygame.draw.polygon(screen, (0, 0, 0), carrot_points, 1)
    
    @staticmethod
    def draw_poison_particle(screen, pos, radius):
        """绘制毒物粒子 - 毒蘑菇形状"""
        x, y = pos
        
        # 蘑菇帽（红色带白点）
        cap_center = (x, y - radius*0.3)
        pygame.draw.circle(screen, (200, 0, 0), cap_center, int(radius*0.8))
        
        # 白色斑点
        spots = [
            (x - radius*0.4, y - radius*0.5),
            (x + radius*0.2, y - radius*0.6),
            (x - radius*0.1, y - radius*0.1),
            (x + radius*0.4, y - radius*0.2)
        ]
        for spot in spots:
            pygame.draw.circle(screen, (255, 255, 255), 
                             (int(spot[0]), int(spot[1])), max(2, radius//6))
        
        # 蘑菇杆（米色）
        stem_rect = pygame.Rect(x - radius*0.2, y, radius*0.4, radius*0.8)
        pygame.draw.ellipse(screen, (230, 220, 180), stem_rect)
        
        # 边框
        pygame.draw.circle(screen, (0, 0, 0), cap_center, int(radius*0.8), 1)
        pygame.draw.ellipse(screen, (0, 0, 0), stem_rect, 1)
    
    @staticmethod
    def draw_status_indicators(screen, pos, radius, health, max_health, algorithm, agent_id):
        """绘制状态指示器"""
        x, y = pos
        
        # 健康值条
        bar_width = radius * 2
        bar_height = max(3, radius // 6)
        bar_x = x - bar_width // 2
        bar_y = y - radius - bar_height - 5
        
        # 背景
        pygame.draw.rect(screen, (60, 60, 60), (bar_x, bar_y, bar_width, bar_height))
        
        # 健康值
        health_ratio = max(0, min(1, health / max_health))
        health_width = int(bar_width * health_ratio)
        
        if health_ratio > 0.7:
            health_color = (0, 255, 0)
        elif health_ratio > 0.3:
            health_color = (255, 255, 0)
        else:
            health_color = (255, 0, 0)
        
        if health_width > 0:
            pygame.draw.rect(screen, health_color, (bar_x, bar_y, health_width, bar_height))
        
        pygame.draw.rect(screen, (255, 255, 255), (bar_x, bar_y, bar_width, bar_height), 1)
        
        # 算法徽章
        badge_pos = (int(x + radius*0.7), int(y - radius*0.7))
        badge_radius = max(4, radius//4)
        
        algo_colors = {
            "default": (128, 128, 128),
            "aggressive": (255, 0, 0),
            "smart": (0, 0, 255),
            "cooperative": (0, 255, 0),
            "evasive": (255, 255, 0),
        }
        badge_color = algo_colors.get(algorithm, (128, 128, 128))
        
        pygame.draw.circle(screen, badge_color, badge_pos, badge_radius)
        pygame.draw.circle(screen, (255, 255, 255), badge_pos, badge_radius, 1)
        
        # ID数字
        if radius > 10:  # 只有足够大时才显示数字
            font = pygame.font.Font(None, max(10, radius//3))
            text = font.render(str(agent_id), True, (255, 255, 255))
            text_rect = text.get_rect(center=badge_pos)
            screen.blit(text, text_rect)


class Obstacle:
    def __init__(self, x, y, pixel_scale=750, radius=0.1):
        self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
        self.body.position = x, y
        self.body.velocity = 0.0, 0.0

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.density = 1
        self.shape.elasticity = 1
        self.shape.custom_value = 1

        self.radius = radius 
        self.color = (120, 176, 178)

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )


class MovingObject:
    def __init__(self, x, y, pixel_scale=750, radius=0.015):
        self.pixel_scale = pixel_scale
        self.body = pymunk.Body()
        self.body.position = x, y

        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.custom_value = 1

        self.shape.reset_position = self.reset_position
        self.shape.reset_velocity = self.reset_velocity
        # 新增：Agent间碰撞的指示器
        self.shape.predator_catch_indicator = 0    # predator捕获prey时设为1
        self.shape.prey_caught_indicator = 0       # prey被捕获时设为1  
        self.shape.same_algo_meet_indicator = 0    # 相同算法相遇时设为1
        self.shape.diff_algo_meet_indicator = 0    # 不同算法相遇时设为1
        self.shape.health = 0  # 新增：健康值属性

        self.radius = radius 

    def add(self, space):
        space.add(self.body, self.shape)

    def draw(self, display, convert_coordinates):
        pygame.draw.circle(
            display, self.color, convert_coordinates(self.body.position), self.radius
        )

    def reset_position(self, x, y):
        self.body.position = x, y

    def reset_velocity(self, vx, vy):
        self.body.velocity = vx, vy


class Evaders(MovingObject):
    def __init__(self, x, y, vx, vy, radius=0.03, collision_type=2, max_speed=100, is_static=False, pixel_scale=750):
        # 先保存参数（在调用 super().__init__ 之前）
        self._radius_world_unit = radius
        self._collision_type = collision_type
        self._max_speed = max_speed
        self._is_static = is_static
        
        if is_static:
            super().__init__(x, y, radius=radius)
            self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
            self.body.position = x, y
            self.body.velocity = 0.0, 0.0
            self.shape = pymunk.Circle(self.body, radius)
            self.shape.elasticity = 1
            self.shape.density = 1
            self.shape.custom_value = 1
            self.shape.reset_position = self.reset_position
            self.shape.reset_velocity = self.reset_velocity
        else:
            super().__init__(x, y, radius=radius)
            self.body.velocity = vx, vy

        self.is_static = is_static
        self.color = (145, 250, 116)
        self.shape.collision_type = collision_type
        self.shape.counter = 0
        self.shape.max_speed = max_speed
        self.shape.density = 0.01
        
        # 新增：保存对父对象的引用，用于查找索引
        self.shape.parent_evader = self
    
    def rebuild_in_place(self, space, x, y, vx, vy):
        """在原地重建：移除旧的 body/shape，创建新的"""
        # ✅ 1. 安全检查：确保 body 确实在 space 中
        if self.body not in space.bodies:
            # Body 已经被移除（可能是环境 reset 或其他原因）
            # 直接创建新的即可
            pass  # 跳过移除步骤
        else:
            # 1. 从 space 移除
            space.remove(self.body, self.shape)
        
        # 2. 创建新的 body
        if self._is_static:
            self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
            self.body.position = x, y
            self.body.velocity = 0.0, 0.0
        else:
            self.body = pymunk.Body()
            self.body.position = x, y
            self.body.velocity = vx, vy
        
        # 3. 创建新的 shape（保持相同的 collision_type）
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 1
        self.shape.density = 0.01 if not self._is_static else 1
        self.shape.custom_value = 1
        self.shape.collision_type = self._collision_type
        self.shape.counter = 0
        self.shape.max_speed = self._max_speed
        
        # 4. 重新绑定方法和引用
        self.shape.reset_position = self.reset_position
        self.shape.reset_velocity = self.reset_velocity
        self.shape.parent_evader = self
        
        # 5. 添加回 space
        space.add(self.body, self.shape)
    
    def get_init_params(self):
        """返回初始化参数"""
        return {
            'radius': self._radius_world_unit,
            'collision_type': self._collision_type,
            'max_speed': self._max_speed,
            'is_static': self._is_static
        }

class Poisons(MovingObject):
    def __init__(self, x, y, vx, vy, radius=0.015 * 3 / 4, collision_type=3, max_speed=100, is_static=False, pixel_scale=750):
        # 先保存参数
        self._radius_world_unit = radius
        self._collision_type = collision_type
        self._max_speed = max_speed
        self._is_static = is_static
        
        if is_static:
            super().__init__(x, y, pixel_scale=pixel_scale, radius=radius)
            self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
            self.body.position = x, y
            self.body.velocity = 0.0, 0.0
            self.shape = pymunk.Circle(self.body, radius)
            self.shape.elasticity = 1
            self.shape.density = 1
            self.shape.custom_value = 1
            self.shape.reset_position = self.reset_position
            self.shape.reset_velocity = self.reset_velocity
        else:
            super().__init__(x, y, radius=radius)
            self.body.velocity = vx, vy

        self.is_static = is_static
        self.color = (238, 116, 106)
        self.shape.collision_type = collision_type
        self.shape.max_speed = max_speed
        
        # 新增：保存对父对象的引用
        self.shape.parent_poison = self
    
    def rebuild_in_place(self, space, x, y, vx, vy):
        """在原地重建：移除旧的 body/shape，创建新的"""
        if self.body not in space.bodies:
            # Body 已经被移除（可能是环境 reset 或其他原因）
            # 直接创建新的即可
            pass  # 跳过移除步骤
        else:
            # 1. 从 space 移除
            space.remove(self.body, self.shape)
        
        # 2. 创建新的 body
        if self._is_static:
            self.body = pymunk.Body(0, 0, pymunk.Body.STATIC)
            self.body.position = x, y
            self.body.velocity = 0.0, 0.0
        else:
            self.body = pymunk.Body()
            self.body.position = x, y
            self.body.velocity = vx, vy
        
        # 3. 创建新的 shape
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 1
        self.shape.density = 1
        self.shape.custom_value = 1
        self.shape.collision_type = self._collision_type
        self.shape.max_speed = self._max_speed
        
        # 4. 重新绑定方法和引用
        self.shape.reset_position = self.reset_position
        self.shape.reset_velocity = self.reset_velocity
        self.shape.parent_poison = self
        
        # 5. 添加回 space
        space.add(self.body, self.shape)
    
    def get_init_params(self):
        """返回初始化参数"""
        return {
            'radius': self._radius_world_unit,
            'collision_type': self._collision_type,
            'max_speed': self._max_speed,
            'is_static': self._is_static
        }
class BaseAgent(MovingObject):
    def __init__(
        self,
        x,
        y,
        max_accel,
        agent_speed,
        radius=0.015,
        n_sensors=30,
        sensor_range=0.2,
        collision_type=1,
        speed_features=True,
        color=None,
        algorithm="default",  # ← 添加这个参数
        pixel_scale=750,  # ← 新增参数
    ):
        super().__init__(x, y, radius=radius)
        # 新增
        self.algorithm = algorithm
        self.color = color if color is not None else (101, 104, 249)
        self.shape.collision_type = collision_type
        self.sensor_color = (0, 0, 0)
        self.n_sensors = n_sensors
        self.sensor_range = sensor_range 
        self.max_accel = max_accel
        self.max_speed = agent_speed
        self.body.velocity = 0.0, 0.0
        

        self.shape.food_indicator = 0  # 1 if food caught at this step, 0 otherwise
        self.shape.food_touched_indicator = (
            0  # 1 if food touched as this step, 0 otherwise
        )
        self.shape.poison_indicator = 0  # 1 if poisoned this step, 0 otherwise

        # Generate self.n_sensors angles, evenly spaced from 0 to 2pi
        # We generate 1 extra angle and remove it because linspace[0] = 0 = 2pi = linspace[-1]
        angles = np.linspace(0.0, 2.0 * np.pi, self.n_sensors + 1)[:-1]

        # Convert angles to x-y coordinates
        sensor_vectors = np.c_[np.cos(angles), np.sin(angles)]
        self._sensors = sensor_vectors
        self.shape.custom_value = 1

        # Number of observation coordinates from each sensor
        self._sensor_obscoord = 5
        if speed_features:
            self._sensor_obscoord += 3

        self.sensor_obs_coord = self.n_sensors * self._sensor_obscoord
        self.obs_dim = (
            self.sensor_obs_coord + 2 + 2 * self.n_sensors
        )  # +1 for is_colliding_evader, +1 for is_colliding_poison

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.float32(-2 * np.sqrt(2)),
            high=np.float32(2 * np.sqrt(2)),
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return spaces.Box(
            low=np.float32(-self.max_accel),
            high=np.float32(self.max_accel),
            shape=(2,),
            dtype=np.float32,
        )

    @property
    def position(self):
        assert self.body.position is not None
        return np.array([self.body.position[0], self.body.position[1]])

    @property
    def velocity(self):
        assert self.body.velocity is not None
        return np.array([self.body.velocity[0], self.body.velocity[1]])

    @property
    def sensors(self):
        assert self._sensors is not None
        return self._sensors

    def draw(self, display, convert_coordinates):
        self.center = convert_coordinates(self.body.position)
        for sensor in self._sensors:
            start = self.center
            end = self.center + self.sensor_range * sensor
            pygame.draw.line(display, self.sensor_color, start, end, 1)

        pygame.draw.circle(display, self.color, self.center, self.radius)

    def get_sensor_barrier_readings(self):
        """Get the distance to the barrier.

        See https://github.com/BolunDai0216/WaterworldRevamp for
        a detailed explanation.
        """
        # Get the endpoint position of each sensor
        sensor_vectors = self._sensors * self.sensor_range
        position_vec = np.array([self.body.position.x, self.body.position.y])
        sensor_endpoints = position_vec + sensor_vectors

        # Clip sensor lines on the environment's barriers.
        # Note that any clipped vectors may not be at the same angle as the original sensors
        clipped_endpoints = np.clip(sensor_endpoints, 0.0, self.pixel_scale)

        # Extract just the sensor vectors after clipping
        clipped_vectors = clipped_endpoints - position_vec

        # Find the ratio of the clipped sensor vector to the original sensor vector
        # Scaling the vector by this ratio will limit the end of the vector to the barriers
        ratios = np.divide(
            clipped_vectors,
            sensor_vectors,
            out=np.ones_like(clipped_vectors),
            where=np.abs(sensor_vectors) > 1e-8,
        )

        # Find the minimum ratio (x or y) of clipped endpoints to original endpoints
        minimum_ratios = np.amin(ratios, axis=1)

        # Convert to 2d array of size (n_sensors, 1)
        sensor_values = np.expand_dims(minimum_ratios, 0)

        # Set values beyond sensor range to 1.0
        does_sense = minimum_ratios < (1.0 - 1e-4)
        does_sense = np.expand_dims(does_sense, 0)
        sensor_values[np.logical_not(does_sense)] = 1.0

        # Convert -0 to 0
        sensor_values[sensor_values == -0] = 0
        # ✅ 添加这行：裁剪到合理范围
        sensor_values = np.clip(sensor_values[0, :], 0.0, 1.0)
        return sensor_values

    def get_sensor_reading(
        self, object_coord, object_radius, object_velocity, object_max_velocity,object_type=None, object_id=None  # 新增可选参数
    ):
        """Get distance and velocity to another object (Obstacle, Agent, Evader, Poison)."""
        # Get location and velocity of agent
        self.center = self.body.position
        _velocity = self.body.velocity

        # Get distance of object in local frame as a 2x1 numpy array
        distance_vec = np.array(
            [[object_coord[0] - self.center[0]], [object_coord[1] - self.center[1]]]
        )
        distance_squared = np.sum(distance_vec**2)

        # Get relative velocity as a 2x1 numpy array
        relative_speed = np.array(
            [
                [object_velocity[0] - _velocity[0]],
                [object_velocity[1] - _velocity[1]],
            ]
        )

        # Project distance to sensor vectors
        sensor_distances = self._sensors @ distance_vec

        # Project velocity vector to sensor vectors
        sensor_velocities = (
            self._sensors @ relative_speed / (object_max_velocity + self.max_speed)
        )

        # if np.any(sensor_velocities < -2 * np.sqrt(2)) or np.any(
        #     sensor_velocities > 2 * np.sqrt(2)
        # ):
        #     set_trace()

        # Check for valid detection criterions
        wrong_direction_idx = sensor_distances < 0
        out_of_range_idx = sensor_distances - object_radius > self.sensor_range
        no_intersection_idx = (
            distance_squared - sensor_distances**2 > object_radius**2
        )
        not_sensed_idx = wrong_direction_idx | out_of_range_idx | no_intersection_idx

        # Set not sensed sensor readings of position to sensor range
        sensor_distances = np.clip(sensor_distances / self.sensor_range, 0, 1)
        sensor_distances[not_sensed_idx] = 1.0

        # Set not sensed sensor readings of velocity to zero
        sensor_velocities[not_sensed_idx] = 0.0
        # 新增：处理类型和ID信息
        if object_type is not None and object_id is not None:
            # 初始化类型和ID数组，默认值为-1（表示未检测到）
            sensor_types = np.full(self.n_sensors, -1.0, dtype=np.float32)
            sensor_ids = np.full(self.n_sensors, -1.0, dtype=np.float32)
            
            # 只有检测到对象的传感器才设置正确的类型和ID
            detected_idx = (~not_sensed_idx).flatten()
            sensor_types[detected_idx] = float(object_type)
            sensor_ids[detected_idx] = float(object_id)

            return sensor_distances, sensor_velocities, sensor_types, sensor_ids
        else:
            # 向后兼容：如果没有提供type和id，只返回distance和velocity
            return sensor_distances, sensor_velocities

        # return sensor_distances, sensor_velocities
    





class Predator(BaseAgent):
    def __init__(self, x, y, max_accel, agent_speed, algorithm="default", **kwargs):
        # 移除 color 参数处理，使用默认红色
        super().__init__(x, y, max_accel, agent_speed, 
                        color=(220, 20, 20),  # 红色
                        algorithm=algorithm,  # ← 传递
                        **kwargs)
        self.agent_type = 'predator'

class Prey(BaseAgent):
    def __init__(self, x, y, max_accel, agent_speed,  algorithm="default",**kwargs):
        # 移除 color 参数处理，使用默认蓝色
        super().__init__(x, y, max_accel, agent_speed,
                        color=(20, 120, 220),  # 蓝色
                        algorithm=algorithm,  # ← 传递
                        **kwargs)
        self.agent_type = 'prey'