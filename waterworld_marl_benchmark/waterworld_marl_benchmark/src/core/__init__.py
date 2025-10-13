"""
核心模块初始化
"""

from .environment import WaterworldEnvManager, SingleAgentWrapper, create_training_env
from .opponent_pool import OpponentPool, MixedOpponentSampler, create_opponent_policies
from .trainer import MultiAgentTrainer
from .agent_manager import AgentManager


__all__ = [
    'WaterworldEnvManager',
    'SingleAgentWrapper',
    'create_training_env',
    'OpponentPool',
    'MixedOpponentSampler',
    'create_opponent_policies',
    'MultiAgentTrainer',
    'AgentManager'
]