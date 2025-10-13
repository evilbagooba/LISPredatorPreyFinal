"""
修复对手池加载时缺少空间信息的问题
"""

from pathlib import Path

print("="*70)
print("修复对手池加载问题")
print("="*70)

# ============================================================================
# 修复 OpponentPool.get_opponent_policy 方法
# ============================================================================
print("\n修改 src/core/opponent_pool.py...")

file_path = Path('src/core/opponent_pool.py')

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. 修改 get_opponent_policy 方法签名
old_signature = """    def get_opponent_policy(self, opponent_info: Dict[str, Any], device: str = "auto"):
        """

new_signature = """    def get_opponent_policy(
        self, 
        opponent_info: Dict[str, Any], 
        device: str = "auto",
        observation_space: gym.Space = None,
        action_space: gym.Space = None
    ):
        """

content = content.replace(old_signature, new_signature)

# 2. 修改 get_opponent_policy 方法体中的 create_algorithm 调用
old_create_call = """    # 创建算法实例（暂不加载模型，延迟加载）
    algo = create_algorithm(
        algo_name=opponent_info['algo'],
        observation_space=None,  # 需要从环境获取
        action_space=None,
        config=opponent_info['config'],
        device=device
    )"""

new_create_call = """    # ✅ 创建算法实例（使用传入的空间信息）
    algo = create_algorithm(
        algo_name=opponent_info['algo'],
        observation_space=observation_space,
        action_space=action_space,
        config=opponent_info['config'],
        device=device
    )"""

content = content.replace(old_create_call, new_create_call)

# 3. 修改 create_opponent_policies 函数中调用 get_opponent_policy 的地方
old_get_policy_call = """                    # 固定池对手（需要加载）
                    loaded_policy = pool.get_opponent_policy(opp, device)
                    policies[agent] = loaded_policy"""

new_get_policy_call = """                    # ✅ 固定池对手（传入空间信息）
                    loaded_policy = pool.get_opponent_policy(
                        opp, 
                        device,
                        obs_space,
                        action_space
                    )
                    policies[agent] = loaded_policy"""

content = content.replace(old_get_policy_call, new_get_policy_call)

# 保存修改
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print(f"✓ 已修改: {file_path}")

# ============================================================================
# 验证修改
# ============================================================================
print("\n验证修改...")

with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

checks = [
    ("observation_space: gym.Space = None" in content, "get_opponent_policy签名包含observation_space"),
    ("action_space: gym.Space = None" in content, "get_opponent_policy签名包含action_space"),
    ("obs_space," in content and "action_space" in content, "调用时传入空间参数")
]

all_passed = True
for check, description in checks:
    if check:
        print(f"  ✓ {description}")
    else:
        print(f"  ✗ {description}")
        all_passed = False

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
if all_passed:
    print("✅ 修复完成！")
else:
    print("⚠️  部分检查未通过，可能需要手动调整")
print("="*70)

print("\n修改内容：")
print("  1. get_opponent_policy 方法增加 observation_space 和 action_space 参数")
print("  2. create_algorithm 调用时使用传入的空间参数")
print("  3. create_opponent_policies 调用时传入空间信息")

print("\n现在可以重新测试：")
print("  python scripts/training/train_stage1_2.py --mode test")

print("="*70)