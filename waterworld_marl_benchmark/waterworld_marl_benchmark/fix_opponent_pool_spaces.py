"""
修复 OpponentPool.get_opponent_policy 中的空间参数传递问题
"""

from pathlib import Path

print("="*70)
print("修复对手池空间参数传递")
print("="*70)

file_path = Path('src/core/opponent_pool.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 找到需要修复的代码
old_code = """        # 创建算法实例
        # 注意：这里需要observation_space和action_space，但我们还没有
        # 暂时先返回None，在实际使用时再加载
        algo = create_algorithm(
            algo_name=opponent_info['algo'],
            observation_space=None,  # 需要从环境获取
            action_space=None,
            config=opponent_info['config'],
            device=device
        )"""

new_code = """        # ✅ 创建算法实例（使用传入的空间信息）
        algo = create_algorithm(
            algo_name=opponent_info['algo'],
            observation_space=observation_space,  # 使用传入的参数
            action_space=action_space,             # 使用传入的参数
            config=opponent_info['config'],
            device=device
        )"""

if old_code in content:
    content = content.replace(old_code, new_code)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已修复 get_opponent_policy 方法")
    print("   - observation_space: None → observation_space")
    print("   - action_space: None → action_space")
else:
    # 尝试更灵活的匹配
    if "observation_space=None,  # 需要从环境获取" in content:
        content = content.replace(
            "observation_space=None,  # 需要从环境获取",
            "observation_space=observation_space,  # 使用传入的参数"
        )
        content = content.replace(
            "action_space=None,",
            "action_space=action_space,"
        )
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 已修复（灵活匹配）")
    else:
        print("⚠️  未找到匹配的代码，请手动修复")
        print("\n请在 src/core/opponent_pool.py 的 get_opponent_policy 方法中：")
        print("将:")
        print("    observation_space=None,")
        print("    action_space=None,")
        print("\n改为:")
        print("    observation_space=observation_space,")
        print("    action_space=action_space,")

print("\n" + "="*70)
print("修复完成！")
print("="*70)