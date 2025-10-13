"""
完整修复测试环境配置（全自动）
1. 创建快速环境配置
2. 修改所有3个训练脚本（stage1_1, stage1_2, stage1_3）
3. 无需手动操作
"""

from pathlib import Path
import yaml
import re

print("="*70)
print("完整修复测试环境配置（全自动）")
print("="*70)

# ============================================================================
# 1. 创建合理的快速环境配置
# ============================================================================
print("\n【步骤1】创建 waterworld_fast.yaml...")

fast_env_config = {
    'environment': {
        'name': 'waterworld_v4',
        
        # 保持智能体数量（保证一致性）
        'n_predators': 5,
        'n_preys': 10,
        'n_evaders': 90,
        'n_poisons': 10,
        'n_obstacles': 2,
        
        'obstacle_coord': [[0.2, 0.2], [0.8, 0.2]],
        
        # 标准速度
        'predator_speed': 0.06,
        'prey_speed': 0.001,
        'evader_speed': 0.01,
        'poison_speed': 0.01,
        
        'sensor_range': 0.8,
        'thrust_penalty': 0.0,
        'local_ratio': 0.5,
        
        'max_cycles': 500,  # 从3000降到500
        'static_food': True,
        'static_poison': True,
        
        'render_mode': None
    },
    'observation_space': {
        'type': 'Box',
        'shape': [212],
        'dtype': 'float32'
    },
    'action_space': {
        'type': 'Box',
        'shape': [2],
        'low': -1.0,
        'high': 1.0,
        'dtype': 'float32'
    }
}

fast_env_path = Path('configs/environments/waterworld_fast.yaml')
fast_env_path.parent.mkdir(parents=True, exist_ok=True)

with open(fast_env_path, 'w', encoding='utf-8') as f:
    yaml.dump(fast_env_config, f, default_flow_style=False, allow_unicode=True)

print(f"   ✓ 已创建: {fast_env_path}")
print(f"   ✓ max_cycles: 500")

# ============================================================================
# 2. 修改 train_stage1_1.py
# ============================================================================
print("\n【步骤2】修改 train_stage1_1.py...")

script_path = Path('scripts/training/train_stage1_1.py')

if not script_path.exists():
    print(f"   ✗ 文件不存在: {script_path}")
else:
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 方法1: 查找并替换训练器创建部分
    # 使用正则表达式找到 trainer = MultiAgentTrainer( 后面的内容
    pattern = r'(def train_one_prey_algo.*?# 创建训练器\n)(    trainer = MultiAgentTrainer\(\n.*?device=args\.device\n    \))'
    
    replacement = r'''\1    # ✅ 根据模式选择环境配置
    if args.mode == 'test':
        env_config_name = 'waterworld_fast'
        print(f"🏃 测试模式，使用快速环境: max_cycles=500")
    else:
        env_config_name = 'waterworld_standard'
    
    # 创建训练器
    trainer = MultiAgentTrainer(
        train_side='prey',
        train_algo=algo,
        opponent_config=opponent_config,
        experiment_name=f"{algo}_prey_warmup",
        stage_name=stage_config['stage']['name'],
        generation=stage_config['stage']['generation'],
        version='v1',
        run_mode=args.mode,
        env_config=get_env_config(env_config_name),  # ✅ 使用动态环境
        total_timesteps=timesteps,
        device=args.device
    )'''
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # 如果正则替换失败，使用简单的字符串替换
    if new_content == content:
        old_code = """    # 创建训练器
    trainer = MultiAgentTrainer(
        train_side='prey',
        train_algo=algo,
        opponent_config=opponent_config,
        experiment_name=f"{algo}_prey_warmup",
        stage_name=stage_config['stage']['name'],
        generation=stage_config['stage']['generation'],
        version='v1',
        run_mode=args.mode,
        total_timesteps=timesteps,
        device=args.device
    )"""
        
        new_code = """    # ✅ 根据模式选择环境配置
    if args.mode == 'test':
        env_config_name = 'waterworld_fast'
        print(f"🏃 测试模式，使用快速环境: max_cycles=500")
    else:
        env_config_name = 'waterworld_standard'
    
    # 创建训练器
    trainer = MultiAgentTrainer(
        train_side='prey',
        train_algo=algo,
        opponent_config=opponent_config,
        experiment_name=f"{algo}_prey_warmup",
        stage_name=stage_config['stage']['name'],
        generation=stage_config['stage']['generation'],
        version='v1',
        run_mode=args.mode,
        env_config=get_env_config(env_config_name),  # ✅ 使用动态环境
        total_timesteps=timesteps,
        device=args.device
    )"""
        
        if old_code in content:
            new_content = content.replace(old_code, new_code)
            print(f"   ✓ 使用字符串替换修改")
        else:
            print(f"   ⚠️  未找到匹配代码，尝试更灵活的匹配...")
            # 更灵活的匹配
            old_code_flexible = "trainer = MultiAgentTrainer("
            if old_code_flexible in content:
                # 在trainer创建前插入环境选择代码
                insertion_code = """    # ✅ 根据模式选择环境配置
    if args.mode == 'test':
        env_config_name = 'waterworld_fast'
        print(f"🏃 测试模式，使用快速环境: max_cycles=500")
    else:
        env_config_name = 'waterworld_standard'
    
    # """
                
                # 找到创建训练器前的注释
                content_lines = content.split('\n')
                new_lines = []
                env_config_added = False
                
                for i, line in enumerate(content_lines):
                    # 如果是 "# 创建训练器" 这一行，在前面插入环境选择代码
                    if '# 创建训练器' in line and 'train_one_prey_algo' in '\n'.join(content_lines[max(0, i-50):i]):
                        if not env_config_added:
                            new_lines.append(insertion_code.rstrip())
                            env_config_added = True
                    
                    # 如果是 trainer = MultiAgentTrainer 的参数行，检查是否需要添加 env_config
                    if 'trainer = MultiAgentTrainer(' in line or (i > 0 and 'trainer = MultiAgentTrainer(' in content_lines[i-1]):
                        if 'total_timesteps=timesteps,' in line and 'env_config=' not in line:
                            # 在 total_timesteps 前插入 env_config
                            new_lines.append(line.replace('total_timesteps=timesteps,', 'env_config=get_env_config(env_config_name),  # ✅ 使用动态环境\n        total_timesteps=timesteps,'))
                            continue
                    
                    new_lines.append(line)
                
                new_content = '\n'.join(new_lines)
                if env_config_added:
                    print(f"   ✓ 使用灵活匹配插入代码")
    
    # 保存修改
    if new_content != content:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        print(f"   ✓ 已修改: {script_path}")
    else:
        print(f"   ✗ 修改失败，请查看详细信息")
        print(f"   ℹ️  请手动在 MultiAgentTrainer 初始化时添加:")
        print(f"       env_config=get_env_config('waterworld_fast' if args.mode == 'test' else 'waterworld_standard')")

# ============================================================================
# 3. 修改 train_stage1_2.py
# ============================================================================
print("\n【步骤3】修改 train_stage1_2.py...")

script_path = Path('scripts/training/train_stage1_2.py')

if not script_path.exists():
    print(f"   ✗ 文件不存在: {script_path}")
else:
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_code = """    # 创建训练器
    trainer = MultiAgentTrainer(
        train_side='predator',
        train_algo=algo,
        opponent_config=opponent_config,
        experiment_name=f"{algo}_predator_guided",
        stage_name=stage_config['stage']['name'],
        generation=stage_config['stage']['generation'],
        version='v1',
        run_mode=args.mode,
        total_timesteps=timesteps,
        device=args.device
    )"""
    
    new_code = """    # ✅ 根据模式选择环境配置
    if args.mode == 'test':
        env_config_name = 'waterworld_fast'
        print(f"🏃 测试模式，使用快速环境: max_cycles=500")
    else:
        env_config_name = 'waterworld_standard'
    
    # 创建训练器
    trainer = MultiAgentTrainer(
        train_side='predator',
        train_algo=algo,
        opponent_config=opponent_config,
        experiment_name=f"{algo}_predator_guided",
        stage_name=stage_config['stage']['name'],
        generation=stage_config['stage']['generation'],
        version='v1',
        run_mode=args.mode,
        env_config=get_env_config(env_config_name),  # ✅ 使用动态环境
        total_timesteps=timesteps,
        device=args.device
    )"""
    
    if old_code in content:
        new_content = content.replace(old_code, new_code)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"   ✓ 已修改: {script_path}")
    else:
        print(f"   ⚠️  未找到匹配代码")
        print(f"   ℹ️  请手动添加环境选择逻辑")

# ============================================================================
# 4. 修改 train_stage1_3.py
# ============================================================================
print("\n【步骤4】修改 train_stage1_3.py...")

script_path = Path('scripts/training/train_stage1_3.py')

if not script_path.exists():
    print(f"   ✗ 文件不存在: {script_path}")
else:
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    old_code = """        # 创建训练器
        trainer = MultiAgentTrainer(
            train_side=train_side,
            train_algo=algo,
            opponent_config=opponent_config,
            experiment_name=f"{algo}_{train_side}_coevo",
            stage_name=f"{stage_config['stage']['name']}/Gen_{generation}",
            generation=generation,
            version=f"v{generation}",
            run_mode=args.mode,
            total_timesteps=args.timesteps_per_gen,
            device=args.device
        )"""
    
    new_code = """        # ✅ 根据模式选择环境配置
        if args.mode == 'test':
            env_config_name = 'waterworld_fast'
            print(f"🏃 测试模式，使用快速环境: max_cycles=500")
        else:
            env_config_name = 'waterworld_standard'
        
        # 创建训练器
        trainer = MultiAgentTrainer(
            train_side=train_side,
            train_algo=algo,
            opponent_config=opponent_config,
            experiment_name=f"{algo}_{train_side}_coevo",
            stage_name=f"{stage_config['stage']['name']}/Gen_{generation}",
            generation=generation,
            version=f"v{generation}",
            run_mode=args.mode,
            env_config=get_env_config(env_config_name),  # ✅ 使用动态环境
            total_timesteps=args.timesteps_per_gen,
            device=args.device
        )"""
    
    if old_code in content:
        new_content = content.replace(old_code, new_code)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"   ✓ 已修改: {script_path}")
    else:
        print(f"   ⚠️  未找到匹配代码")
        print(f"   ℹ️  请手动添加环境选择逻辑")

# ============================================================================
# 5. 验证修改
# ============================================================================
print("\n【步骤5】验证修改...")

all_scripts = [
    'scripts/training/train_stage1_1.py',
    'scripts/training/train_stage1_2.py',
    'scripts/training/train_stage1_3.py'
]

verification_passed = True

for script_name in all_scripts:
    script_path = Path(script_name)
    if script_path.exists():
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含环境选择逻辑
        if "env_config_name = 'waterworld_fast'" in content:
            print(f"   ✓ {script_path.name}: 已包含环境选择逻辑")
        else:
            print(f"   ✗ {script_path.name}: 未找到环境选择逻辑")
            verification_passed = False
        
        # 检查是否使用了动态环境
        if "env_config=get_env_config(env_config_name)" in content:
            print(f"   ✓ {script_path.name}: 已使用动态环境配置")
        else:
            print(f"   ✗ {script_path.name}: 未使用动态环境配置")
            verification_passed = False

# ============================================================================
# 总结
# ============================================================================
print("\n" + "="*70)
if verification_passed:
    print("✅ 全部修复完成！")
else:
    print("⚠️  部分修复完成（可能需要手动调整）")
print("="*70)

print("\n修改内容：")
print("  1. ✓ 创建了 waterworld_fast.yaml (max_cycles=500)")
print("  2. ✓ 修改了 train_stage1_1.py")
print("  3. ✓ 修改了 train_stage1_2.py")
print("  4. ✓ 修改了 train_stage1_3.py")

print("\n关于训练步数：")
print("  - test模式设置: total_timesteps=500")
print("  - PPO的n_steps=2048，所以实际会跑约512-2048步")
print("  - 这是正常的，因为PPO必须收集完整的rollout")
print("  - test模式的目的是验证流程，不是真的训练模型")

print("\n评估长度：")
print("  - 修复前: 3000步/episode × 2 episodes = 6000步")
print("  - 修复后: 500步/episode × 2 episodes = 1000步")
print("  - 评估加速: 6倍")
print("  - 预计评估时间: ~12分钟 → ~2分钟")

print("\n现在运行测试：")
print("  python scripts/training/train_stage1_1.py --mode test")

print("\n应该看到：")
print("  🏃 测试模式，使用快速环境: max_cycles=500")
print("  平均长度: 500-501  (而不是3001)")

print("\n如果仍有问题，手动修改提示：")
print("  在 MultiAgentTrainer(...) 的参数中添加：")
print("  env_config=get_env_config('waterworld_fast' if args.mode == 'test' else 'waterworld_standard')")

print("="*70)