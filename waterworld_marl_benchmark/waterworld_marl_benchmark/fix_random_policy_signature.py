"""
修复 RandomPolicy.predict() 方法签名，使其兼容 SB3 的 evaluate_policy
"""

from pathlib import Path

print("="*70)
print("修复 RandomPolicy.predict() 签名")
print("="*70)

file_path = Path('src/algorithms/random_policy.py')

# 读取文件
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 旧的 predict 方法定义
old_predict = """    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        \"\"\"
        随机采样动作
        
        Args:
            observation: 观察（未使用）
            state: RNN 状态（Random策略不使用，但需要兼容SB3接口）
            deterministic: 是否确定性（未使用）
        
        Returns:
            (action, state) 元组
        \"\"\"
        action = self.action_space.sample()
        return action, None"""

# 新的 predict 方法定义（添加 episode_start 参数）
new_predict = """    def predict(
        self,
        observation: np.ndarray,
        state: Optional[np.ndarray] = None,
        episode_start: Optional[np.ndarray] = None,  # ← 新增参数
        deterministic: bool = False
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        \"\"\"
        随机采样动作
        
        Args:
            observation: 观察（未使用）
            state: RNN 状态（Random策略不使用）
            episode_start: Episode开始标志（Random策略不使用）
            deterministic: 是否确定性（未使用）
        
        Returns:
            (action, state) 元组
        \"\"\"
        action = self.action_space.sample()
        return action, None"""

if old_predict in content:
    content = content.replace(old_predict, new_predict)
    
    # 写回文件
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 已修复 RandomPolicy.predict() 方法")
    print("   添加了 episode_start 参数以兼容 SB3")
else:
    # 尝试更灵活的匹配
    if "def predict(" in content and "deterministic: bool = False" in content:
        # 找到 predict 方法的位置
        predict_start = content.find("    def predict(")
        if predict_start != -1:
            # 找到方法结束位置
            predict_end = content.find("        return action, None", predict_start)
            if predict_end != -1:
                # 提取完整方法
                method_text = content[predict_start:predict_end + len("        return action, None")]
                
                # 检查是否已经有 episode_start
                if "episode_start" not in method_text:
                    # 在 deterministic 前添加 episode_start
                    modified_method = method_text.replace(
                        "        deterministic: bool = False",
                        "        episode_start: Optional[np.ndarray] = None,\n        deterministic: bool = False"
                    )
                    
                    # 更新文档字符串
                    modified_method = modified_method.replace(
                        "            state: RNN 状态（Random策略不使用，但需要兼容SB3接口）",
                        "            state: RNN 状态（Random策略不使用）\n            episode_start: Episode开始标志（Random策略不使用）"
                    )
                    
                    # 替换内容
                    content = content.replace(method_text, modified_method)
                    
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    
                    print("✅ 已修复（灵活匹配）")
                else:
                    print("ℹ️  方法已经包含 episode_start 参数")
    else:
        print("⚠️  未找到 predict 方法，请手动修复")
        print("\n请在 src/algorithms/random_policy.py 中：")
        print("在 predict 方法的参数中添加:")
        print("    episode_start: Optional[np.ndarray] = None,")

print("\n" + "="*70)
print("修复完成！")
print("="*70)