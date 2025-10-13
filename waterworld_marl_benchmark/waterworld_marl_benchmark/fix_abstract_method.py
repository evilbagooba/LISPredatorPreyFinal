"""
添加缺失的抽象方法
"""

from pathlib import Path

# 读取当前文件
file_path = Path('src/core/environment.py')
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# 在 MixedAgentVecEnv 类的最后添加缺失的方法
# 找到 env_method 方法后添加
insertion_point = content.find('    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):')
if insertion_point != -1:
    # 找到该方法的结束位置
    end_point = content.find('\n\n', insertion_point)
    if end_point == -1:
        end_point = len(content)
    
    # 在方法后添加缺失的方法
    new_method = '''
    
    def env_is_wrapped(self, wrapper_class, indices=None):
        """检查环境是否被包装"""
        return self.venv.env_is_wrapped(wrapper_class, indices)'''
    
    content = content[:end_point] + new_method + content[end_point:]

# 写回文件
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ 已添加缺失的 env_is_wrapped 方法")
print("\n现在请重新运行测试：")
print("python test_training_system.py")