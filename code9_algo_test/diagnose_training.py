"""
训练诊断工具 - 检测梯度爆炸/消失的根本原因
"""

import torch
import numpy as np
from collections import defaultdict
from datetime import datetime
import json
import os

class TrainingDiagnostics:
    """训练过程诊断器"""
    
    def __init__(self, log_dir="diagnostics"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 统计容器
        self.stats = defaultdict(list)
        self.anomalies = []
        
        # 阈值设置
        self.thresholds = {
            'reward_max': 100.0,
            'reward_min': -100.0,
            'value_max': 200.0,
            'value_min': -200.0,
            'grad_norm_max': 10.0,
            'loss_max': 1000.0,
            'td_error_max': 50.0,
        }
        
    def check_batch_data(self, batch, step, agent_name=""):
        """检查batch数据的异常"""
        prefix = f"{agent_name}/" if agent_name else ""
        
        # 1. 检查奖励分布
        if hasattr(batch, 'rew'):
            rewards = batch.rew
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.cpu().numpy()
            
            r_mean = np.mean(rewards)
            r_std = np.std(rewards)
            r_min = np.min(rewards)
            r_max = np.max(rewards)
            
            self.stats[f'{prefix}reward_mean'].append((step, r_mean))
            self.stats[f'{prefix}reward_std'].append((step, r_std))
            
            # 检查异常值
            if r_max > self.thresholds['reward_max']:
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': 'reward_too_high',
                    'value': float(r_max),
                    'threshold': self.thresholds['reward_max']
                })
            
            if r_min < self.thresholds['reward_min']:
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': 'reward_too_low',
                    'value': float(r_min),
                    'threshold': self.thresholds['reward_min']
                })
            
            # 检查NaN/Inf
            if np.any(np.isnan(rewards)):
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': 'reward_nan',
                    'count': int(np.sum(np.isnan(rewards)))
                })
            
            if np.any(np.isinf(rewards)):
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': 'reward_inf',
                    'count': int(np.sum(np.isinf(rewards)))
                })
        
        # 2. 检查returns（如果有）
        if hasattr(batch, 'returns'):
            returns = batch.returns
            if isinstance(returns, torch.Tensor):
                returns = returns.cpu().numpy()
            
            ret_mean = np.mean(returns)
            ret_max = np.max(returns)
            ret_min = np.min(returns)
            
            self.stats[f'{prefix}returns_mean'].append((step, ret_mean))
            
            if np.abs(ret_max) > 500:
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': 'returns_extreme',
                    'max': float(ret_max),
                    'min': float(ret_min)
                })
    
    def check_network_output(self, values, step, agent_name="", name="critic"):
        """检查网络输出"""
        prefix = f"{agent_name}/" if agent_name else ""
        
        # ✅ 处理多种输入类型
        if isinstance(values, torch.Tensor):
            values = values.detach().cpu().numpy()
        elif isinstance(values, (list, tuple)):
            values = np.array(values)
        elif not isinstance(values, np.ndarray):
            try:
                values = np.array(values)
            except:
                print(f"Warning: Cannot convert {type(values)} to numpy array")
                return
        
        # 确保是 1D 或 2D 数组
        if values.ndim == 0:
            values = values.reshape(1)
        
        v_mean = float(np.mean(values))
        v_std = float(np.std(values))
        v_min = float(np.min(values))
        v_max = float(np.max(values))
        
        self.stats[f'{prefix}{name}_output_mean'].append((step, v_mean))
        self.stats[f'{prefix}{name}_output_std'].append((step, v_std))
        
        # 检查异常
        if v_max > self.thresholds['value_max']:
            self.anomalies.append({
                'step': step,
                'agent': agent_name,
                'type': f'{name}_output_too_high',
                'value': float(v_max),
                'mean': float(v_mean)
            })
        
        if v_min < self.thresholds['value_min']:
            self.anomalies.append({
                'step': step,
                'agent': agent_name,
                'type': f'{name}_output_too_low',
                'value': float(v_min),
                'mean': float(v_mean)
            })
        
        if np.any(np.isnan(values)):
            self.anomalies.append({
                'step': step,
                'agent': agent_name,
                'type': f'{name}_output_nan',
                'count': int(np.sum(np.isnan(values)))
            })
        
        if np.any(np.isinf(values)):
            self.anomalies.append({
                'step': step,
                'agent': agent_name,
                'type': f'{name}_output_inf',
                'count': int(np.sum(np.isinf(values)))
            })
    
    def check_gradients(self, model, step, agent_name="", model_name=""):
        """检查梯度"""
        prefix = f"{agent_name}/{model_name}/" if agent_name and model_name else ""
        
        total_norm = 0.0
        grad_norms = {}
        nan_grads = []
        inf_grads = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                grad_norms[name] = param_norm
                
                # 检查NaN/Inf
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
                if torch.isinf(param.grad).any():
                    inf_grads.append(name)
        
        total_norm = total_norm ** 0.5
        self.stats[f'{prefix}grad_norm'].append((step, total_norm))
        
        # 检查异常
        if total_norm > self.thresholds['grad_norm_max']:
            # 找出最大的几个梯度
            top_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:5]
            
            self.anomalies.append({
                'step': step,
                'agent': agent_name,
                'model': model_name,
                'type': 'grad_norm_high',
                'total_norm': float(total_norm),
                'top_grads': [(name, float(norm)) for name, norm in top_grads]
            })
        
        if nan_grads:
            self.anomalies.append({
                'step': step,
                'agent': agent_name,
                'model': model_name,
                'type': 'grad_nan',
                'params': nan_grads
            })
        
        if inf_grads:
            self.anomalies.append({
                'step': step,
                'agent': agent_name,
                'model': model_name,
                'type': 'grad_inf',
                'params': inf_grads
            })
    
    def check_loss(self, losses, step, agent_name=""):
        """检查损失函数"""
        prefix = f"{agent_name}/" if agent_name else ""
        
        for loss_name, loss_value in losses.items():
            # ✅ 处理多种类型的损失值
            if isinstance(loss_value, (list, tuple)):
                # 如果是列表/元组，取平均值
                loss_value = float(np.mean(loss_value))
            elif isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item() if loss_value.numel() == 1 else float(loss_value.mean())
            elif not isinstance(loss_value, (int, float)):
                # 跳过无法处理的类型
                continue
            
            # ✅ 确保是标量
            loss_value = float(loss_value)
            
            self.stats[f'{prefix}loss_{loss_name}'].append((step, loss_value))
            
            # 检查异常
            if loss_value > self.thresholds['loss_max']:
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': f'loss_{loss_name}_high',
                    'value': float(loss_value)
                })
            
            if np.isnan(loss_value):
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': f'loss_{loss_name}_nan'
                })
            
            if np.isinf(loss_value):
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': f'loss_{loss_name}_inf'
                })
    def check_td_error(self, values, returns, step, agent_name=""):
        """检查TD误差 - 安全版本"""
        prefix = f"{agent_name}/" if agent_name else ""
        
        try:
            # 转换并验证
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().numpy()
            if isinstance(returns, torch.Tensor):
                returns = returns.detach().cpu().numpy()
            
            # 确保1D数组
            values = np.atleast_1d(values.flatten())
            returns = np.atleast_1d(returns.flatten())
            
            # 取最小长度
            min_len = min(len(values), len(returns))
            if min_len == 0:
                return
            
            values = values[:min_len]
            returns = returns[:min_len]
            
            td_error = returns - values
            td_mean = float(np.mean(np.abs(td_error)))
            td_max = float(np.max(np.abs(td_error)))
            
            self.stats[f'{prefix}td_error_mean'].append((step, td_mean))
            self.stats[f'{prefix}td_error_max'].append((step, td_max))
            
            if td_max > self.thresholds['td_error_max']:
                max_idx = int(np.argmax(np.abs(td_error)))
                self.anomalies.append({
                    'step': step,
                    'agent': agent_name,
                    'type': 'td_error_high',
                    'max_td': float(td_error[max_idx]),
                    'value_at_max': float(values[max_idx]),
                    'return_at_max': float(returns[max_idx])
                })
        except Exception as e:
            # 静默失败，不影响训练
            pass
    
    def generate_report(self):
        """生成诊断报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.log_dir, f"diagnostic_report_{timestamp}.json")
        
        # 分析异常模式
        anomaly_summary = defaultdict(int)
        for anomaly in self.anomalies:
            anomaly_summary[anomaly['type']] += 1
        
        # 找出最严重的异常
        critical_anomalies = [
            a for a in self.anomalies 
            if a['type'] in ['grad_nan', 'grad_inf', 'reward_nan', 'reward_inf', 
                            'loss_vf_nan', 'loss_vf_inf']
        ]
        
        report = {
            'timestamp': timestamp,
            'total_anomalies': len(self.anomalies),
            'anomaly_types': dict(anomaly_summary),
            'critical_anomalies': critical_anomalies[:10],  # 前10个最严重的
            'all_anomalies': self.anomalies,
            'stats_summary': self._summarize_stats()
        }
        
        # 保存报告
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 打印摘要
        print(f"\n{'='*60}")
        print(f"诊断报告已生成: {report_path}")
        print(f"{'='*60}")
        print(f"总异常数: {len(self.anomalies)}")
        print(f"\n异常类型统计:")
        for atype, count in sorted(anomaly_summary.items(), key=lambda x: x[1], reverse=True):
            print(f"  {atype}: {count}")
        
        if critical_anomalies:
            print(f"\n⚠️ 发现 {len(critical_anomalies)} 个严重异常 (NaN/Inf)!")
            print(f"详情请查看: {report_path}")
        
        return report_path
    
    def _summarize_stats(self):
        """统计数据摘要"""
        summary = {}
        for key, values in self.stats.items():
            if values:
                data = [v for _, v in values]
                summary[key] = {
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data)),
                    'min': float(np.min(data)),
                    'max': float(np.max(data)),
                    'count': len(data)
                }
        return summary