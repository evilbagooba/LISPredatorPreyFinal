#!/bin/bash
# 一键测试完整流程

echo "================================"
echo "🧪 开始测试完整训练流程"
echo "================================"

# Stage 1.1
echo "▶ Stage 1.1: Prey预热训练"
python scripts/training/train_stage1_1.py --mode test --algos PPO A2C

# Stage 1.2
echo "▶ Stage 1.2: Predator引导训练"
python scripts/training/train_stage1_2.py --mode test --algos PPO A2C --prey-pool test_outputs/fixed_pools/prey_pool_v1

# Stage 1.3 (只跑1代)
echo "▶ Stage 1.3: 共进化训练 (1代)"
python scripts/training/train_stage1_3.py --mode test --max-generations 1 --algos PPO

echo "================================"
echo "✅ 测试完成！"
echo "查看结果: tensorboard --logdir test_outputs/tensorboard_logs"
echo "================================"