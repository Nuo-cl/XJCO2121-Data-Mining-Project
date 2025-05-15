import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 定义实验参数
prompt_types = ['minimal', 'basic', 'rich_background']
batch_sizes = [20, 50, 100]
# 更换为更美观的配色方案
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 使用更现代、美观的配色

# 创建输出目录
output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True, parents=True)

# 初始化数据存储
f1_scores = {prompt_type: [] for prompt_type in prompt_types}

# 从结果文件中读取F1分数
for prompt_type in prompt_types:
    for batch_size in batch_sizes:
        # 构建结果目录路径
        result_dir = f"../result/{prompt_type}_batchsize{batch_size}"
        
        # 检查目录是否存在
        if not os.path.exists(result_dir):
            print(f"Warning: {result_dir} does not exist")
            f1_scores[prompt_type].append(None)  # 如果目录不存在，添加None作为占位符
            continue
        
        # 尝试读取评估结果文件
        eval_file = os.path.join(result_dir, "evaluation_results.json")
        if not os.path.exists(eval_file):
            print(f"Warning: {eval_file} does not exist")
            f1_scores[prompt_type].append(None)
            continue
        
        # 读取评估结果并提取F1分数
        try:
            with open(eval_file, 'r') as f:
                results = json.load(f)
                f1_score = results.get('f1')
                if f1_score is not None:
                    f1_scores[prompt_type].append(f1_score)
                else:
                    print(f"Warning: F1 score not found in {eval_file}")
                    f1_scores[prompt_type].append(None)
        except Exception as e:
            print(f"Error reading {eval_file}: {e}")
            f1_scores[prompt_type].append(None)

# 创建折线图
plt.figure(figsize=(10, 6))

# 设置图表内边距，缩小绘图区域，并为底部图例留出空间
plt.subplots_adjust(left=0.15, right=0.85, top=0.9, bottom=0.2)

# 为每种提示词类型绘制一条线
for i, prompt_type in enumerate(prompt_types):
    # 过滤掉None值
    valid_indices = [j for j, score in enumerate(f1_scores[prompt_type]) if score is not None]
    valid_batch_sizes = [batch_sizes[j] for j in valid_indices]
    valid_scores = [f1_scores[prompt_type][j] for j in valid_indices]
    
    if valid_scores:  # 只有当有有效分数时才绘制
        plt.plot(valid_batch_sizes, valid_scores, marker='o', linestyle='-', color=colors[i], 
                 label=f'{prompt_type}', markerfacecolor='white', markeredgecolor=colors[i], 
                 markeredgewidth=1.5, markersize=12, linewidth=4.0)

# 添加图表标题和标签
plt.title('Comparison of F1 scores for different prompt\ntypes and batch sizes', 
          fontsize=20, fontweight='bold', pad=20)
plt.xlabel('Batch Size', fontsize=18)
plt.ylabel('F1 Score', fontsize=18, labelpad=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=16, loc='upper right', bbox_to_anchor=(1.0, -0.05))

# 设置X轴刻度为具体的批处理大小
plt.xticks(batch_sizes, fontsize=16)
plt.yticks(fontsize=16)

# 保存图表
output_path = output_dir / 'f1_line_chart.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Plot saved to: {output_path}")

# 显示图表（可选）
# plt.show()
