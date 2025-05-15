import os
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.ticker as ticker
from matplotlib.legend_handler import HandlerLine2D

# 创建输出目录
output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True, parents=True)

# 定义Token数据（从控制台输出中提取）
token_data = {
    "minimal": {
        20: 6366,
        50: 4811,
        100: 4292
    },
    "basic": {
        20: 12976,
        50: 7455,
        100: 5614
    },
    "rich_background": {
        20: 13886,
        50: 7819,
        100: 5796
    }
}

# 初始化数据存储
accuracy_scores = {}

# 读取准确率数据
for prompt_type in token_data.keys():
    accuracy_scores[prompt_type] = {}
    for batch_size in token_data[prompt_type].keys():
        # 构建结果目录路径
        result_dir = f"../result/{prompt_type}_batchsize{batch_size}"
        
        # 检查目录是否存在
        if not os.path.exists(result_dir):
            print(f"Warning: {result_dir} does not exist")
            accuracy_scores[prompt_type][batch_size] = None
            continue
            
        # 尝试读取评估结果文件
        eval_file = os.path.join(result_dir, "evaluation_results.json")
        if not os.path.exists(eval_file):
            print(f"Warning: {eval_file} does not exist")
            accuracy_scores[prompt_type][batch_size] = None
            continue
            
        # 读取评估结果并提取准确率
        try:
            with open(eval_file, 'r') as f:
                results = json.load(f)
                accuracy = results.get('accuracy')
                if accuracy is not None:
                    accuracy_scores[prompt_type][batch_size] = accuracy
                else:
                    print(f"Warning: Accuracy not found in {eval_file}")
                    accuracy_scores[prompt_type][batch_size] = None
        except Exception as e:
            print(f"Error reading {eval_file}: {e}")
            accuracy_scores[prompt_type][batch_size] = None

# 准备散点图数据
token_values = []
accuracy_values = []
colors = []
labels = []
markers = []

color_map = {
    "minimal": "#1f77b4",     # 蓝色
    "basic": "#ff7f0e",       # 橙色
    "rich_background": "#2ca02c"  # 绿色
}

marker_map = {
    20: "o",      # 圆形
    50: "s",      # 方形
    100: "^"      # 三角形
}

# 为每个配置创建数据点
for prompt_type in token_data.keys():
    for batch_size in token_data[prompt_type].keys():
        token_count = token_data[prompt_type][batch_size]
        accuracy = accuracy_scores[prompt_type][batch_size]
        
        if accuracy is not None:
            token_values.append(token_count)
            accuracy_values.append(accuracy)
            colors.append(color_map[prompt_type])
            markers.append(marker_map[batch_size])
            labels.append(f"{prompt_type}_{batch_size}")

# 创建散点图
plt.figure(figsize=(12, 8))

# 为每个点绘制散点
for i in range(len(token_values)):
    plt.scatter(token_values[i], accuracy_values[i], 
                color=colors[i], 
                marker=markers[i], 
                s=250,  # 点的大小
                alpha=0.8, 
                edgecolors='none', 
                linewidths=1.5)
    
    # 添加标签，增大字体
    plt.annotate(labels[i], 
                 (token_values[i], accuracy_values[i]),
                 textcoords="offset points", 
                 xytext=(0, 10), 
                 ha='center',
                 fontsize=14)

# 设置图表标题和标签，增大字体
plt.title('Prompt Token Count vs Accuracy', fontsize=28, fontweight='bold', pad=20)
plt.xlabel('Total Token Count', fontsize=24)
plt.ylabel('Accuracy', fontsize=24)

# 设置网格和轴格式
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

# 增大坐标轴刻度字体
plt.xticks(fontsize=18)  # 添加刻度字体大小
plt.yticks(fontsize=18)  # 添加刻度字体大小

# 添加图例
prompt_legend = [plt.Line2D([0], [0], color=color, lw=4, label=prompt_type) 
                 for prompt_type, color in color_map.items()]
batch_legend = [plt.Line2D([0], [0], marker=marker, color='black', label=f'Batch {batch_size}',
                          markersize=10, linestyle='None')  # 增大图例marker大小，从10到12
                for batch_size, marker in marker_map.items()]

# 创建一个组合图例，将两个图例并排放在右下角
all_legend_elements = prompt_legend + batch_legend
plt.legend(handles=all_legend_elements, 
           loc='lower right', 
           ncol=2,  # 两列并排
           fontsize=16,  # 从14增加到18
           handler_map={plt.Line2D: HandlerLine2D(numpoints=1)},
           columnspacing=1.0)  # 调整列间距

# 保存图表
output_path = output_dir / 'token_vs_accuracy_scatter.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Scatter plot saved to: {output_path}")

# 显示图表（可选）
# plt.show() 