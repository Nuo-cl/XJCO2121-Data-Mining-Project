import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.append("../")
from emotions import emotion_dict

# 定义实验参数
prompt_types = ['minimal', 'basic', 'rich_background']
batch_sizes = [20, 50, 100]

# 创建输出目录
output_dir = Path('./outputs')
output_dir.mkdir(exist_ok=True, parents=True)

# 创建实验配置标签
configs = [f"{prompt}_{bs}" for prompt in prompt_types for bs in batch_sizes]

# 从emotion_dict获取情绪类别并按照ID排序
emotions = sorted(emotion_dict.items(), key=lambda x: x[1])
emotion_names = [emotion[0] for emotion in emotions]

# 初始化热力图数据矩阵（28行情绪 × 9列配置）
heatmap_data = np.zeros((len(emotion_names), len(configs)))
heatmap_data.fill(np.nan)  # 填充NaN以便在热力图上显示为灰色或特定颜色

# 从结果文件中读取每个情绪类别的准确率
for i, config in enumerate(configs):
    # 修复拆分问题，处理包含下划线的prompt_type
    parts = config.split('_')
    # 最后一部分是batch_size，其余部分组成prompt_type
    bs = parts[-1]
    prompt_type = '_'.join(parts[:-1])
    
    result_dir = f"../result/{prompt_type}_batchsize{bs}"
    
    # 检查目录是否存在
    if not os.path.exists(result_dir):
        print(f"Warning: {result_dir} does not exist")
        continue
    
    # 尝试读取评估结果文件
    eval_file = os.path.join(result_dir, "evaluation_results.json")
    if not os.path.exists(eval_file):
        print(f"Warning: {eval_file} does not exist")
        continue
    
    # 读取评估结果并提取每个情绪类别的准确率
    try:
        with open(eval_file, 'r') as f:
            results = json.load(f)
            class_metrics = results.get('class_metrics', {})
            
            for j, emotion_name in enumerate(emotion_names):
                # 尝试获取该情绪的指标，如果不存在则跳过
                emotion_metrics = class_metrics.get(emotion_name)
                if emotion_metrics:
                    # 使用准确率或F1分数（取决于评估文件中的可用指标）
                    accuracy = emotion_metrics.get('precision', None)
                    if accuracy is not None:
                        heatmap_data[j, i] = accuracy
    except Exception as e:
        print(f"Error reading {eval_file}: {e}")

# 创建热力图
# 调整图形大小，使单元格成为正方形
# 数据是28行×9列，因此高宽比约为3.11:1
plt.figure(figsize=(10, 31))

# 设置热力图
ax = sns.heatmap(
    heatmap_data,
    annot=False,             # 不显示数值
    cmap="coolwarm",         # 蓝到红的配色
    linewidths=0.5,          # 单元格边框宽度
    cbar_kws={
        "label": "Accuracy",
        "shrink": 0.5,       # 使图例高度与热力图高度一致
        "aspect": 26,         # 调整宽高比，使图例更窄
        "pad": 0.02,          # 调整与热力图的距离
    },  
    vmin=0.0,                # 最小值
    vmax=1.0,                # 最大值
    mask=np.isnan(heatmap_data),  # 隐藏NaN值
    square=True              # 确保单元格为正方形
)

# 设置坐标轴标签
plt.yticks(np.arange(len(emotion_names))+0.5, emotion_names, fontsize=24, rotation=45, ha='right', va='top')
plt.xticks(np.arange(len(configs))+0.5, configs, fontsize=22, rotation=45, ha='right')

# 设置标题
plt.title('Emotion Classification Accuracy\nAcross Different Configurations', fontsize=36, fontweight='bold', pad=20)

# 调整布局以确保所有元素都可见
plt.tight_layout()

# 设置色温图例的标签和刻度字体大小
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel("Accuracy", fontsize=24)
cbar.ax.tick_params(labelsize=20)

# 保存图表
output_path = output_dir / 'emotion_accuracy_heatmap.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Heatmap saved to: {output_path}")

# 显示图表（可选）
# plt.show() 