import transformers
import os
import json
import numpy as np
from pathlib import Path
from prompt_utils import prompt_templates


chat_tokenizer_dir = "./deepseek_v3_tokenizer"

def cal_token(text):
    tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=True)
    return len(tokenizer.encode(text))


def calculate_prompt_tokens_by_config():
    """计算每种实验配置下的提示词token数量，使用与实际实验相同的数据和方法"""
    tokenizer = transformers.AutoTokenizer.from_pretrained(chat_tokenizer_dir, trust_remote_code=True)
    
    # 定义实验参数
    prompt_types = ['minimal', 'basic', 'rich_background']
    batch_sizes = [20, 50, 100]
    
    # 加载与deepseek_emotion_classifier.py相同的数据
    try:
        # 加载测试数据
        with open('datasets/pilot_study/data.txt', 'r', encoding='utf-8') as f:
            test_texts = f.read().splitlines()
        
        # 计算数据总数
        total_data_count = len(test_texts)
        print(f"Loaded {total_data_count} test texts from data.txt")
    except Exception as e:
        print(f"Error loading test data: {e}")
        test_texts = [f"Fallback test text {i}" for i in range(100)]
        total_data_count = len(test_texts)
    
    # 加载示例数据
    try:
        # 与deepseek_emotion_classifier.py中相同的示例加载方式
        examples = []
        example_file = 'datasets/pilot_study/example.txt'
        with open(example_file, 'r', encoding='utf-8') as f:
            example_lines = f.read().splitlines()
        
        # 解析每行，提取文本和标签
        for line in example_lines:
            last_colon_idx = line.rfind(':')
            if last_colon_idx != -1:
                text = line[:last_colon_idx].strip()
                try:
                    label = int(line[last_colon_idx+1:].strip())
                    examples.append({
                        'text': text,
                        'label': label
                    })
                except ValueError as e:
                    print(f"Warning: Could not parse label in example: {e}")
        
        print(f"Loaded {len(examples)} examples from example.txt")
    except Exception as e:
        print(f"Error loading examples: {e}")
        examples = [{"text": "Example text", "label": 0}]
    
    # 导入emotion_dict
    from emotions import emotion_dict
    
    # 初始化结果字典
    token_counts = {}
    detailed_token_counts = {}
    
    # 计算每种配置的token数量
    for prompt_type in prompt_types:
        token_counts[prompt_type] = {}
        detailed_token_counts[prompt_type] = {}
        prompt_template = prompt_templates[prompt_type]
        
        for batch_size in batch_sizes:
            # 计算需要处理的批次数
            num_batches = (total_data_count + batch_size - 1) // batch_size
            
            # 为每个批次计算token并累加
            total_tokens = 0
            batch_tokens = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_data_count)
                
                # 获取当前批次的文本
                batch_texts = test_texts[start_idx:end_idx]
                
                # 使用模板生成实际的提示词
                prompt = prompt_template(batch_texts, examples, emotion_dict)
                
                # 计算token数量
                token_count = cal_token(prompt)
                batch_tokens.append(token_count)
                total_tokens += token_count
            
            # 保存总token数和详细信息
            token_counts[prompt_type][batch_size] = total_tokens
            detailed_token_counts[prompt_type][batch_size] = {
                "total_tokens": total_tokens,
                "num_batches": num_batches,
                "tokens_per_batch": batch_tokens,
                "avg_tokens_per_batch": total_tokens / num_batches if num_batches > 0 else 0
            }
    
    return token_counts, detailed_token_counts

def save_token_counts_to_file(output_path="./outputs/token_counts.json", detailed_output_path="./outputs/detailed_token_counts.json"):
    """计算各配置token数量并保存到文件"""
    token_counts, detailed_token_counts = calculate_prompt_tokens_by_config()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存简化版本到JSON文件
    with open(output_path, 'w') as f:
        json.dump(token_counts, f, indent=4)
    
    # 保存详细版本到JSON文件
    with open(detailed_output_path, 'w') as f:
        json.dump(detailed_token_counts, f, indent=4)
    
    print(f"Token counts saved to {output_path}")
    print(f"Detailed token counts saved to {detailed_output_path}")
    
    return token_counts, detailed_token_counts

# 测试函数
if __name__ == "__main__":
    token_counts, detailed_counts = save_token_counts_to_file()
    
    # 打印结果
    print("\nTotal token counts by configuration:")
    for prompt_type, batch_counts in token_counts.items():
        print(f"\n{prompt_type}:")
        for batch_size, count in batch_counts.items():
            avg_per_batch = detailed_counts[prompt_type][batch_size]["avg_tokens_per_batch"]
            num_batches = detailed_counts[prompt_type][batch_size]["num_batches"]
            print(f"  Batch size {batch_size}: {count} tokens total, {num_batches} batches, avg {avg_per_batch:.1f} tokens/batch")

