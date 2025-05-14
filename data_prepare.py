import os
import numpy as np
import csv


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(file_list):
    data = []
    for file in file_list:
        with open(file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines:
                data.append(line.split("\t")[:-1])
    return data

def filter_multiple_label(data):
    filtered_data = []
    for line in data:
        if len(line[1].split(",")) > 1:
            continue
        filtered_data.append(line)
    return filtered_data

def save_processed_data(data, path, rate=0, number=None, use_number=False):
    mkdir(path)
    
    if use_number:
        selection_budget = number
    else:
        selection_budget = int(len(data) * rate)
        
    random_index = []
    for i in range(selection_budget):
        random_index.append(np.random.randint(0, len(data)))
        
    with open(path+"data.txt", "w", encoding="utf-8") as f:
        for i in random_index:
            f.write(data[i][0]+"\n")
    
    labels = []
    for i in range(len(random_index)):
        labels.append(data[random_index[i]][1])
    labels = np.array(labels, dtype=np.int32)
    np.save(path+"labels.npy", labels)
    
def save_example_data(data, path):
    example_data = [""]*28
    
    count = 0
    while count < 28:
        random_index = np.random.randint(0, len(data))
        if example_data[int(data[random_index][1])] == "":
            example_data[int(data[random_index][1])] = data[random_index]
            count += 1
    
    with open(path+"example.txt", "w", encoding="utf-8") as f:
        for i in range(28):
            f.write(":".join(example_data[i])+"\n")

if __name__ == "__main__":
    file_list = ["./datasets/prepared/train.tsv"]
    data = load_data(file_list)
    filtered_data = filter_multiple_label(data)
    
    # save pilot study data
    save_processed_data(filtered_data, "./datasets/pilot_study/", number=100, use_number=True)
    
    save_example_data(filtered_data, "./datasets/pilot_study/")