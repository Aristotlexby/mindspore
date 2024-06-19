import collections
import json
import numpy as np
import os
import argparse

def CalLayerTimeCosts(config):
    model_name = config.get("model_name")
    num_layers = int(config.get("num_layers"))
    num_gpus = int(config.get("mp_degree"))
    num_steps = int(config.get("steps"))
    layers = []
    
    if model_name == "gpt":
        layer_name = "GPTTransformerDecoderLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
    # TODO: add bert and t5
    elif model_name == "bert":
        layer_name = "TransformerEncoderLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
    elif model_name == "t5":
        layer_name = "TransformerEncoderLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
        layer_name = "TransformerDecoderLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
    elif model_name == "llama2":
        layer_name = "LLamaDecodeLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
    # 读取每张卡上的数据
    costs = []
    
    for j in range(8):
        cost = []
        # 读入timeline文件内容
        gpu_timeline_file = f"cost_profile/results/{model_name}/{model_name}-{num_steps}-{num_layers}-{num_gpus}/ascend_timeline_display_{j}.json"
        data = json.load(open(gpu_timeline_file,'r',encoding='utf-8'),object_pairs_hook=collections.OrderedDict)
        
        # 计算耗时 单位为秒
        Sum = {}
        layer_num = 0
        for layer in layers:
            Sum[layer] = 0

        i = 0
        cur_layer = str(i) + "-" + layer_name
        flag = True
        first_step_flag = True
        ste = 0
        for k,item in enumerate(data):

            if item.get("name") == str(i+1) + "-" + layer_name:
                i += 1
                cur_layer = str(i) + "-" + layer_name
            if item.get("dur") is not None and item.get("name") == cur_layer and flag:
                if not first_step_flag:
                    Sum[item.get("name")] += item.get("dur") / 1e6

            if cur_layer == str(num_layers-1) + "-" + layer_name:
                if item.get("name") == "Gradients":
                    flag = False
            if (not flag):
                if first_step_flag:
                    first_step_flag = False
                # if item.get("name").isdigit():
                if item.get("name") == str(0) + "-" + layer_name:
                    ste += 1
                    flag = True
                    i = 0
                    cur_layer = str(i) + "-" + layer_name
        for layer in layers:
            # cost.append(Sum[layer] / num_steps)
            cost.append(Sum[layer] / ste)
        print(Sum)
        costs.append(cost)

    return costs

def profile(args):
    # 从中提取配置信息
    config = {
        'model_name': args.model_name,
        'num_layers': args.num_layers,
        'mp_degree': args.mp_degree,
        'steps': args.steps,
    }

    # 根据layers和steps获取时间数组
    time_costs = CalLayerTimeCosts(config)

    # 保存为numpy文件
    if config['model_name'] == 'gpt':
        config['model_name'] == "gpt2"
    save_file = f"cost_profile/known_cost_mds/{config['model_name']}_{config['num_layers']}_{config['mp_degree']}.npy"
    # save_file_name = save_path + '{}_{}_{}_{}.npy'.format(model_name, config.get("num_layers"), config.get("data_parallel"), config.get("model_parallel"))
    time_costs_np = np.array(time_costs)
    max_values = np.amax(time_costs_np, axis=0)
    print(max_values)

    np.save(save_file, max_values)


if __name__ == "__main__":
    # 解析命令行参数
    # --model_name=$MODEL_NAME --num_layers=$NUM_LAYERS --mp_degree=$MP_DEGREE --steps=$NUM_SAMPLES
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=["gpt", "bert", "t5", "llama2"])
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--mp_degree', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    
    profile(args)import collections
import json
import numpy as np
import os
import argparse

def CalLayerTimeCosts(config):
    model_name = config.get("model_name")
    num_layers = int(config.get("num_layers"))
    num_gpus = int(config.get("mp_degree"))
    num_steps = int(config.get("steps"))
    layers = []
    
    if model_name == "gpt":
        layer_name = "GPTTransformerDecoderLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
    # TODO: add bert and t5
    elif model_name == "bert":
        layer_name = "TransformerEncoderLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
    elif model_name == "t5":
        layer_name = "TransformerEncoderLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
        layer_name = "TransformerDecoderLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
    elif model_name == "llama2":
        layer_name = "LLamaDecodeLayer"
        for i in range(num_layers):
            layers.append(str(i) + "-" + layer_name)
    # 读取每张卡上的数据
    costs = []
    
    for j in range(8):
        cost = []
        # 读入timeline文件内容
        gpu_timeline_file = f"cost_profile/results/{model_name}/{model_name}-{num_steps}-{num_layers}-{num_gpus}/ascend_timeline_display_{j}.json"
        data = json.load(open(gpu_timeline_file,'r',encoding='utf-8'),object_pairs_hook=collections.OrderedDict)
        
        # 计算耗时 单位为秒
        Sum = {}
        layer_num = 0
        for layer in layers:
            Sum[layer] = 0

        i = 0
        cur_layer = str(i) + "-" + layer_name
        flag = True
        first_step_flag = True
        ste = 0
        for k,item in enumerate(data):

            if item.get("name") == str(i+1) + "-" + layer_name:
                i += 1
                cur_layer = str(i) + "-" + layer_name
            if item.get("dur") is not None and item.get("name") == cur_layer and flag:
                if not first_step_flag:
                    Sum[item.get("name")] += item.get("dur") / 1e6

            if cur_layer == str(num_layers-1) + "-" + layer_name:
                if item.get("name") == "Gradients":
                    flag = False
            if (not flag):
                if first_step_flag:
                    first_step_flag = False
                # if item.get("name").isdigit():
                if item.get("name") == str(0) + "-" + layer_name:
                    ste += 1
                    flag = True
                    i = 0
                    cur_layer = str(i) + "-" + layer_name
        for layer in layers:
            # cost.append(Sum[layer] / num_steps)
            cost.append(Sum[layer] / ste)
        print(Sum)
        costs.append(cost)

    return costs

def profile(args):
    # 从中提取配置信息
    config = {
        'model_name': args.model_name,
        'num_layers': args.num_layers,
        'mp_degree': args.mp_degree,
        'steps': args.steps,
    }

    # 根据layers和steps获取时间数组
    time_costs = CalLayerTimeCosts(config)

    # 保存为numpy文件
    if config['model_name'] == 'gpt':
        config['model_name'] == "gpt2"
    save_file = f"cost_profile/known_cost_mds/{config['model_name']}_{config['num_layers']}_{config['mp_degree']}.npy"
    # save_file_name = save_path + '{}_{}_{}_{}.npy'.format(model_name, config.get("num_layers"), config.get("data_parallel"), config.get("model_parallel"))
    time_costs_np = np.array(time_costs)
    max_values = np.amax(time_costs_np, axis=0)
    print(max_values)

    np.save(save_file, max_values)


if __name__ == "__main__":
    # 解析命令行参数
    # --model_name=$MODEL_NAME --num_layers=$NUM_LAYERS --mp_degree=$MP_DEGREE --steps=$NUM_SAMPLES
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, choices=["gpt", "bert", "t5", "llama2"])
    parser.add_argument('--num_layers', type=int, required=True)
    parser.add_argument('--mp_degree', type=int, required=True)
    parser.add_argument('--steps', type=int, required=True)
    args = parser.parse_args()
    
    profile(args)
