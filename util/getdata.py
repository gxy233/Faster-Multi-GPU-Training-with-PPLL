import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator

# 替换为你自己 .event 文件的路径
event_file = '/home/chengqixu/gxy/temp_name1/runs/k2_resnet32_ppll_k8/events.out.tfevents.1730889030.dill-sage.1761867.0'

# 用于存储从 .event 文件中提取的数据
data = {}

# 读取事件文件
for e in summary_iterator(event_file):
    for v in e.summary.value:
        # 每个事件的标签（例如：train_loss，val_acc 等）
        tag = v.tag
        # 事件的 step（如 epoch 数）
        step = e.step
        # 事件的数值
        value = v.simple_value
        
        if tag not in data:
            data[tag] = []
        
        # 保存 (step, value) 对
        data[tag].append((step, value))

# 打印每个 tag 中的数据
for tag, values in data.items():
    print(f"\nTag: {tag}")
    for step, value in values:
        print(f"Step: {step}, Value: {value}")
