import torch
from urban import UrbanSound8KDataset_classwise_onlyone
from torch.utils.data import DataLoader
import random
import os

def main():
    # 假设你有三个PyTorch模型 model1, model2, model3，并且已经加载了预训练的权重
    base_path = 'data'

    def filter_correct_predictions(class_, data_loader, model1, model2, model3, num_samples=100):
        correct_data = []
        correct_labels = []

        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            outputs3 = model3(inputs)

            predictions1 = torch.argmax(outputs1, dim=1)
            predictions2 = torch.argmax(outputs2, dim=1)
            predictions3 = torch.argmax(outputs3, dim=1)

            correct_mask = (predictions1 == labels) & (predictions2 == labels) & (predictions3 == labels)

            correct_data.extend(inputs[correct_mask])
            correct_labels.extend(labels[correct_mask])


        print(len(correct_data))
        # 随机选择指定数量的样本
        random_indices = random.sample(range(len(correct_data)), min(num_samples, len(correct_data)))
        correct_data = torch.stack([correct_data[i] for i in random_indices])
        correct_labels = torch.stack([correct_labels[i] for i in random_indices])

        # 保存为张量文件
        torch.save(correct_data, os.path.join(base_path, f'class{class_}', 'correct_datas.pt'))
        torch.save(correct_labels, os.path.join(base_path, f'class{class_}', 'correct_labels.pt'))

    model1 = torch.load("models/VGG13.pt").eval()
    model2 = torch.load("models/VGG16.pt").eval()
    model3 = torch.load("models/CRNN.pt").eval()

    for i in range(10):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 使用你的数据集和数据加载器
        datasets = UrbanSound8KDataset_classwise_onlyone("D:\\UrbanSound8K", class_num=i)
        data_loader = DataLoader(datasets, batch_size=64, shuffle=False)
        # 调用函数筛选出所有三个模型都能够预测正确的数据
        filter_correct_predictions(i, data_loader, model1, model2, model3)


if __name__ == "__main__":
    main()
