import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from urban import UrbanSound8KDataset
from Model.VGG13 import VGG13
from Model.VGG16 import VGG16
from Model.CRNN import CRNN
from tqdm import tqdm


def train_model(model, train_loader, optimizer, criterion, device):
    """
    训练模型函数
    """
    model.train()
    running_loss = 0.0

    for (inputs, labels) in tqdm(train_loader, ncols=100):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)

    return train_loss


def test_model(model, test_loader, device):
    """
    测试模型函数
    """
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, ncols=100):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predicted.cpu().numpy().tolist())

    acc = accuracy_score(y_true, y_pred)

    return acc


if __name__ == '__main__':
    train_dataset = UrbanSound8KDataset("D:\\UrbanSound8K", train=True)
    test_dataset = UrbanSound8KDataset("D:\\UrbanSound8K", train=False)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=6)
    test_loader = DataLoader(test_dataset, batch_size=64, num_workers=6)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CRNN(num_classes=10,init_weights=True).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        train_loss = train_model(model, train_loader, optimizer, criterion, device)
        acc = test_model(model, test_loader, device)
        print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, test_acc={acc:.4f}")
    torch.save(model, "./models/CRNN.pt")