# Importing necessary libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pandas as pd
from tensorboardX import SummaryWriter
from tqdm import tqdm
from google.colab import drive
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch.nn.functional as F

# Connect Colab to Google Drive
drive.mount('/content/drive')

# Download Celeb-DF dataset
!wget -P /content/drive/MyDrive/datasets/ https://github.com/yuezunli/Celeb-DF-v2/releases/download/v2.0/Celeb-DF-v2.zip
!unzip /content/drive/MyDrive/datasets/Celeb-DF-v2.zip -d /content/drive/MyDrive/datasets/

# Xception model implementation (directly integrated from the provided xception.py)
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            self.rep = nn.Sequential(*rep)

        if strides != 1:
            self.pool = nn.MaxPool2d(3, strides, 1)
        else:
            self.pool = None

    def forward(self, inp):
        x = self.rep(inp)
        if self.pool is not None:
            x = self.pool(x)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x

class Xception(nn.Module):
    def __init__(self, num_classes=2):
        super(Xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.block1 = Block(32, 64, reps=2, strides=2, start_with_relu=False, grow_first=True)
        self.block2 = Block(64, 128, reps=2, strides=2, start_with_relu=True, grow_first=True)
        self.block3 = Block(128, 256, reps=2, strides=2, start_with_relu=True, grow_first=True)

        self.last_linear = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.last_linear(x)
        return x

# Set the device for computations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Xception(num_classes=2).to(device)

# Freezing all layers except the last fully connected layer to avoid overfitting
for param in model.parameters():
    param.requires_grad = False
for param in model.last_linear.parameters():
    param.requires_grad = True

# SVM classifier definition
def train_svm(features, labels):
    # Create an SVM model pipeline with normalization
    svm_model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    svm_model.fit(features, labels)
    return svm_model

# Training loop with extraction of features and using SVM for classification
def train_on_epochs(train_loader: DataLoader, test_loader: DataLoader, epochs):
    # Prepare optimizer for the fully connected layers of the Xception model
    optimizer = optim.Adam(model.last_linear.parameters(), lr=1e-4)
    writer = SummaryWriter(logdir='./log-xception-svm')

    for ep in range(epochs):
        model.train()
        train_features, train_labels = [], []
        train_losses = []

        # Extract features from Xception and gather labels for training SVM
        for i, (X, y) in enumerate(train_loader):
            X = X.to(device)
            y = y.to(device)
            optimizer.zero_grad()

            features = model.forward(X)  # Extracting features before the fully connected layer
            features = features.view(features.size(0), -1).cpu().detach().numpy()
            train_features.extend(features)
            train_labels.extend(y.cpu().numpy())

            logits = model.last_linear(torch.tensor(features).to(device))
            loss = nn.CrossEntropyLoss()(logits, y)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            if (i + 1) % 10 == 0:
                print(f"[Epoch {ep}] Training step {i + 1}/{len(train_loader)}, Loss: {loss.item()}")

        # Train SVM using extracted features from Xception
        svm_model = train_svm(train_features, train_labels)

        # Validation phase
        model.eval()
        test_features, test_labels = [], []
        y_gd, y_pred = [], []
        with torch.no_grad():
            for X, y in tqdm(test_loader, desc='Validating'):
                X, y = X.to(device), y.to(device)
                features = model.forward(X)  # Extracting features from the Xception model
                features = features.view(features.size(0), -1).cpu().detach().numpy()
                test_features.extend(features)
                test_labels.extend(y.cpu().numpy())

            # Predict with SVM on the validation set
            predictions = svm_model.predict(test_features)
            y_pred = predictions
            y_gd = test_labels

            # Calculating metrics
            accuracy = accuracy_score(y_gd, y_pred)
            recall = recall_score(y_gd, y_pred)
            f1 = f1_score(y_gd, y_pred)
            precision = precision_score(y_gd, y_pred)
            print(f'[Epoch {ep}] Validation accuracy: {accuracy}, recall: {recall}, f1: {f1}, precision: {precision}')

            # Logging metrics
            writer.add_scalar('test_accuracy', accuracy, ep)
            writer.add_scalar('test_recall', recall, ep)
            writer.add_scalar('test_f1', f1, ep)
            writer.add_scalar('test_precision', precision, ep)

    writer.close()
    print('Training completed!')

# Arguments and dataset loaders
if __name__ == "__main__":
    data_path = '/content/drive/MyDrive/datasets/Celeb-DF-v2'  # Modify this path accordingly
    train_transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_data = ImageFolder(root=os.path.join(data_path, 'train'), transform=train_transform)
    test_data = ImageFolder(root=os.path.join(data_path, 'test'), transform=train_transform)

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=4)

    train_on_epochs(train_loader, test_loader, epochs=20)
