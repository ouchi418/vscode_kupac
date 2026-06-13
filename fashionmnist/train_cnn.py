import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# --- ハイパーパラメータの設定 ---
EPOCHS = 60              # 拡張データに対応するため学習期間を延長
BATCH_SIZE = 128
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. データセットの準備 (安全なデータ拡張の導入) ---
# 訓練用：画像を最大2ピクセルだけランダムに平行移動（靴の向きは維持）
transform_train = transforms.Compose([
    transforms.RandomCrop(28, padding=2), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# テスト用：データ拡張は行わず、正規化のみ
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

print("Loading Fashion-MNIST Dataset with Augmentation...")
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. モデルの定義 (VGGスタイルの強力なCNNアーキテクチャ) ---
class FashionCNN_Pro(nn.Module):
    def __init__(self):
        super(FashionCNN_Pro, self).__init__()
        
        # 第1ブロック: Conv(32) x2 -> MaxPool
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 出力サイズ: 14x14
        )
        
        # 第2ブロック: Conv(64) x2 -> MaxPool
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 出力サイズ: 7x7
        )

        # 第3ブロック: Conv(128) -> MaxPool (さらに特徴を凝縮)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) # 出力サイズ: 3x3
        )
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256), # 全結合層の前にも正規化を入れて安定化
            nn.ReLU(),
            nn.Dropout(0.4),     # 表現力が上がった分、Dropoutを少し強めに
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x

model = FashionCNN_Pro().to(DEVICE)
print(f"Using device: {DEVICE}")

# --- 3. 最適化手法・損失関数・スケジューラ ---
# Label Smoothingで過信を防ぎ、AdamWのWeight Decayで過学習を抑制
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# --- 4. 学習ループ ---
print("Starting Competitive Training...")
best_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    start_time = time.time()
    
    # 【訓練】
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    scheduler.step()

    # 【評価】
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_acc = 100 * correct / total
    epoch_loss = running_loss / len(train_loader)
    current_lr = scheduler.get_last_lr()[0]
    
    # 最高精度を更新したら保存
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_cnn_model.pth')
        
    elapsed = time.time() - start_time
    print(f"Epoch {epoch:02d}/{EPOCHS} | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} | Test Acc: {test_acc:.2f}% | Time: {elapsed:.1f}s")

print("-" * 40)
print(f"Training Complete! Best Test Accuracy: {best_acc:.2f}%")
print("Model saved to 'best_cnn_model.pth'")