import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# --- 1. モデルの構造を再度定義する（train_cnn.pyと同じもの） ---
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
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 第2ブロック: Conv(64) x2 -> MaxPool
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第3ブロック: Conv(128) -> MaxPool
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # 全結合層
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.fc(x)
        return x

def main() -> int:
    # --- 2. 設定とテストデータの準備 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_PATH = 'best_cnn_model.pth'  # train_cnn.py が保存した重みファイル

    # テスト時はデータ拡張（ズラし）は入れず、正規化のみ行います
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # テストデータ（1万枚）のみを読み込む
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # --- 3. モデルの読み込み ---
    # FashionCNN_Pro でモデルを呼び出します
    model = FashionCNN_Pro().to(DEVICE)
    
    try:
        # 保存された学習済みのパラメータをセットする
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True))
        print(f"[OK] Loaded trained weights from '{WEIGHTS_PATH}'")
    except FileNotFoundError:
        print(f"[ERROR] Weights file not found: {WEIGHTS_PATH}")
        print("先に train_cnn.py を実行して学習を完了させてください。")
        return 1

    # --- 4. 最終テストの実行 ---
    model.eval()  # テストモードに切り替え（Dropoutなどをオフにする）
    correct = 0
    total = 0
    
    print("Running final evaluation on 10,000 test images...")
    
    with torch.no_grad():  # テスト時は学習しないので勾配計算をストップ（高速化）
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100.0 * correct / total
    
    print("\n" + "=" * 40)
    print(f"🎉 Final CNN Pro Test Accuracy: {acc:.2f}%")
    print("=" * 40)
    
    return 0

if __name__ == "__main__":
    raise SystemExit(main())