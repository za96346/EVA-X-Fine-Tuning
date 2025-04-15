import torch
import json
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from eva_x import eva_x_tiny_patch16

# 假設您已有下列函式，可根據實際 import 路徑調整
# from timm.models.eva import eva_x_tiny_patch16

# 自定義參數
num_classes = 2
num_epochs = 10
batch_size = 32
k_folds = 5  # 設定幾折交叉驗證
allHistory = {}

eva_x_ti_pt = '/mnt/data/EVA-X/eva_x_tiny_patch16_merged520k_mim.pt'
datasetPath = "/mnt/data/TB_Chest_Radiography_Database"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("使用裝置:", device)

# 圖像預處理
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 建立完整資料集
dataset = datasets.ImageFolder(datasetPath, transform=transform)
dataset_size = len(dataset)
print("資料集大小:", dataset_size)

# 建立 KFold
kfold = KFold(n_splits=k_folds, shuffle=True)

########################################
# 交叉驗證流程
########################################
for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
    allHistory[fold] = {
        "accuracy": [],
        "loss": [],
        "val_accuracy": [],
        "val_loss": []
    }
    print(f"\n===== 開始第 {fold+1} / {k_folds} 折訓練 =====")

    # 1. 依據 train_idx, val_idx 建立資料子集
    train_sub = Subset(dataset, train_idx)
    val_sub   = Subset(dataset, val_idx)

    # 2. 建立 DataLoader
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_sub, batch_size=batch_size, shuffle=False)

    # 3. 建立與初始化模型
    eva_x_ti = eva_x_tiny_patch16(pretrained=eva_x_ti_pt)  
    eva_x_ti.head = nn.Linear(eva_x_ti.head.in_features, num_classes)
    eva_x_ti = eva_x_ti.to(device)

    # 4. 定義損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(eva_x_ti.parameters(), lr=1e-4)

    # 5. 訓練 & 驗證
    for epoch in range(num_epochs):
        ########################
        # Training
        ########################
        eva_x_ti.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = eva_x_ti(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = train_correct / train_total

        ########################
        # Validation
        ########################
        eva_x_ti.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = eva_x_ti(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total

        print(f"Fold [{fold+1}/{k_folds}] | Epoch [{epoch+1}/{num_epochs}] "
              f"| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Accuracy: {val_accuracy:.2f}%")
        
        allHistory[fold]["accuracy"].append(train_accuracy)
        allHistory[fold]["loss"].append(train_loss)
        allHistory[fold]["val_accuracy"].append(val_accuracy)
        allHistory[fold]["val_loss"].append(val_loss)

    # 6. 第 fold 折結束後，儲存模型
    save_path = f"./models/eva_x_ti_model_fold_{fold+1}.pth"
    torch.save(eva_x_ti.state_dict(), save_path)
    print(f"--> 第 {fold+1} 折的模型已儲存至: {save_path}")

    with open(f'./models/training_history.json', 'w') as json_file:
        json.dump(allHistory, json_file, indent=4)

print("所有折訓練完畢。")
