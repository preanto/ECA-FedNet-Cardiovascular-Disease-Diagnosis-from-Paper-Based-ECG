import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import copy
import os
import random
import networkx as nx

# ==========================================
# 1. CONFIGURATION & FL HYPERPARAMETERS
# ==========================================
DATA_ROOT_DIR = '/kaggle/input/sorowardi-aug/sorowardi_aug_2'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Federated Learning Settings
NUM_CLIENTS = 4          # Number of hospitals/devices
COMMUNICATION_ROUNDS = 10 # Number of times server updates global model
LOCAL_EPOCHS = 2         # Epochs per client per round (Low to simulate frequent updates)
BATCH_SIZE = 16
LR = 3e-4
NUM_CLASSES = 3
SEED = 42

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(SEED)

# ==========================================
# 2. VISUALIZATION ENGINE (PAPER WORTHY)
# ==========================================
def plot_fl_topology(num_clients):
    """Generates a Q1-style schematic of the Federated Learning System."""
    G = nx.Graph()
    server_node = "Global\nServer"
    G.add_node(server_node)
    
    pos = {server_node: (0, 0)}
    colors = ['#FF6B6B']
    sizes = [3000]
    
    # Create satellite clients
    rad = 1.5
    for i in range(num_clients):
        client_name = f"Client {i+1}\n(Hospital {i+1})"
        angle = (2 * np.pi * i) / num_clients
        x = rad * np.cos(angle)
        y = rad * np.sin(angle)
        
        G.add_node(client_name)
        G.add_edge(server_node, client_name)
        pos[client_name] = (x, y)
        colors.append('#4ECDC4')
        sizes.append(2000)

    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    # Draw Edges
    for edge in G.edges():
        x_vals = [pos[edge[0]][0], pos[edge[1]][0]]
        y_vals = [pos[edge[0]][1], pos[edge[1]][1]]
        ax.plot(x_vals, y_vals, color='gray', linestyle='--', alpha=0.5, zorder=1)
        # Add aggregation arrows
        mid_x = np.mean(x_vals)
        mid_y = np.mean(y_vals)
        ax.text(mid_x, mid_y, "Avg Weights\n⇄\nParams", ha='center', va='center', fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    # Draw Nodes
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors='black', linewidths=1.5, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', font_color='black', ax=ax)
    
    plt.title("Federated Learning Simulation Topology\n(Multi-Split Architecture)", fontsize=15, fontweight='bold')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# ==========================================
# 3. DATA PIPELINE & PARTITIONING
# ==========================================
class AugmentedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

train_transform = transforms.Compose([
    transforms.Resize((256, 256)), # Slightly smaller for simulation speed
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data
try:
    full_dataset = datasets.ImageFolder(root=DATA_ROOT_DIR)
    class_names = full_dataset.classes
except:
    print("Dataset not found. Generating Dummy Data.")
    full_dataset = datasets.FakeData(size=300, image_size=(3, 256, 256), num_classes=3)
    class_names = ['Abnormal', 'MI', 'Normal']

# --- FEDERATED PARTITIONING STRATEGY ---
total_len = len(full_dataset)
test_len = int(0.2 * total_len) # 20% Global Test Set
train_len = total_len - test_len

# 1. Create Global Train/Test Split
global_train_subset, global_test_subset = random_split(
    full_dataset, [train_len, test_len], generator=torch.Generator().manual_seed(SEED)
)

# 2. Split Global Train into N Clients (IID Partitioning for simplicity, can be Non-IID)
split_sizes = [len(global_train_subset) // NUM_CLIENTS] * NUM_CLIENTS
# Handle remainder
split_sizes[-1] += len(global_train_subset) - sum(split_sizes)

client_subsets = random_split(global_train_subset, split_sizes, generator=torch.Generator().manual_seed(SEED))

# 3. Create Loaders
client_loaders = []
for subset in client_subsets:
    ds = AugmentedDataset(subset, transform=train_transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)
    client_loaders.append(loader)

global_test_ds = AugmentedDataset(global_test_subset, transform=val_transform)
global_test_loader = DataLoader(global_test_ds, batch_size=BATCH_SIZE, shuffle=False)

print(f"Data Partitioning Complete:")
print(f"Global Test Set: {len(global_test_subset)} images")
for i, l in enumerate(client_loaders):
    print(f"Client {i+1}: {len(l.dataset)} images")

# ==========================================
# 4. MODEL (EfficientNet + CBAM)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x))

class CBAM(nn.Module):
    def __init__(self, planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class AttentionECGModel(nn.Module):
    def __init__(self, num_classes=3):
        super(AttentionECGModel, self).__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.backbone = models.efficientnet_b0(weights=weights)
        self.feature_dim = self.backbone.classifier[1].in_features 
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        self.attention = CBAM(self.feature_dim)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 5. FEDERATED AVERAGING (The Engine)
# ==========================================
def local_train(model, loader, epochs):
    """Trains a client model locally."""
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    
    for epoch in range(epochs):
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    return model.state_dict()

def federated_averaging(global_weights, local_weights_list, client_sizes):
    """
    Aggregation Algorithm:
    w_global = sum(w_client * (n_client / n_total))
    """
    total_samples = sum(client_sizes)
    
    # Create a copy of the first client's weights to hold the average
    avg_weights = copy.deepcopy(local_weights_list[0])
    
    for key in avg_weights.keys():
        # Initialize with weighted first client
        weight_sum = local_weights_list[0][key] * (client_sizes[0] / total_samples)
        
        # Add remaining clients
        for i in range(1, len(local_weights_list)):
            weight_sum += local_weights_list[i][key] * (client_sizes[i] / total_samples)
            
        avg_weights[key] = weight_sum
        
    return avg_weights

def evaluate_global(model, loader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    acc = 100 * correct / total
    return acc, all_targets, all_preds, all_probs

# ==========================================
# 6. RUN SIMULATION
# ==========================================
print("\nInitializing Federated Learning Simulation...")
plot_fl_topology(NUM_CLIENTS)

# Initialize Global Server Model
global_model = AttentionECGModel(num_classes=NUM_CLASSES).to(DEVICE)
global_weights = global_model.state_dict()

# History tracking
history = {'round': [], 'acc': []}

for comm_round in range(COMMUNICATION_ROUNDS):
    print(f"\n--- Communication Round {comm_round+1}/{COMMUNICATION_ROUNDS} ---")
    
    local_weights_list = []
    client_sizes = []
    
    # 1. CLIENT UPDATE STEP
    for i in range(NUM_CLIENTS):
        # Create local model copy
        local_model = AttentionECGModel(num_classes=NUM_CLASSES).to(DEVICE)
        # Load global weights
        local_model.load_state_dict(global_weights)
        
        # Train locally
        # print(f"  Training Client {i+1}...")
        updated_weights = local_train(local_model, client_loaders[i], LOCAL_EPOCHS)
        
        local_weights_list.append(updated_weights)
        client_sizes.append(len(client_loaders[i].dataset))
        
        # Free memory
        del local_model
        torch.cuda.empty_cache()

    # 2. SERVER AGGREGATION STEP
    print("  Aggregating weights at Global Server...")
    global_weights = federated_averaging(global_weights, local_weights_list, client_sizes)
    
    # Update Global Model
    global_model.load_state_dict(global_weights)
    
    # 3. GLOBAL EVALUATION STEP
    val_acc, _, _, _ = evaluate_global(global_model, global_test_loader)
    history['round'].append(comm_round+1)
    history['acc'].append(val_acc)
    
    print(f"  Global Model Test Accuracy: {val_acc:.2f}%")

# ==========================================
# 7. FINAL METRICS & VISUALIZATION
# ==========================================

# 1. Accuracy Curve
plt.figure(figsize=(10, 5))
plt.plot(history['round'], history['acc'], marker='o', linestyle='-', color='b', linewidth=2)
plt.title(f'Global Model Convergence (Acc vs Rounds)', fontsize=14)
plt.xlabel('Communication Rounds')
plt.ylabel('Test Accuracy (%)')
plt.grid(True, alpha=0.3)
plt.show()

# Final Evaluation
print("\nGenerating Final Evaluation Reports...")
final_acc, y_true, y_pred, y_probs = evaluate_global(global_model, global_test_loader)

# 2. Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Final Global Model Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# 3. Classification Report
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# 4. ROC Curve
y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
y_probs = np.array(y_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Global Model ROC Curve')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.show()