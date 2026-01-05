import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, matthews_corrcoef
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
import copy
import os
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
# NOTE: Replace with your actual data path if running outside a context that handles it
DATA_ROOT_DIR = '/kaggle/input/sorowardi-aug/sorowardi_aug_2' 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 15
K_FOLDS = 5
NUM_CLASSES = 3
SEED = 42

# Define a path for saving the final model
MODEL_SAVE_PATH = 'final_best_attention_ecg_model.pth'

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True # set to True for performance, False for strict determinism

seed_everything(SEED)

# ==========================================
# 2. DATA PIPELINE
# ==========================================
class AugmentedDataset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        # The subset is a torch.utils.data.Subset, which returns (data, target) from the original dataset
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

train_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Data or generate Dummy Data
try:
    full_dataset = datasets.ImageFolder(root=DATA_ROOT_DIR)
    class_names = full_dataset.classes
    targets = full_dataset.targets
except:
    print("Dataset not found. Generating Dummy Data for code verification.")
    full_dataset = datasets.FakeData(size=100, image_size=(3, 300, 300), num_classes=3)
    class_names = ['Abnormal', 'MI', 'Normal']
    # Extract dummy targets
    targets = [y for _, y in full_dataset]


# ==========================================
# 3. ATTENTION MODEL (CBAM)
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
        # Load EfficientNet-B3 with pretrained weights
        weights = models.EfficientNet_B3_Weights.DEFAULT
        self.backbone = models.efficientnet_b3(weights=weights)
        
        # Get feature dimension before the classifier
        self.feature_dim = self.backbone.classifier[1].in_features 
        self.features = self.backbone.features
        self.avgpool = self.backbone.avgpool
        
        # Insert CBAM after the feature extractor and before the pooling layer
        self.attention = CBAM(self.feature_dim)
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(self.feature_dim, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # Apply CBAM
        x = self.attention(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# ==========================================
# 4. K-FOLD TRAINING ENGINE (MODIFIED)
# ==========================================
def train_one_fold(fold_index, train_idx, val_idx, full_dataset):
    
    print(f"\n{'='*20} FOLD {fold_index+1}/{K_FOLDS} {'='*20}")
    
    # Create Subsets and apply Transforms
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    train_ds = AugmentedDataset(train_subset, transform=train_transform)
    val_ds = AugmentedDataset(val_subset, transform=val_transform)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # Init Model
    model = AttentionECGModel(num_classes=NUM_CLASSES).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=3)
    
    # --- History Tracking ---
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())
    
    for epoch in range(EPOCHS):
        # Training Phase
        model.train()
        train_loss, correct, total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"Fold {fold_index+1} Ep {epoch+1}", leave=False)
        
        for imgs, labels in loop:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
            loop.set_postfix(loss=loss.item())
        
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                val_loss += criterion(outputs, labels).item()
                _, pred = outputs.max(1)
                val_correct += pred.eq(labels).sum().item()
                val_total += labels.size(0)
        
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        scheduler.step(val_acc)
        
        # Update history
        history['train_loss'].append(train_loss_avg)
        history['val_loss'].append(val_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss_avg:.4f} | Val Loss: {val_loss_avg:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            # Save best fold weights (Optional: can be used for ensembling)
            torch.save(best_weights, f"best_fold_{fold_index}.pth")
            
    print(f"Fold {fold_index+1} Best Val Acc: {best_acc:.2f}%")
    
    # Reload best weights to generate predictions for this fold
    model.load_state_dict(best_weights)
    model.eval()
    
    # Collect predictions for global metrics on the validation set
    fold_preds = []
    fold_targets = []
    fold_probs = []
    
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = F.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            fold_preds.extend(preds.cpu().numpy())
            fold_targets.extend(labels.cpu().numpy())
            fold_probs.extend(probs.cpu().numpy())
            
    return fold_targets, fold_preds, fold_probs, best_acc, history, best_weights

# ==========================================
# 5. RUNNING K-FOLD
# ==========================================

# Prepare Global containers
global_y_true = []
global_y_pred = []
global_y_probs = []
fold_accuracies = []
global_history = []
# Best weights across all folds (for the final model save)
overall_best_acc = 0.0
overall_best_weights = None

# Stratified K-Fold
skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)
X_dummy = np.zeros(len(full_dataset)) 

for fold_i, (train_idx, val_idx) in enumerate(skf.split(X_dummy, targets)):
    y_true, y_pred, y_prob, acc, history, best_weights = train_one_fold(fold_i, train_idx, val_idx, full_dataset)
    
    # Accumulate results
    global_y_true.extend(y_true)
    global_y_pred.extend(y_pred)
    global_y_probs.extend(y_prob)
    fold_accuracies.append(acc)
    global_history.append(history)
    
    # Track the single best model across all folds
    if acc > overall_best_acc:
        overall_best_acc = acc
        overall_best_weights = best_weights

# Save the single best model's weights across all K folds
if overall_best_weights:
    torch.save(overall_best_weights, MODEL_SAVE_PATH)
    print(f"\n✅ Overall best model weights saved to: **{MODEL_SAVE_PATH}** (Best Fold Acc: {overall_best_acc:.2f}%)")

# ==========================================
# 6. FINAL EVALUATION (Aggregated)
# ==========================================

def plot_curves(global_history, metric_name, title):
    """Plots the average training and validation curves."""
    plt.figure(figsize=(10, 6))
    
    train_history = [h[f'train_{metric_name}'] for h in global_history]
    val_history = [h[f'val_{metric_name}'] for h in global_history]
    
    # Convert to numpy arrays and average across folds
    train_mean = np.mean(train_history, axis=0)
    train_std = np.std(train_history, axis=0)
    val_mean = np.mean(val_history, axis=0)
    val_std = np.std(val_history, axis=0)
    
    epochs_range = range(1, EPOCHS + 1)
    
    plt.plot(epochs_range, train_mean, label=f'Average Training {metric_name.title()}')
    plt.fill_between(epochs_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
    
    plt.plot(epochs_range, val_mean, label=f'Average Validation {metric_name.title()}')
    plt.fill_between(epochs_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
    
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel(metric_name.title())
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_specificity(cm):
    """Calculates class-wise and average specificity from a confusion matrix."""
    num_classes = cm.shape[0]
    specificities = []
    
    for i in range(num_classes):
        # True Negatives (TN): Sum of all cells except the i-th row and i-th column
        tn_sum = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
        # False Positives (FP): Sum of the i-th column, excluding the main diagonal (i.e., cm[i,i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        
        specificity = tn_sum / (tn_sum + fp) if (tn_sum + fp) > 0 else 0
        specificities.append(specificity)
        
    avg_specificity = np.mean(specificities)
    return specificities, avg_specificity

print(f"\n{'='*20} FINAL RESULTS {'='*20}")
print(f"Accuracies per fold: {[f'{x:.2f}%' for x in fold_accuracies]}")
print(f"Average Accuracy: {np.mean(fold_accuracies):.2f}% (+/- {np.std(fold_accuracies):.2f})")

# --- 1. Accuracy vs. Loss Curves (Requested Change 1 & 3) ---
# 
plot_curves(global_history, 'acc', 'Average Accuracy vs. Epochs (K-Fold CV)')
# 
plot_curves(global_history, 'loss', 'Average Loss vs. Epochs (K-Fold CV)')

# --- 2. Confusion Matrix (Requested Change 3) ---
cm = confusion_matrix(global_y_true, global_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f'Global Confusion Matrix ({K_FOLDS}-Fold CV)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --- 3. Classification Report & Specificity (Requested Change 2) ---
print("\nGlobal Classification Report:\n")
print(classification_report(global_y_true, global_y_pred, target_names=class_names))

# Calculate Specificity
specificities, avg_specificity = calculate_specificity(cm)
print("\n--- Specificity Metrics ---")
for name, spec in zip(class_names, specificities):
    print(f"Specificity ({name}): {spec:.4f}")
print(f"Average Specificity: {avg_specificity:.4f}")

# Calculate MCC
mcc = matthews_corrcoef(global_y_true, global_y_pred)
print(f"Matthews Correlation Coefficient (MCC): {mcc:.4f}")

# --- 4. ROC Curve (Requested Change 3) ---
global_y_true_bin = label_binarize(global_y_true, classes=range(len(class_names)))
global_y_probs = np.array(global_y_probs)

fpr = dict()
tpr = dict()
roc_auc = dict()

plt.figure(figsize=(10, 8))
# 
for i in range(len(class_names)):
    fpr[i], tpr[i], _ = roc_curve(global_y_true_bin[:, i], global_y_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_names[i]} (AUC = {roc_auc[i]:.4f})') # Changed format to .4f

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Global Multiclass ROC Curve ({K_FOLDS}-Fold)')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# --- 5. Model Saving (Requested Change 4) ---
# The best model across all folds has been saved to: MODEL_SAVE_PATH

# Final verification step: Load the saved model to confirm it works
try:
    final_model = AttentionECGModel(num_classes=NUM_CLASSES)
    final_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    final_model.eval()
    print(f"\nVerification: Model loaded successfully from {MODEL_SAVE_PATH} and ready for inference/Grad-CAM.")
except Exception as e:
    print(f"\nError verifying model load: {e}")

