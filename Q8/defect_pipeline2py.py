# %% [markdown]
# #### **Importing all the required Libraries**
# %%
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
# %% [markdown]
# 
# #### **1. INGESTION, PREPROCESSING & AUGMENTATION**
# %%
print("Executing Step 1: Ingesting Kaggle 'Casting Product' data and applying Augmentation...")
data_dir = 'casting_512x512'
target_size = (64, 64)
raw_images = []
labels = []

categories = ['ok_front', 'def_front']

for category in categories:
    category_path = os.path.join(data_dir, category)
    if not os.path.exists(category_path):
        continue
        # Target Mapping: Defective (def_front) = 1, Perfect (ok_front) = 0
    target_val = 1 if category == 'def_front' else 0
    # We will limit ingestion to 500 images per class for speed during testing
    file_names = os.listdir(category_path)[:500]

    for file_name in file_names:
        img_path = os.path.join(category_path, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) # Grayscale is optimal for metal textures

        if img is None: continue

        # Standard Preprocessing: Resize
        resized_img = cv2.resize(img, target_size)

        # Original Image append
        raw_images.append(resized_img)
        labels.append(target_val)

        # AUGMENTATION: Horizontal Flip (Simulating metal components rotating on the conveyor)
        flipped_img = cv2.flip(resized_img, 1)
        raw_images.append(flipped_img)
        labels.append(target_val)
# %% [markdown]
# 
# #### **2. NORMALIZATION & MATRIX FLATTENING**
# %%
print("Executing Step 2: Normalizing tensors and flattening for ML induction...")
# Convert to float and scale between 0.0 and 1.0
X = np.array(raw_images).astype('float32') / 255.0
y = np.array(labels)

# Flatten 2D grayscale image arrays (64x64) into 1D arrays (4096 features)
X_flattened = X.reshape(X.shape[0], -1)
# %% [markdown]
# 
# #### **3. PARTITIONING & MODEL DUEL**
# %%
print("Executing Step 3: Stratifying Train/Test sets...")
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42, stratify=y)

print("Executing Step 4: Initiating Model Duel (Logistic Regression vs Random Forest)...")

# Model 1: Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

# Model 2: Random Forest
rand_forest = RandomForestClassifier(n_estimators=100, random_state=42)
rand_forest.fit(X_train, y_train)
y_pred_rf = rand_forest.predict(X_test)
# %% [markdown]
# 
# #### **4. DIAGNOSTICS & EVALUATION**
# %%
def get_metrics(y_true, y_pred):
    return [
        accuracy_score(y_true, y_pred) * 100,
        precision_score(y_true, y_pred) * 100,
        recall_score(y_true, y_pred) * 100
    ]

lr_metrics = get_metrics(y_test, y_pred_lr)
rf_metrics = get_metrics(y_test, y_pred_rf)

print("\n" + "="*55)
print("      QUALITY CONTROL: CLASSIFIER MODEL DUEL     ")
print("="*55)
print(f"Total Database Size (Augmented): {X.shape[0]} Images")
print(f"Flattened Feature Dimensions:    {X_flattened.shape[1]} Pixels per Image")
print("-" * 55)
print(f"METRIC       | LOGISTIC REGRESSION | RANDOM FOREST   ")
print(f"-------------------------------------------------------")
print(f"Accuracy     | {lr_metrics[0]:.2f}%               | {rf_metrics[0]:.2f}%")
print(f"Precision    | {lr_metrics[1]:.2f}%               | {rf_metrics[1]:.2f}%")
print(f"Recall       | {lr_metrics[2]:.2f}%               | {rf_metrics[2]:.2f}%")
print("="*55 + "\n")