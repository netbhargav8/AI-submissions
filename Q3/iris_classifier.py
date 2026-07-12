# %% [markdown]
# #### **Importing all the libraries**
# %%
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
# %% [markdown]
# 
# #### **1. Dataset Loading & Feature Formatting**
# %%
print("Executing Step 1: Loading external Iris CSV dataset...")
# Load the CSV file into a variable named dataset
dataset = pd.read_csv('iris.csv')
dataset.columns = dataset.columns.str.strip()

print("--- Dataset Sample Preview (First 3 Rows) ---")
print(dataset.head(3), "\n")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
# %% [markdown]
# 
# #### **2. Encoding Categorical Target Labels**
# %%
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Store species names for our confusion matrix display
species_labels = label_encoder.classes_
# %% [markdown]
# 
# #### **3. Dataset Partitioning (Train/Test Split)**
# %%
print("Executing Step 2: Segmenting data into train and test sets...")
# 80% used to train the system, 20% reserved to evaluate it blindly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# %% [markdown]
# 
# #### **4. Model Training (Decision Tree Induction)**
# %%
print("Executing Step 3: Fitting Decision Tree Classifier model...")
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
# %% [markdown]
# 
# ####  **5. Model Evaluation & Diagnostics**
# %%
print("Executing Step 4: Assessing prediction metrics...")
# Generate blind predictions from test features
y_pred = classifier.predict(X_test)

# Calculate performance metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
# Print final diagnostic matrix results
print("\n" + "="*45)
print("       MACHINE LEARNING MODEL EVALUATION    ")
print("="*45)
print(f"Overall Model Classification Accuracy: {accuracy * 100:.2f}%\n")

print("--- CONFUSION MATRIX GRAPHICS ---")
print("                Predicted")
print(f"             {species_labels[0][:3]}  {species_labels[1][:3]}  {species_labels[2][:3]}")
print(f"Actual {species_labels[0][:3]}:  {conf_matrix[0][0]:>2}   {conf_matrix[0][1]:>2}   {conf_matrix[0][2]:>2}")
print(f"Actual {species_labels[1][:3]}:  {conf_matrix[1][0]:>2}   {conf_matrix[1][1]:>2}   {conf_matrix[1][2]:>2}")
print(f"Actual {species_labels[2][:3]}:  {conf_matrix[2][0]:>2}   {conf_matrix[2][1]:>2}   {conf_matrix[2][2]:>2}")
print("="*45 + "\n")
# %% [markdown]
# 