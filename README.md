# Binary-Classification-with-Neural-Networks-on-the-Census-Income-Dataset

## Name: DHARSHAN D
## Reg no: 212223230045
## Overview
This project builds a binary classification model using PyTorch to predict whether an individual earns more than $50,000 annually based on census data. The model uses embeddings for categorical features and batch-normalized continuous features.  

- **Dataset:** Census Income Dataset (income.csv)  
- **Input Features:** Categorical and continuous features such as Workclass, Education, Marital Status, Age, Hours per Week, etc.  
- **Output:** Binary label (<=50K or >50K)  
- **Model:** Single hidden layer neural network with embeddings for categorical features and batch normalization for continuous features.  
- **Training:** 300 epochs with Adam optimizer and CrossEntropyLoss.

---

### Program

```

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import random

# Step 1: Load data
data_path = "income.csv"
df = pd.read_csv(data_path)

# Step 2: Identify categorical, continuous, and label columns
categorical_columns = [
    "Workclass", "Education", "Marital Status", "Occupation",
    "Relationship", "Race", "Gender", "Native Country"
]
continuous_columns = [
    "Age", "Final Weight", "EducationNum", "Capital Gain",
    "capital loss", "Hours per Week"
]
label_column = "Income"

# Step 3: Data preprocessing
for col in categorical_columns:
    df[col] = df[col].str.strip()
    df[col] = df[col].replace('?', np.nan)
df = df.dropna(subset=categorical_columns + continuous_columns + [label_column])

df[label_column] = df[label_column].apply(lambda x: 1 if '>50K' in x else 0)

label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = StandardScaler()
df[continuous_columns] = scaler.fit_transform(df[continuous_columns])

cat_data = df[categorical_columns].values.astype(np.int64)
cont_data = df[continuous_columns].values.astype(np.float32)
labels = df[label_column].values.astype(np.int64)

cat_tensor = torch.tensor(cat_data)
cont_tensor = torch.tensor(cont_data)
label_tensor = torch.tensor(labels)

# Step 4: Split dataset 80/20 and use all rows
total_samples = len(df)
train_size = int(0.8 * total_samples)
test_size = total_samples - train_size

dataset = TensorDataset(cat_tensor, cont_tensor, label_tensor)
train_dataset, test_dataset = random_split(dataset, [train_size, test_size],
                                           generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, drop_last=False)

# Step 5: Define model
class TabularModel(nn.Module):
    def __init__(self, emb_dims, no_of_cont):
        super().__init__()
        self.emb_layers = nn.ModuleList([nn.Embedding(x, y) for x, y in emb_dims])
        self.emb_dropout = nn.Dropout(0.4)
        self.batch_norm_cont = nn.BatchNorm1d(no_of_cont)

        n_emb = sum([y for _, y in emb_dims])
        n_cont = no_of_cont

        self.layer1 = nn.Linear(n_emb + n_cont, 50)
        self.layer1_bn = nn.BatchNorm1d(50)
        self.dropout = nn.Dropout(0.4)
        self.output = nn.Linear(50, 2)

    def forward(self, x_cat, x_cont):
        embeddings = [emb_layer(x_cat[:, i]) for i, emb_layer in enumerate(self.emb_layers)]
        x_emb = torch.cat(embeddings, 1)
        x_emb = self.emb_dropout(x_emb)
        x_cont = self.batch_norm_cont(x_cont)
        x = torch.cat([x_emb, x_cont], 1)
        x = F.relu(self.layer1_bn(self.layer1(x)))
        x = self.dropout(x)
        x = self.output(x)
        return x

# Step 6: Prepare embeddings
emb_dims = []
for col in categorical_columns:
    num_unique = len(label_encoders[col].classes_)
    emb_dim = min(50, (num_unique + 1) // 2)
    emb_dims.append((num_unique, emb_dim))

model = TabularModel(emb_dims, len(continuous_columns))

# Step 7: Set seeds
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Step 8: Train
epochs = 300
model.train()

for epoch in range(epochs):
    epoch_loss = 0
    for x_cat_batch, x_cont_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(x_cat_batch, x_cont_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    if (epoch + 1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")

# Step 9: Evaluate
model.eval()
all_preds, all_labels = [], []
test_loss = 0
with torch.no_grad():
    for x_cat_batch, x_cont_batch, y_batch in test_loader:
        outputs = model(x_cat_batch, x_cont_batch)
        loss = criterion(outputs, y_batch)
        test_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.append(preds)
        all_labels.append(y_batch)

avg_test_loss = test_loss / len(test_loader)
all_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
accuracy = accuracy_score(all_labels.cpu(), all_preds.cpu())

print(f"Test Loss: {avg_test_loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Prediction function (optional)
def predict_income(model, input_dict, label_encoders, scaler, categorical_columns, continuous_columns):
    model.eval()
    processed_cat, processed_cont = [], []
    for col in categorical_columns:
        val = input_dict.get(col, "")
        if val in label_encoders[col].classes_:
            processed_cat.append(label_encoders[col].transform([val])[0])
        else:
            processed_cat.append(0)
    for col in continuous_columns:
        processed_cont.append(input_dict.get(col, 0))
    processed_cont = scaler.transform([processed_cont])
    cat_tensor = torch.tensor([processed_cat], dtype=torch.int64)
    cont_tensor = torch.tensor(processed_cont, dtype=torch.float32)
    with torch.no_grad():
        output = model(cat_tensor, cont_tensor)
        pred = torch.softmax(output, dim=1)
        pred_label = torch.argmax(pred, dim=1).item()
        prob = pred[0][pred_label].item()
    return pred_label, prob


```


### Output:

<img width="378" height="185" alt="image" src="https://github.com/user-attachments/assets/4093461e-11be-47de-8a6a-d7143b36c852" />


### Result: 
Hence the program is completed successfully.
