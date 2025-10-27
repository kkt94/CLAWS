import argparse
import json
import numpy as np
import os
import csv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_curve, auc, f1_score, classification_report
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical 


def get_roc_scores(scores: np.array, labels: np.array):
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.05)[0][-1]]
    return arc, acc, low

def get_roc_auc_scores(scores: np.array, labels: np.array):
    fpr, tpr, _ = roc_curve(labels, scores)
    arc = auc(fpr, tpr)
    acc = np.max(1 - (fpr + (1 - tpr)) / 2)
    low = tpr[np.where(fpr < 0.05)[0][-1]]
    return arc, acc, low, fpr, tpr


class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, num_classes)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) 
        return x
    

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser(description="Run Step 4. Evaluation Strategy - MLP")
    parser.add_argument("--seed", type=int, default=42, help="randoom seed setting")
    parser.add_argument("--model_name", type=str, default="Deepseek-math-7b-rl", help="The model was used in the experiment and will be evaluated.")
    parser.add_argument("--output_dir", type=str, default="output", help="evaluation output dir")
    parser.add_argument("--dataset_name", type=str, default="TEST")

    args = parser.parse_args()

    set_seed(args)

    output_dir = os.path.join(args.output_dir, args.dataset_name, "evaluation", "MLP")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{args.model_name}.csv')

    with open(output_path, 'a+', encoding='utf-8', newline='') as csvf:
        wr = csv.writer(csvf)
            
        train_path = os.path.join(args.output_dir, "REF", "labeling", f"{args.model_name}_labeled.json")
        with open(train_path, 'r') as f:
            train_data = json.load(f)
            # print("Train sample count: ", len(train_data))

        test_path = os.path.join(args.output_dir, args.dataset_name, "labeling", f"{args.model_name}_labeled.json")
        if not os.path.isfile(test_path):
            return
        with open(test_path, 'r') as f:
            test_data = json.load(f)
            # print("Test sample count: ", len(test_data))
        

        X_train_list = []
        y_train_list = []

        for item in train_data:
            X_train_list.append(item['CLAWS'])
            y_train_list.append(item['label'])

        X_train = np.array(X_train_list)
        y_train = np.array(y_train_list)

        unique_labels, counts = np.unique(y_train, return_counts=True)
        for label, count in zip(unique_labels, counts):
            print(f"Label: {label}, Count: {count}")

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        num_classes = len(np.unique(y_train_encoded))
        y_train_categorical = to_categorical(y_train_encoded, num_classes=num_classes)
        class_weights = compute_class_weight(class_weight='balanced', 
                                            classes=np.unique(y_train_encoded), 
                                            y=y_train_encoded)
        # class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
        # print("Computed class weights:", class_weight_dict)
        
        X_test_list = []
        y_test_list = []

        for item in test_data:
            X_test_list.append(item['CLAWS'])
            y_test_list.append(item['label'])

        X_test = np.array(X_test_list)
        y_test = np.array(y_test_list)

        y_test_encoded = le.transform(y_test)
        y_test_categorical = to_categorical(y_test_encoded, num_classes=num_classes)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.long)

        batch_size = 8
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


        input_dim = X_train.shape[1]
        model = MLP(input_dim=input_dim, num_classes=num_classes)
        print("Model parameter: ", sum(p.numel() for p in model.parameters() if p.requires_grad))


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)


        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        num_epochs = 10
        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            epoch_loss = running_loss / total
            epoch_acc = correct / total
            history['loss'].append(epoch_loss)
            history['accuracy'].append(epoch_acc)
            

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item() * inputs.size(0)
                    
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_epoch_loss = val_loss / val_total
            val_epoch_acc = val_correct / val_total
            history['val_loss'].append(val_epoch_loss)
            history['val_accuracy'].append(val_epoch_acc)
            
            # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")


        model.eval()
        with torch.no_grad():
            outputs = model(X_test_tensor.to(device))
            test_loss = criterion(outputs, y_test_tensor.to(device)).item()
            probs = F.softmax(outputs, dim=1).detach().cpu().numpy()
            _, predicted_labels = torch.max(outputs, 1)
            predicted_labels = predicted_labels.detach().cpu().numpy()

        # print("Test Loss:", test_loss)
        # print("Test Accuracy:", np.mean(predicted_labels == y_test_encoded))


        f1_weighted = f1_score(y_test_encoded, predicted_labels, average='weighted')
        f1_macro = f1_score(y_test_encoded, predicted_labels, average='macro')
        f1_micro = f1_score(y_test_encoded, predicted_labels, average='micro')

        print(f"\n{args.dataset_name} {args.model_name} {args.method}")
        wr.writerow([args.dataset_name, args.model_name, args.method])
        print("F1 Score (weighted):", f1_weighted)
        wr.writerow(["F1 Score (weighted):", f1_weighted])
        print("F1 Score (macro):", f1_macro)
        wr.writerow(["F1 Score (macro):", f1_macro])
        print("F1 Score (micro):", f1_micro)
        wr.writerow(["F1 Score (micro):", f1_micro])

        auprc_list = []
        for i in range(num_classes):
            ap = average_precision_score(y_test_categorical[:, i], probs[:, i])
            auprc_list.append(ap)
        auprc_macro = np.mean(auprc_list)
        print(f"Multi-class Average Precision (macro-average): {auprc_macro:.4f}")
        wr.writerow(["Multi-class Average Precision (macro-average):", auprc_macro])

        roc_auc_list = []
        roc_auc_dict = {}
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_test_categorical[:, i], probs[:, i])
            roc_auc_val = auc(fpr, tpr)
            roc_auc_list.append(roc_auc_val)
            class_name = le.classes_[i]
            roc_auc_dict[class_name] = roc_auc_val
            # print(f"Class '{class_name}': ROC AUC = {roc_auc_val:.4f}")

        roc_auc_overall = np.mean(roc_auc_list)
        print(f"Multi-class ROC AUC (macro-average): {roc_auc_overall:.4f}")
        wr.writerow(["Multi-class ROC AUC (macro-average):", roc_auc_overall])

        print("\nClassification Report:")
        print(le.classes_)
        print(classification_report(y_test_encoded, predicted_labels, target_names=le.classes_, zero_division=0))
        wr.writerow([classification_report(y_test_encoded, predicted_labels, target_names=le.classes_, zero_division=0)])

    
if __name__ == "__main__":
    main()