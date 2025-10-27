import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, balanced_accuracy_score, matthews_corrcoef, roc_curve, auc, average_precision_score
import argparse
import json
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
import csv

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


class CombinedModel(nn.Module):
    def __init__(self, input_dim, latent_dim, num_classes):
        super(CombinedModel, self).__init__()

        self.encoder_fc1 = nn.Linear(input_dim, 16)
        self.encoder_fc2 = nn.Linear(16, 8)
        self.encoder_fc3 = nn.Linear(8, latent_dim)

        combined_input_dim = latent_dim + input_dim
        self.fc_combined = nn.Linear(combined_input_dim, 16)
        self.fc_out = nn.Linear(16, num_classes)
        
    def forward(self, x):
        h = F.relu(self.encoder_fc1(x))
        h = F.relu(self.encoder_fc2(h))
        z = self.encoder_fc3(h)  # (batch_size, latent_dim)

        combined_features = torch.cat([z, x], dim=1)  # (batch_size, latent_dim + input_dim)

        # --- Classification head ---
        h_combined = F.relu(self.fc_combined(combined_features))
        logits = self.fc_out(h_combined)
        
        return logits, z

def pairwise_distance(embeddings):
    # embeddings: (batch_size, d)
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = torch.unsqueeze(square_norm, 1) - 2 * dot_product + torch.unsqueeze(square_norm, 0)
    distances = torch.sqrt(torch.clamp(distances, min=1e-16))
    return distances

def batch_all_triplet_loss(labels, embeddings, margin=1.0):
    """
    labels: (batch_size,) tensor of int64
    embeddings: (batch_size, d) tensor
    margin: margin for triplet loss
    """
    batch_size = embeddings.size(0)
    # pairwise distances (batch_size, batch_size)
    pdist = pairwise_distance(embeddings)
    
    # Create masks for positive and negative pairs
    labels = labels.unsqueeze(1)  # (batch_size, 1)
    mask_positive = (labels == labels.t())  # (batch_size, batch_size)
    mask_negative = (labels != labels.t())
    
    # Exclude self-comparisons in positive mask
    diag = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    mask_positive = mask_positive & (~diag)
    
    # Compute triplet loss for every (anchor, positive, negative) triplet
    # Expand dims: anchor-positive distances: (batch_size, batch_size, 1)
    anchor_positive_dist = pdist.unsqueeze(2)
    # Expand negative distances: (batch_size, 1, batch_size)
    anchor_negative_dist = pdist.unsqueeze(1)
    
    triplet_loss_tensor = anchor_positive_dist - anchor_negative_dist + margin  # (batch_size, batch_size, batch_size)
    
    # Apply masks: valid if (anchor, positive) is positive pair and (anchor, negative) is negative pair
    mask_positive = mask_positive.unsqueeze(2).float()  # (batch_size, batch_size, 1)
    mask_negative = mask_negative.unsqueeze(1).float()  # (batch_size, 1, batch_size)
    valid_mask = mask_positive * mask_negative  # (batch_size, batch_size, batch_size)
    
    triplet_loss_tensor = valid_mask * triplet_loss_tensor
    triplet_loss_tensor = F.relu(triplet_loss_tensor)
    
    # Average over number of positive triplets
    valid_triplets = (triplet_loss_tensor > 1e-16).float()
    num_positive_triplets = torch.sum(valid_triplets)
    triplet_loss = torch.sum(triplet_loss_tensor) / (num_positive_triplets + 1e-16)
    
    return triplet_loss


def compute_prototypes(model, dataloader, device):
    model.eval()
    latent_list = []
    label_list = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            _, z = model(x)
            latent_list.append(z.cpu())
            label_list.append(y.cpu())
    latent_all = torch.cat(latent_list, dim=0)
    labels_all = torch.cat(label_list, dim=0)
    
    prototypes = {}
    unique_labels = torch.unique(labels_all)
    for label in unique_labels:
        mask = (labels_all == label)
        prototypes[label.item()] = latent_all[mask].mean(dim=0)
    return prototypes

def predict_with_prototypes(model, dataloader, prototypes, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch
            x = x.to(device)
            _, z = model(x)
            z = z.cpu()
            for latent in z:
                distances = []
                
                for label, proto in prototypes.items():
                    distance = torch.norm(latent - proto, p=2)
                    distances.append((label, distance.item()))
                
                pred_label = min(distances, key=lambda tup: tup[1])[0]
                predictions.append(pred_label)
            true_labels.extend(y.cpu().numpy())
    return np.array(true_labels), np.array(predictions)


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser(description="Run Step 4. Evaluation Strategy - Prototype")
    parser.add_argument("--seed", type=int, default=42, help="randoom seed setting")
    parser.add_argument("--model_name", type=str, default="Deepseek-math-7b-rl", help="The model was used in the experiment and will be evaluated.")
    parser.add_argument("--output_dir", type=str, default="output", help="evaluation output dir")
    parser.add_argument("--dataset_name", type=str, default="TEST")
    
    args = parser.parse_args()

    set_seed(args)
    
    output_dir = os.path.join(args.output_dir, args.dataset_name, "evaluation", "Prototype")
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

        X_train = np.array(X_train_list, dtype=np.float32)
        y_train = np.array(y_train_list)

        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)
        num_classes = len(np.unique(y_train_encoded))
        print(f"number of classes: {num_classes}")


        X_test_list = []
        y_test_list = []
        for item in test_data:
            X_test_list.append(item['CLAWS'])
            y_test_list.append(item['label'])
        
        X_test = np.array(X_test_list, dtype=np.float32)
        y_test = np.array(y_test_list)
        y_test_encoded = le.transform(y_test)


        train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train_encoded, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test_encoded, dtype=torch.long))

        batch_size = 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        input_dim = 5         
        latent_dim = 2 
        model = CombinedModel(input_dim=input_dim, latent_dim=latent_dim, num_classes=num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        learning_rate = 1e-3
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        margin = 1.0          
        lambda_triplet = 0.1  

        epochs = 5

        for epoch in range(epochs):
            model.train()
            epoch_cls_loss = 0.0
            epoch_triplet_loss = 0.0
            epoch_total_loss = 0.0
            steps = 0
            
            for batch in train_loader:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                optimizer.zero_grad()
                logits, embeddings = model(x_batch)
                
                cls_loss = criterion(logits, y_batch)
                t_loss = batch_all_triplet_loss(y_batch, embeddings, margin=margin)
                total_loss = cls_loss + lambda_triplet * t_loss
                
                total_loss.backward()
                optimizer.step()
                
                epoch_cls_loss += cls_loss.item()
                epoch_triplet_loss += t_loss.item()
                epoch_total_loss += total_loss.item()
                steps += 1
            
            # avg_cls_loss = epoch_cls_loss / steps
            # avg_triplet_loss = epoch_triplet_loss / steps
            # avg_total_loss = epoch_total_loss / steps
            # print(f"Epoch {epoch+1}/{epochs}: cls_loss={avg_cls_loss:.4f}, triplet_loss={avg_triplet_loss:.4f}, total_loss={avg_total_loss:.4f}")
            
            model.eval()
            total_correct = 0
            total_samples = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch in test_loader:
                    x_batch, y_batch = batch
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    logits, _ = model(x_batch)
                    loss = criterion(logits, y_batch)
                    test_loss += loss.item() * x_batch.size(0)
                    preds = torch.argmax(logits, dim=1)
                    total_correct += (preds == y_batch).sum().item()
                    total_samples += x_batch.size(0)
            # avg_test_loss = test_loss / total_samples
            # test_acc = total_correct / total_samples
            # print(f"  Test: loss={avg_test_loss:.4f}, accuracy={test_acc:.4f}\n")


        model.eval()
        all_preds = []
        all_probs = []
        all_y_true = []
        with torch.no_grad():
            for batch in test_loader:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                logits, _ = model(x_batch)
                probs = F.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_y_true.extend(y_batch.cpu().numpy())


        train_loader_for_proto = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        prototypes = compute_prototypes(model, train_loader_for_proto, device)

        for label, proto in prototypes.items():
            class_name = le.inverse_transform([label])[0]
            print(f"class '{class_name}' (label {label}): {proto.numpy()}")
        true_proto, pred_proto = predict_with_prototypes(model, test_loader, prototypes, device)

        print(f"\n{args.dataset_name} {args.model_name} {args.method}")
        wr.writerow([args.dataset_name, args.model_name, args.method])

        f1_weighted_proto = f1_score(true_proto, pred_proto, average='weighted')
        f1_macro_proto = f1_score(true_proto, pred_proto, average='macro')
        f1_micro_proto = f1_score(true_proto, pred_proto, average='micro')
        print("F1 Score (weighted, Proto):", f1_weighted_proto)
        wr.writerow(["F1 Score (weighted):", f1_weighted_proto])
        print("F1 Score (macro, Proto):", f1_macro_proto)
        wr.writerow(["F1 Score (macro):", f1_macro_proto])
        print("F1 Score (micro, Proto):", f1_micro_proto)
        wr.writerow(["F1 Score (micro):", f1_micro_proto])

        if num_classes > 2:    
            auprc_list_proto = []
            true_one_hot = np.eye(num_classes)[true_proto]
            pred_one_hot = np.eye(num_classes)[pred_proto]
            for i in range(num_classes):
                ap = average_precision_score(true_one_hot[:, i], pred_one_hot[:, i])
                auprc_list_proto.append(ap)
                class_name = le.classes_[i]
                # print(f"Class '{class_name}' Average Precision (AUPRC, Proto): {ap:.4f}")
            auprc_macro_proto = np.mean(auprc_list_proto)
            print("Multi-class Average Precision (macro-average, Proto):", auprc_macro_proto)
            wr.writerow(["Multi-class Average Precision (macro-average):", auprc_macro_proto.item()])
            
            roc_auc_list_proto = []
            roc_auc_dict_proto = {}
            for i in range(num_classes):
                true_one_hot = np.eye(num_classes)[true_proto]
                fpr, tpr, _ = roc_curve(true_one_hot[:, i], np.eye(num_classes)[pred_proto][:, i])
                roc_auc_val = auc(fpr, tpr)
                roc_auc_list_proto.append(roc_auc_val)
                class_name = le.classes_[i]
                roc_auc_dict_proto[class_name] = roc_auc_val
                print(f"Class '{class_name}' (Proto): ROC AUC = {roc_auc_val:.4f}")
            roc_auc_overall_proto = np.mean(roc_auc_list_proto)
            print("Multi-class ROC AUC (macro-average, Proto):", roc_auc_overall_proto)
            wr.writerow(["Multi-class ROC AUC (macro-average):", roc_auc_overall_proto.item()])
            
        else:
            true_binary = (true_proto==1).astype(int)
            pred_binary = np.eye(num_classes)[pred_proto][:, 1]
            ap = average_precision_score(true_binary, pred_binary)
            print(f"Binary Average Precision (AUPRC, Proto): {ap:.4f}")
            wr.writerow(["Binary Average Precision (macro-average):", ap])

            fpr, tpr, _ = roc_curve((true_proto==1).astype(int), np.eye(num_classes)[pred_proto][:, 1])
            roc_auc_val = auc(fpr, tpr)
            print(f"Binary ROC AUC (Proto): {roc_auc_val:.4f}")
            wr.writerow(["Binary ROC AUC (macro-average):", roc_auc_val])


        print("\nClassification Report (Proto):")
        print(classification_report(true_proto, pred_proto, target_names=le.classes_, zero_division=0))
        wr.writerow([classification_report(true_proto, pred_proto, target_names=le.classes_, zero_division=0)])

        print(f"\nAccuracy: {np.mean(true_proto == pred_proto)*100:.2f}%, F1-score (macro): {f1_macro_proto:.4f}")


if __name__ == "__main__":
    main()