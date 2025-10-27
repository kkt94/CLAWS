import os
import json
import numpy as np
from sklearn.metrics import (
    classification_report, f1_score,
    roc_curve, auc, average_precision_score
)
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
import argparse
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


def main():
    parser = argparse.ArgumentParser(description="Run Step 4. Evaluation Strategy - XGBoost")
    parser.add_argument("--seed", type=int, default=42, help="randoom seed setting")
    parser.add_argument("--model_name", type=str, default="Deepseek-math-7b-rl", help="The model was used in the experiment and will be evaluated.")
    parser.add_argument("--output_dir", type=str, default="output", help="evaluation output dir")
    parser.add_argument("--dataset_name", type=str, default="TEST")

    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.dataset_name, "evaluation", "XGBoost")
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

        X_train = np.array(X_train_list)  # shape: (n_samples, 5)
        y_train = np.array(y_train_list)
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(y_train)

        with open(test_path, 'r') as f:
            test_data = json.load(f)
            print("Test sample count: ", len(test_data))
        
        X_test_list = []
        y_test_list = []

        for item in test_data:
            X_test_list.append(item['CLAWS'])
            y_test_list.append(item['label'])

        X_test = np.array(X_test_list)  # shape: (n_samples, 5)
        y_test = np.array(y_test_list)
        y_test_encoded = le.transform(y_test)

        X_train_reshaped = X_train
        X_test_reshaped  = X_test

        num_class = len(np.unique(y_train_encoded))
        if num_class > 2:
            clf = XGBClassifier(
                objective="multi:softmax",
                num_class=num_class,
                random_state=args.seed,
                use_label_encoder=False
            )
        else:
            clf = XGBClassifier(
            objective="binary:logistic",
            random_state=args.seed,
            use_label_encoder=False,
            eval_metric="logloss"
        )
            
        clf.fit(X_train_reshaped, y_train_encoded)
        test_preds = clf.predict(X_test_reshaped)
        probs      = clf.predict_proba(X_test_reshaped)

        print(f"\n{args.dataset_name} {args.model_name} {args.method}")
        wr.writerow([args.dataset_name, args.model_name, args.method])
        if num_class == 2:
            f1_macro    = f1_score(y_test_encoded, test_preds, average="macro",    zero_division=0)
            f1_micro    = f1_score(y_test_encoded, test_preds, average="micro",    zero_division=0)
            f1_weighted = f1_score(y_test_encoded, test_preds, average="weighted", zero_division=0)

            print("F1 Score (weighted):", f1_weighted)
            wr.writerow(["F1 Score (weighted):", f1_weighted])
            print("F1 Score (macro):", f1_macro)
            wr.writerow(["F1 Score (macro):", f1_macro])
            print("F1 Score (micro):", f1_micro)
            wr.writerow(["F1 Score (micro):", f1_micro])

            fpr, tpr, _ = roc_curve(y_test_encoded, probs[:, 1])
            roc_auc_val = auc(fpr, tpr)
            ap_val = average_precision_score((y_test_encoded == 1).astype(int), probs[:, 1])

            print(f"Binary Average Precision (macro-average): {ap_val:.4f}")
            wr.writerow(["Binary Average Precision (macro-average):", ap_val])
            print(f"Binary ROC AUC: {roc_auc_val:.4f}")
            wr.writerow(["Binary ROC AUC (macro-average):", roc_auc_val])

        else:
            f1_macro    = f1_score(y_test_encoded, test_preds, average="macro",    zero_division=0)
            f1_micro    = f1_score(y_test_encoded, test_preds, average="micro",    zero_division=0)
            f1_weighted = f1_score(y_test_encoded, test_preds, average="weighted", zero_division=0)
            print("F1 Score (weighted):", f1_weighted)
            wr.writerow(["F1 Score (weighted):", f1_weighted])
            print("F1 Score (macro):", f1_macro)
            wr.writerow(["F1 Score (macro):", f1_macro])
            print("F1 Score (micro):", f1_micro)
            wr.writerow(["F1 Score (micro):", f1_micro])

            ap_vals = []
            for i, cls in enumerate(le.classes_):
                val = average_precision_score((y_test_encoded == i).astype(int), probs[:, i])
                ap_vals.append(val)
                # print(f"Class '{cls}': AP = {val:.4f}")
            print("Multi-class Average Precision (macro-average):", f"{np.mean(ap_vals):.4f}")
            wr.writerow(["Multi-class Average Precision (macro-average):", np.mean(ap_vals).item()])

            roc_vals = []
            for i, cls in enumerate(le.classes_):
                fpr, tpr, _ = roc_curve((y_test_encoded == i).astype(int), probs[:, i])
                val = auc(fpr, tpr)
                roc_vals.append(val)
                # print(f"Class '{cls}': ROC AUC = {val:.4f}")
            print("Multi-class ROC AUC (macro-average):", f"{np.mean(roc_vals):.4f}")
            wr.writerow(["Multi-class ROC AUC (macro-average):", np.mean(roc_vals).item()])
            
        print("\n Classification Report:")
        print(le.classes_)
        print(classification_report(y_test_encoded, test_preds, target_names=le.classes_, zero_division=0))
        wr.writerow([classification_report(y_test_encoded, test_preds, target_names=le.classes_, zero_division=0)])


if __name__ == "__main__":
    main()