import json
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_random_forest(model, X_test, y_test, feature_names, label_classes, results_dir: str = None):
    """
    Evaluates the Random Forest model on the test set.
    Computes key classification metrics, the confusion matrix, and feature importances.
    Saves metrics to results_dir if provided.
    """
    print(f"Evaluating Random Forest on {X_test.shape[0]} test samples...")
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Assume binary classification where 'NO_FORM' is class 1 and 'LOGIN_FORM' is class 0
    # To compute ROC-AUC properly, we use the probability of the positive class (class 1)
    if y_proba.shape[1] > 1:
        auc = roc_auc_score(y_test, y_proba[:, 1])
    else:
        auc = 0.0

    accuracy = accuracy_score(y_test, y_pred)
    # Using weighted metrics to account for class imbalance generically
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1,
        "ROC-AUC": auc
    }
    
    print("\n--- Evaluation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {c}" for c in label_classes],
        columns=[f"Pred {c}" for c in label_classes]
    )
    print("\n--- Confusion Matrix ---")
    print(cm_df)
    
    # Feature Importances
    importances = model.feature_importances_
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\n--- Top 20 Features ---")
    print(fi_df.head(20))
    
    if results_dir:
        res_path = Path(results_dir)
        res_path.mkdir(parents=True, exist_ok=True)
        
        with open(res_path / 'random_forest_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)
            
        fi_df.to_csv(res_path / 'random_forest_feature_importance.csv', index=False)
        print(f"\nSaved metrics and feature importances to {res_path}/")
        
    return metrics, cm_df, fi_df
