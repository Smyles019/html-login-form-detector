import numpy as np
import pandas as pd
import pickle
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix
import os

def interpret_confusion_matrix(cm):
    """Provides a plain-English explanation of the confusion matrix."""
    true_negatives = cm[0, 0]
    false_positives = cm[0, 1]
    false_negatives = cm[1, 0]
    true_positives = cm[1, 1]
    
    print("\n--- CONFUSION MATRIX EXPLAINED ---")
    print(f"True Positives (TP): {true_positives} pages were correctly identified as MALICIOUS phishing forms.")
    print(f"True Negatives (TN): {true_negatives} pages were correctly identified as BENIGN (normal) pages.")
    
    print(f"\nFalse Positives (FP): {false_positives} benign pages were INCORRECTLY flagged as malicious.")
    print("  -> Meaning: Legitimate users or admins might be blocked or warned unnecessarily.")
    
    print(f"False Negatives (FN): {false_negatives} malicious phishing pages were MISSED by the model.")
    print("  -> Meaning: This is dangerous. These are phishing attacks that slipped past our defense.")

def evaluate_and_interpret():
    print("Loading test data and model artifacts...")
    
    # Check if files exist
    if not os.path.exists('data/processed/X_test.npz'):
        print("Test data not found! Please run 'python src/train_model.py' first.")
        return
        
    # Load test data
    X_test = sparse.load_npz('data/processed/X_test.npz')
    y_test = np.load('data/processed/y_test.npy')
    
    # Load model and vectorizer
    with open('models/svm_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/ngram_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    # Make predictions
    print("Evaluating the model on unseen test data...\n")
    y_pred = model.predict(X_test)
    
    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    interpret_confusion_matrix(cm)
    
    # 2. Classification Report
    print("\n--- PERFORMANCE METRICS ---")
    report = classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malicious (1)'])
    print(report)
    print("How to interpret these numbers:")
    print("- Precision (Malicious): When the model says 'This is phishing', how often is it actually right?")
    print("- Recall (Malicious): Out of ALL the real phishing pages in the test set, what percentage did we successfully catch?")
    print("- F1-Score: A balanced average between Precision and Recall. High F1 means the model is both accurate and catches most attacks.\n")
    
    # 3. Feature Importance
    print("--- WHAT IS THE MODEL LOOKING AT? (Top Features) ---")
    struct_cols = ['signature_length', 'max_depth', 'total_tags', 'unique_tags_count',
                   'form_count', 'input_count', 'button_count', 'div_count',
                   'script_count', 'a_count', 'img_count', 'input_per_form_ratio']
    
    feature_names = struct_cols + [f"pattern: {name}" for name in vectorizer.get_feature_names_out()]
    coefs = model.coef_[0]
    
    coef_df = pd.DataFrame({
        'feature': feature_names,
        'importance': coefs
    }).sort_values('importance', key=abs, ascending=False)
    
    print("Top 10 features that flag a page as MALICIOUS (Positive importance):")
    print(coef_df[coef_df['importance'] > 0].head(10).to_string(index=False))
    
    print("\nTop 10 features that flag a page as BENIGN (Negative importance):")
    print(coef_df[coef_df['importance'] < 0].head(10).to_string(index=False))
    
if __name__ == "__main__":
    evaluate_and_interpret()
