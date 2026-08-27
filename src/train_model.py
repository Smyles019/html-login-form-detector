import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import os

from data_processing import load_and_merge_data, process_structural_features, clean_signature_for_ngrams

def prepare_and_train():
    # 1. Load Data
    file_path_1 = '../data/raw/Dataset 1/train_login_form.csv'
    file_path_2 = '../data/raw/Dataset 1/train_no_form.csv'
    df = load_and_merge_data(file_path_1, file_path_2)

    # 2. Extract Features
    df_struct = process_structural_features(df)
    
    # Define features and target
    X_struct = df_struct
    # Map labels to 0/1 (0 = NO_FORM, 1 = LOGIN_FORM_MALICIOUS)
    label_map = {'NO_FORM': 0, 'LOGIN_FORM_MALICIOUS': 1}
    y = df['label'].map(label_map).values
    signatures = df['html_signature']

    # 3. Train/Test Split (before N-gram vectorizer to prevent data leakage)
    X_struct_train, X_struct_test, y_train, y_test, sig_train, sig_test = train_test_split(
        X_struct, y, signatures, test_size=0.20, random_state=42, stratify=y
    )

    # 4. N-Gram Feature Extraction
    print("Extracting N-gram features...")
    clean_train = sig_train.apply(clean_signature_for_ngrams)
    clean_test = sig_test.apply(clean_signature_for_ngrams)
    
    vectorizer = CountVectorizer(ngram_range=(2, 3), token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b', min_df=2)
    X_ngram_train = vectorizer.fit_transform(clean_train)
    X_ngram_test = vectorizer.transform(clean_test)

    # 5. Scale Structural Features
    print("Scaling structural features...")
    scaler = StandardScaler()
    struct_cols = ['signature_length', 'max_depth', 'total_tags', 'unique_tags_count',
                   'form_count', 'input_count', 'button_count', 'div_count',
                   'script_count', 'a_count', 'img_count', 'input_per_form_ratio']
    
    X_struct_train_scaled = scaler.fit_transform(X_struct_train[struct_cols])
    X_struct_test_scaled = scaler.transform(X_struct_test[struct_cols])
    
    X_struct_train_sparse = sparse.csr_matrix(X_struct_train_scaled)
    X_struct_test_sparse = sparse.csr_matrix(X_struct_test_scaled)

    # 6. Combine Features
    print("Combining structural and N-gram features...")
    X_train = sparse.hstack([X_struct_train_sparse, X_ngram_train], format='csr')
    X_test = sparse.hstack([X_struct_test_sparse, X_ngram_test], format='csr')

    # 7. Train Model
    print("Training Support Vector Machine (LinearSVC) model...")
    model = LinearSVC(
        class_weight='balanced', 
        C=1.0, 
        random_state=42,
        max_iter=2000
    )
    model.fit(X_train, y_train)
    print("Model training complete.")

    # 8. Save Artifacts for Evaluation
    os.makedirs('../models', exist_ok=True)
    os.makedirs('../data/processed', exist_ok=True)
    
    with open('../models/svm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('../models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('../models/ngram_vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
        
    sparse.save_npz('../data/processed/X_test.npz', X_test)
    np.save('../data/processed/y_test.npy', y_test)
    
    print("Saved trained model, preprocessors, and test data successfully.")

if __name__ == "__main__":
    prepare_and_train()
