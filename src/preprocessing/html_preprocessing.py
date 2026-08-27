import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack

STRUCTURAL_FEATURE_NAMES = [
    'c_form', 'c_input', 'c_button', 'c_label', 'c_a', 'c_div',
    'has_form_input', 'has_multi_input', 'has_input_button', 'has_form_button',
    'input_to_div_ratio', 'max_depth'
]

def load_and_merge(data_path, form_file, no_form_file):
    """Helper function to load and clean a specific split"""
    df1 = pd.read_csv(data_path / form_file)
    df2 = pd.read_csv(data_path / no_form_file)
    df = pd.concat([df1, df2], ignore_index=True)

    # Label schema mapping
    label_mapping = {
        'LOGIN_FORM_MALICIOUS': 'LOGIN_FORM',
        'TEST_LOGIN_FORM': 'LOGIN_FORM',
        'LOGIN_FORM': 'LOGIN_FORM',
        'TEST_NO_FORM': 'NO_FORM',
        'NO_FORM': 'NO_FORM'
    }

    df['html_signature'] = df['html_signature'].fillna('')
    df['clean_label'] = df['label'].map(label_mapping).fillna(df['label'])
    
    df = df.dropna(subset=['clean_label']).reset_index(drop=True)
    return df

def load_data(data_dir: str):
    """
    Loads raw CSV files (train and validation) from the given directory.
    Maps labels and fills missing HTML signatures without deleting rows.
    """
    data_path = Path(data_dir)
    df_train = load_and_merge(data_path, 'train_login_form.csv', 'train_no_form.csv')
    df_val = load_and_merge(data_path, 'validation_login_form.csv', 'validation_no_form.csv')
    return df_train, df_val

def extract_structural_topology(html_series):
    """
    Extracts structural count and relational metrics from the HTML signature strings.
    Returns a numpy array of features for each row.
    """
    metrics = []
    for s in html_series:
        s = str(s).lower()
        c_form = s.count('(form')
        c_input = s.count('(input')
        c_button = s.count('(button')
        c_label = s.count('(label')
        c_a = s.count('(a')
        c_div = s.count('(div')
        
        has_form_input = 1 if re.search(r'\(form.*\(input', s) else 0
        has_multi_input = 1 if c_input >= 2 else 0
        has_input_button = 1 if re.search(r'\(input.*\(button', s) else 0
        has_form_button = 1 if re.search(r'\(form.*\(button', s) else 0
        
        input_to_div_ratio = c_input / (c_div + 1)
        
        max_depth = 0
        curr_depth = 0
        for char in s:
            if char == '(':
                curr_depth += 1
                if curr_depth > max_depth:
                    max_depth = curr_depth
            elif char == ')':
                curr_depth = max(0, curr_depth - 1)
                
        metrics.append([
            c_form, c_input, c_button, c_label, c_a, c_div,
            has_form_input, has_multi_input, has_input_button, has_form_button,
            input_to_div_ratio, max_depth
        ])
    return np.array(metrics)

def preprocess_and_split(df_train, df_val):
    """
    Uses the predefined train and validation data splits.
    Fits vectorizers and scalers ONLY on the training data.
    """
    # 1. Encode Labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['clean_label'].astype(str))
    y_test = label_encoder.transform(df_val['clean_label'].astype(str))
    
    # 2. Structural Features
    X_train_top_raw = extract_structural_topology(df_train['html_signature'])
    X_test_top_raw = extract_structural_topology(df_val['html_signature'])
    
    scaler = RobustScaler()
    X_train_top = scaler.fit_transform(X_train_top_raw)
    X_test_top = scaler.transform(X_test_top_raw)
    
    # 3. N-Grams
    seq_vectorizer = CountVectorizer(
        token_pattern=r'\([a-z0-9]+|\)',
        ngram_range=(1, 3),
        min_df=2,
        max_features=4000
    )
    
    X_train_seq = seq_vectorizer.fit_transform(df_train['html_signature'])
    X_test_seq = seq_vectorizer.transform(df_val['html_signature'])
    
    # 4. Combine Features
    X_train = hstack([X_train_top, X_train_seq]).tocsr()
    X_test = hstack([X_test_top, X_test_seq]).tocsr()
    
    feature_names = STRUCTURAL_FEATURE_NAMES + list(seq_vectorizer.get_feature_names_out())
    
    return X_train, X_test, y_train, y_test, feature_names, label_encoder
