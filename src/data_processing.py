import pandas as pd
import re
from collections import Counter
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

def load_and_merge_data(file_path_1, file_path_2):
    """
    Loads two CSV files containing HTML signatures and labels,
    merges them, and handles missing values and duplicates.
    """
    print(f"Loading data from {file_path_1} and {file_path_2}...")
    df1 = pd.read_csv(file_path_1)
    df2 = pd.read_csv(file_path_2)
    df = pd.concat([df1, df2], ignore_index=True)
    
    # Handle missing values
    df['html_signature'] = df['html_signature'].fillna('')
    
    # Remove duplicates based on sha256
    before = len(df)
    df = df.drop_duplicates(subset='sha256', keep='first').reset_index(drop=True)
    duplicates_removed = before - len(df)
    if duplicates_removed > 0:
        print(f"Removed {duplicates_removed} duplicate records based on SHA256.")
        
    return df

def extract_structural_features(sig: str) -> dict:
    """
    Calculates structural statistics from an HTML signature string.
    """
    max_depth = 0
    current_depth = 0
    
    # Calculate maximum DOM depth based on parentheses
    for char in sig:
        if char == '(':
            current_depth += 1
            max_depth = max(max_depth, current_depth)
        elif char == ')':
            current_depth = max(0, current_depth - 1)
            
    # Extract tags (alphanumeric strings)
    tags = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', sig.lower())
    tag_counts = Counter(tags)
    
    form_count = tag_counts.get("form", 0)
    input_count = tag_counts.get("input", 0)
    
    return {
        "signature_length": len(sig),
        "max_depth": max_depth,
        "total_tags": len(tags),
        "unique_tags_count": len(tag_counts),
        "form_count": form_count,
        "input_count": input_count,
        "button_count": tag_counts.get("button", 0),
        "div_count": tag_counts.get("div", 0),
        "script_count": tag_counts.get("script", 0),
        "a_count": tag_counts.get("a", 0),
        "img_count": tag_counts.get("img", 0),
        "input_per_form_ratio": input_count / form_count if form_count > 0 else 0.0
    }

def process_structural_features(df):
    """
    Applies the structural feature extraction to the entire DataFrame.
    Returns a DataFrame containing only the numeric features.
    """
    print("Extracting structural features from HTML signatures...")
    struct_features = [extract_structural_features(sig) for sig in df['html_signature']]
    return pd.DataFrame(struct_features)

def clean_signature_for_ngrams(sig):
    """
    Removes parentheses and returns a space-separated string of tags 
    for CountVectorizer to process.
    """
    tags = re.findall(r'[a-zA-Z][a-zA-Z0-9]*', sig.lower())
    return ' '.join(tags)
