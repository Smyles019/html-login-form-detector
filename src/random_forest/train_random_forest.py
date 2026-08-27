import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

def train_random_forest(X_train, y_train, model_save_path: str = None):
    """
    Trains a baseline Random Forest classifier.
    Saves the model to disk if a path is provided.
    """
    print(f"Training Random Forest on {X_train.shape[0]} samples with {X_train.shape[1]} features...")
    
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced"
    )
    
    model.fit(X_train, y_train)
    print("Training complete.")
    
    if model_save_path:
        save_path = Path(model_save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_path)
        print(f"Model saved to {save_path}")
        
    return model
