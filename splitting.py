import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

def split_data(
    y: np.ndarray, df: pd.DataFrame
) -> list[tuple[np.ndarray, np.ndarray | None, np.ndarray]]:
    """Split data into 5 folds for robust evaluation."""
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    splits = []
    
    for train_val_idx, test_idx in skf.split(np.zeros(len(y)), y):
        y_train_val = y[train_val_idx]
        tr_idx, va_idx = train_test_split(
            train_val_idx, 
            test_size=0.20, 
            stratify=y_train_val, 
            random_state=42
        )
        splits.append((tr_idx, va_idx, test_idx))

    return splits


