"""
Subject-level data loading to prevent leakage
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit
import json

class AlzheimerDataset(Dataset):
    """PyTorch dataset for Alzheimer's MRI slices"""
    
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load .npy file
        img = np.load(self.file_paths[idx])
        
        # Convert to 3-channel for pretrained models
        img_3ch = np.stack([img, img, img], axis=2)
        
        # Apply transforms
        if self.transform:
            img_3ch = self.transform(img_3ch)
        
        label = self.labels[idx]
        return img_3ch, label

def get_subject_id(filename):
    """Extract subject ID from filename to prevent leakage"""
    # Assumes format: subjectID_sliceNum.npy
    # Adapt this based on your filename pattern
    parts = Path(filename).stem.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]

def create_subject_level_splits(data_dir, test_size=0.15, random_state=42):
    """Create train/val/test splits at subject level"""
    data_dir = Path(data_dir)
    
    classes = ['Normal', 'VeryMild', 'Mild', 'Moderate']
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    all_files = []
    all_labels = []
    all_subjects = []
    
    for cls in classes:
        cls_dir = data_dir / cls
        if not cls_dir.exists():
            continue
        
        for file in cls_dir.glob('*.npy'):
            all_files.append(str(file))
            all_labels.append(class_to_idx[cls])
            all_subjects.append(get_subject_id(file))
    
    all_files = np.array(all_files)
    all_labels = np.array(all_labels)
    all_subjects = np.array(all_subjects)
    
    # First split: separate test set
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_val_idx, test_idx = next(gss1.split(all_files, all_labels, groups=all_subjects))
    
    # Second split: separate validation from train
    gss2 = GroupShuffleSplit(n_splits=1, test_size=test_size/(1-test_size), random_state=random_state)
    train_idx, val_idx = next(gss2.split(all_files[train_val_idx], all_labels[train_val_idx], 
                                         groups=all_subjects[train_val_idx]))
    
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]
    
    # Verify no subject leakage
    train_subjects = set(all_subjects[train_idx])
    val_subjects = set(all_subjects[val_idx])
    test_subjects = set(all_subjects[test_idx])
    
    assert not (train_subjects & val_subjects), "Subject leakage: train-val"
    assert not (train_subjects & test_subjects), "Subject leakage: train-test"
    assert not (val_subjects & test_subjects), "Subject leakage: val-test"
    
    splits = {
        'train': {'files': all_files[train_idx].tolist(), 'labels': all_labels[train_idx].tolist()},
        'val': {'files': all_files[val_idx].tolist(), 'labels': all_labels[val_idx].tolist()},
        'test': {'files': all_files[test_idx].tolist(), 'labels': all_labels[test_idx].tolist()},
        'classes': classes
    }
    
    return splits

def save_splits(splits, output_path):
    """Save splits to JSON"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

def load_splits(splits_path):
    """Load splits from JSON"""
    with open(splits_path, 'r') as f:
        return json.load(f)
