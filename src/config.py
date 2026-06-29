#====LIBRARY IMPORT====#
import os
import pickle
import random
import unicodedata
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import precision_recall_fscore_support
from tqdm import tqdm
from transformers import TimesformerModel
import wandb

import warnings
warnings.filterwarnings('ignore')

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


##====CONSTANTS & PATHS====#
NUM_CLASSES    = 100
TARGET_FRAMES  = 8        
IMG_SIZE       = 224

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_PATH          = 'your/path/here'
DATASET_ZIP        = os.path.join(BASE_PATH, 'dataset.zip')
LOCAL_EXTRACT_PATH = '/content/dataset_local'
TRAIN_ROOT         = '/content/dataset_local/dataset/train'
LABEL_MAP_PATH     = os.path.join(LOCAL_EXTRACT_PATH, 'dataset/label_mapping.pkl')
SAVE_PATH          = 'best_model_timesformer.pth'

# ── Training hyperparams ───────────────────────────────────────────────────
BATCH_SIZE         = 8    
ACCUM_STEPS        = 4   # effective batch = 4 * 8 = 32
LR_BACKBONE = 1e-5 
LR_HEAD     = 1e-4 
EPOCHS      = 10   
IMAGENET_MEAN      = [0.45, 0.45, 0.45]  
IMAGENET_STD       = [0.225, 0.225, 0.225]

# ── WandB ──────────────────────────────────────────────────────────────────
WANDB_PROJECT = 'sign-language-detector'
WANDB_ENTITY  = 'your-workspace'
