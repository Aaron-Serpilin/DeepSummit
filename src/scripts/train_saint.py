import sys
import os
import subprocess
import pathlib
from pathlib import Path

try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[0]) >= 2, "torch version should be 2.+"
    assert int(torchvision.__version__.split(".")[1]) >= 15, "torchvision version should be 0.15+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except (ImportError, AssertionError) as e:
    print(f"[INFO] torch/torchvision versions not correct or missing: {e}")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-U", "torch", "torchvision", "torchaudio", "--extra-index-url", "https://download.pytorch.org/whl/cu113"],
        check=True
    )
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

# --- matplotlib ---
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[INFO] Couldn't find matplotlib…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib"], check=True)
    import matplotlib.pyplot as plt

# --- torchinfo.summary ---
try:
    from torchinfo import summary
except ImportError:
    print("[INFO] Couldn't find torchinfo…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchinfo"], check=True)
    from torchinfo import summary

# --- tqdm.auto.tqdm ---
try:
    from tqdm.auto import tqdm
except ImportError:
    print("[INFO] Couldn't find tqdm…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tqdm"], check=True)
    from tqdm.auto import tqdm

# --- dbfread.DBF ---
try:
    from dbfread import DBF
except ImportError:
    print("[INFO] Couldn't find dbfread…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "dbfread"], check=True)
    from dbfread import DBF

# --- torch.utils.tensorboard.SummaryWriter ---
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("[INFO] Couldn't find tensorboard…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tensorboard"], check=True)
    from torch.utils.tensorboard import SummaryWriter

# --- torchmetrics, mlxtend ---
try:
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    # ensure mlxtend ≥ 0.19
    assert int(mlxtend.__version__.split(".")[1]) >= 19
except (ImportError, AssertionError):
    print("[INFO] Installing/upgrading torchmetrics and mlxtend")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torchmetrics", "-U", "mlxtend"], check=True)
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")

# --- cdsapi ---
try:
    import cdsapi
except ImportError:
    print("[INFO] Couldn't find cdsapi…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cdsapi"], check=True)
    import cdsapi

# --- pandas ---
try:
    import pandas as pd
except ImportError:
    print("[INFO] Couldn't find pandas…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pandas"], check=True)
    import pandas as pd

# --- einops.rearrange, einops.repeat ---
try:
    from einops import rearrange, repeat
except ImportError:
    print("[INFO] Couldn't find einops…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops"], check=True)
    from einops import rearrange, repeat

# --- pygrib ---
try:
    import pygrib
except ImportError:
    print("[INFO] Couldn't find pygrib…installing it")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pygrib"], check=True)
    import pygrib

# --- numpy as np ---
try:
    import numpy as np
    print(f"numpy version: {np.__version__}")
except ImportError:
    print("[INFO] Couldn't find numpy…installing it")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "numpy"],
        check=True
    )
    import numpy as np
    print(f"numpy version: {np.__version__}")

script_dir    = os.path.dirname(__file__)           
project_root  = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from torch import nn
from torchvision import transforms

from src.tab_transformer.tab_train import train_step, test_step, train
from src.helper_functions import set_seeds, set_data_splits, create_dataloaders, plot_loss_curves, save_model, create_writer
from src.tab_transformer.tab_utils import TabularDataset
from src.tab_transformer.tab_model import SAINT

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is: {device}\n")

set_seeds(42)

splits_path = Path("data/himalayas_data")
tab_csv = splits_path / "himalayas_data.csv"
tab_df = pd.read_csv(tab_csv, parse_dates=["SMTDATE"])

categorical_columns = ['SEX', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1', 'SEASON', 'O2USED']
continuous_columns = ['CALCAGE', 'HEIGHTM', 'MDEATHS', 'HDEATHS', 'SMTMEMBERS', 'SMTHIRED']
feature_columns = categorical_columns + continuous_columns

X = tab_df[feature_columns]
y = tab_df["Target"]

set_data_splits(X, y, splits_path, seed=42)

continuous_means = [tab_df[col].mean() for col in continuous_columns]
continuous_stds = [tab_df[col].std() for col in continuous_columns]
continuous_mean_std = list(zip(continuous_means, continuous_stds))

tabular_train_dataloader, tabular_val_dataloader, tabular_test_dataloader = create_dataloaders(
        dataset_class=TabularDataset,
        train_file=splits_path / "train" / "train.csv",
        val_file=splits_path / "val" / "val.csv",
        test_file=splits_path / "test" / "test.csv",
        num_workers=os.cpu_count(),
        batch_size=32,
        dataset_kwargs={
            'target_column': 'Target',
            'cat_cols': categorical_columns,
            'continuous_mean_std': continuous_mean_std,
        }
    )

print(f"Tabular train dataloader: {tabular_train_dataloader}\nTabular val dataloader: {tabular_val_dataloader}\nTabular test dataloader: {tabular_test_dataloader}\n")

himalayan_train_dir = splits_path / "train"
himalayan_train_file = himalayan_train_dir / "train.csv"
df_train = pd.read_csv(himalayan_train_file)

categorical_columns = ['SEX', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1', 'SEASON', 'O2USED']
continuous_columns = ['CALCAGE', 'HEIGHTM', 'MDEATHS', 'HDEATHS', 'SMTMEMBERS', 'SMTHIRED']

cat_dims = [len(np.unique(df_train[col])) for col in categorical_columns]

saint = SAINT(
    categories = tuple(cat_dims), 
    num_continuous = len(continuous_columns),                
    dim = 32,                           
    dim_out = 1,                       
    depth = 6,                       
    heads = 8,  
    num_special_tokens=1,                      
    attn_dropout = 0.1,             
    ff_dropout = 0.1,                  
    mlp_hidden_mults = (4, 2),       
    cont_embeddings = 'MLP',
    attentiontype = 'colrow',
    final_mlp_style = 'sep',
    y_dim = 2 # Binary classification
)

saint.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(saint.parameters(),lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

saint_results = train(model=saint,
                train_dataloader=tabular_train_dataloader,
                val_dataloader=tabular_val_dataloader,
                test_dataloader=tabular_test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=50,
                writer=create_writer(experiment_name="first_training_run_saint",
                                    model_name="saint",
                                    extra="50_epochs"))

plot_loss_curves(saint_results)

save_model(saint,
          "/var/scratch/ase347/DeepSummit/checkpoints",
          "saint_epoch50.pth")