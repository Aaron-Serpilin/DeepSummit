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

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.tab_transformer.tab_model import SAINT
from src.met_transformer.met_model import Stormer
from src.late_fusion.model import DeepSummit

from src.late_fusion.utils import FusionDataset
from src.tab_transformer.tab_utils import TabularDataset
from src.met_transformer.met_utils import WeatherDataset

from src.late_fusion.extract_logits import extract_logits_tab, extract_logits_met
from src.helper_functions import set_seeds, save_model, create_writer
from src.late_fusion.train import train_step, test_step, train

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is: {device}\n")

set_seeds(42)

#### SAINT Setup ####

from src.scripts.train_saint import tabular_train_dataloader, tabular_val_dataloader, tabular_test_dataloader

saint_path = Path("src/models/saint_model.pth")
saint = torch.load(saint_path, map_location=device).to(device)

for parameter in saint.parameters(): parameter.requires_grad = False

#### Stormer Setup ####

from src.scripts.train_stormer import weather_train_dataloader, weather_val_dataloader, weather_test_dataloader

stormer_path = Path("src/models/stormer_model.pth")
stormer = torch.load(stormer_path, map_location=device).to(device)

for parameter in stormer.parameters(): parameter.requires_grad = False

#### DeepSummit Setup ####

tabular_logits_train, y_train_tabular = extract_logits_tab(saint, tabular_train_dataloader, device)
weather_logits_train, _ = extract_logits_met(stormer, weather_train_dataloader, device)

tabular_logits_val, y_val_tabular  = extract_logits_tab(saint, tabular_val_dataloader, device)
weather_logits_val,  _  = extract_logits_met(stormer, weather_val_dataloader, device)

tabular_logits_test, y_test_tabular = extract_logits_tab(saint, tabular_test_dataloader, device)
weather_logits_test,  _ = extract_logits_met(stormer, weather_test_dataloader, device)

# Different structure than the other datasets and DataLoaders, hence, we don't reuse create_dataloaders

train_dataset = FusionDataset(tabular_logits_train, weather_logits_train, y_train_tabular)
val_dataset   = FusionDataset(tabular_logits_val,   weather_logits_val,   y_val_tabular)
test_dataset  = FusionDataset(tabular_logits_test,  weather_logits_test,  y_test_tabular)

train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=os.cpu_count(), pin_memory=True, shuffle=True)
val_dataloader   = DataLoader(val_dataset,   batch_size=64, num_workers=os.cpu_count(), pin_memory=True)
test_dataloader  = DataLoader(test_dataset,  batch_size=64, num_workers=os.cpu_count(), pin_memory=True)

print(f"Fusion train dataloader: {train_dataloader}\nFusion val dataloader: {val_dataloader}\nFusion test dataloader: {test_dataloader}\n")

deepsummit = DeepSummit(
    tabular_model_path=saint_path,
    weather_model_path=stormer_path,
    num_classes=2,           
    freeze_layers=True,
    device=device
).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(deepsummit.parameters(), lr=1e-3)

deepsummit_results = train(model=deepsummit,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=10,
                writer=create_writer(experiment_name="deepsummit_runs",
                                    extra="deepsummit_model_epochs_10_fusion_layers_4"))

save_model(saint,
          "/var/scratch/ase347/DeepSummit/checkpoints",
          "deepsummit_model_epochs_10_fusion_layers_4.pth")

