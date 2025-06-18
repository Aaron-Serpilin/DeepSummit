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

script_dir    = os.path.dirname(__file__)           
project_root  = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))
sys.path.insert(0, project_root)

from torch import nn
from torchvision import transforms

from src.met_transformer.met_train import train_step, test_step, train
from src.helper_functions import set_seeds, set_data_splits, create_dataloaders, plot_loss_curves, save_model, create_writer
from src.met_transformer.met_utils import WeatherDataset
from src.met_transformer.met_model import Stormer

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device is: {device}\n")

weather_mapping = {
    '10u':   '10 metre U wind component',
    '10v':   '10 metre V wind component',
    '2d':    '2 metre dewpoint temperature',
    '2t':    '2 metre temperature',
    'msl':   'Mean sea level pressure',
    'sp':    'Surface pressure',
    'tp':    'Total precipitation',
    'istl1': 'Ice temperature layer 1',
    'istl2': 'Ice temperature layer 2',
    'istl3': 'Ice temperature layer 3',
    'istl4': 'Ice temperature layer 4',
    'mx2t':  'Maximum temperature at 2 metres since previous post-processing',
    'mn2t':  'Minimum temperature at 2 metres since previous post-processing',
    'skt':   'Skin temperature',
    '100u':  '100 metre U wind component',
    '100v':  '100 metre V wind component',
    'u10n':  '10 metre u-component of neutral wind',
    'v10n':  '10 metre v-component of neutral wind',
    '10fg':  'Maximum 10 metre wind gust since previous post-processing',
    'i10fg': 'Instantaneous 10 metre wind gust',
    'cbh':   'Cloud base height',
    'hcc':   'High cloud cover',
    'lcc':   'Low cloud cover',
    'mcc':   'Medium cloud cover',
    'tcc':   'Total cloud cover',
    'tciw':  'Total column cloud ice water',
    'tclw':  'Total column cloud liquid water',
    'viiwd': 'Vertical integral of divergence of cloud frozen water flux',
    'vilwd': 'Vertical integral of divergence of cloud liquid water flux',
    'viiwe': 'Vertical integral of eastward cloud frozen water flux',
    'vilwe': 'Vertical integral of eastward cloud liquid water flux',
    'viiwn': 'Vertical integral of northward cloud frozen water flux',
    'vilwn': 'Vertical integral of northward cloud liquid water flux',
    'cp':    'Convective precipitation',
    'crr':   'Convective rain rate',
    'ilspf': 'Instantaneous large-scale surface precipitation fraction',
    'lsrr':  'Large scale rain rate',
    'lsp':   'Large-scale precipitation',
    'lspf':  'Large-scale precipitation fraction',
    'mxtpr': 'Maximum total precipitation rate since previous post-processing',
    'mntpr': 'Minimum total precipitation rate since previous post-processing',
    'ptype': 'Precipitation type',
    'tcrw':  'Total column rain water',
    'csf':   'Convective snowfall',
    'csfr':  'Convective snowfall rate water equivalent',
    'lssfr': 'Large scale snowfall rate water equivalent',
    'lsf':   'Large-scale snowfall',
    'asn':   'Snow albedo',
    'rsn':   'Snow density',
    'sd':    'Snow depth',
    'es':    'Snow evaporation',
    'sf':    'Snowfall',
    'smlt':  'Snowmelt',
    'tsn':   'Temperature of snow layer',
    'tcsw':  'Total column snow water'
}

met_weights = {
    key: 1.2 
    for key, description in weather_mapping.items()
    if any(term in description.lower() for term in ("wind", "temperature", "pressure"))
}

priority_features = [key for key in met_weights]

variables = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_dewpoint_temperature",
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure",
        "total_precipitation",
        "ice_temperature_layer_1",
        "ice_temperature_layer_2",
        "ice_temperature_layer_3",
        "ice_temperature_layer_4",
        "maximum_2m_temperature_since_previous_post_processing",
        "minimum_2m_temperature_since_previous_post_processing",
        "skin_temperature",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "10m_u_component_of_neutral_wind",
        "10m_v_component_of_neutral_wind",
        "10m_wind_gust_since_previous_post_processing",
        "instantaneous_10m_wind_gust",
        "cloud_base_height",
        "high_cloud_cover",
        "low_cloud_cover",
        "medium_cloud_cover",
        "total_cloud_cover",
        "total_column_cloud_ice_water",
        "total_column_cloud_liquid_water",
        "vertical_integral_of_divergence_of_cloud_frozen_water_flux",
        "vertical_integral_of_divergence_of_cloud_liquid_water_flux",
        "vertical_integral_of_eastward_cloud_frozen_water_flux",
        "vertical_integral_of_eastward_cloud_liquid_water_flux",
        "vertical_integral_of_northward_cloud_frozen_water_flux",
        "vertical_integral_of_northward_cloud_liquid_water_flux",
        "convective_precipitation",
        "convective_rain_rate",
        "instantaneous_large_scale_surface_precipitation_fraction",
        "large_scale_rain_rate",
        "large_scale_precipitation",
        "large_scale_precipitation_fraction",
        "maximum_total_precipitation_rate_since_previous_post_processing",
        "minimum_total_precipitation_rate_since_previous_post_processing",
        "precipitation_type",
        "total_column_rain_water",
        "convective_snowfall",
        "convective_snowfall_rate_water_equivalent",
        "large_scale_snowfall_rate_water_equivalent",
        "large_scale_snowfall",
        "snow_albedo",
        "snow_density",
        "snow_depth",
        "snow_evaporation",
        "snowfall",
        "snowmelt",
        "temperature_of_snow_layer",
        "total_column_snow_water"
    ]

set_seeds(42)

splits_path = Path("data/era5_data")
weather_csv = Path("data/era5_data/era5_data.csv")
weather_df = pd.read_csv(weather_csv, parse_dates=["event_date"])

metadata_cols = ["PEAKID", "parent_peakid", "event_date"]
X = weather_df.drop(columns=["Target"])
y = weather_df["Target"]

set_data_splits(X, y, splits_path, seed=42)

weather_train_dataloader, weather_val_dataloader, weather_test_dataloader = create_dataloaders(
    dataset_class=WeatherDataset,
    train_file=splits_path / "train" / "train.csv",
    val_file=splits_path / "val" / "val.csv",
    test_file=splits_path / "test" / "test.csv",
    batch_size=32,
    dataset_kwargs={
        'target_column': 'Target',
        'metadata_cols': metadata_cols,
        'continuous_mean_std': None,
        'priority_features':  priority_features,
        'variables': variables
    }

)

print(f"Weather train dataloader: {weather_train_dataloader}\nWeather val dataloader: {weather_val_dataloader}\nWeather test dataloader: {weather_test_dataloader}\n")

offsets= range(0, 8)
met_weights_with_offset = {
    f"{feat}_t-{off}":weight
    for feat, weight in met_weights.items()
    for off in offsets
}


stormer = Stormer(img_size=[128, 256],
                  variables=variables,
                  met_weights=met_weights_with_offset,
                  patch_size=2,
                  hidden_size=612,
                  depth=12,
                  num_heads=12,
                  mlp_ratio=4.0)

print("Passed Initialization (hidden_size % num_heads == 0)")

stormer.to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(stormer.parameters(),lr=0.00001, betas=(0.9, 0.999), weight_decay=1e-3)

stormer_results = train(model=stormer,
                train_dataloader=weather_train_dataloader,
                val_dataloader=weather_val_dataloader,
                test_dataloader=weather_test_dataloader,
                optimizer=optimizer,
                loss_fn=loss_fn,
                epochs=50,
                writer=create_writer(experiment_name="stormer_runs",
                                    extra="epochs_50_lr_1e-5_hidden_size_612_depth_12_heads_12_decay_1e-3"))

plot_loss_curves(stormer_results)

save_model(stormer,
          "/var/scratch/ase347/DeepSummit/checkpoints",
          "stormer_epochs_50_lr_1e-5_hidden_size_612_depth_12_heads_12_decay_1e-3.pth")
