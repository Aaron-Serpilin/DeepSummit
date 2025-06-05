import sys
from pathlib import Path
# import torch
# import torchvision
# import matplotlib.pyplot as plt
# from tqdm.auto import tqdm
# from dbfread import DBF
# from torch.utils.tensorboard import SummaryWriter
# import cdsapi
# import pandas as pd
# from einops import rearrange, repeat
# import pygrib

# ─── torch & torchvision ───────────────────────────────────────────────────────
try:
    import torch
    import torchvision

    # Enforce minimum versions: torch ≥ 2.0, torchvision ≥ 0.15
    torch_major = int(torch.__version__.split(".")[0])
    torchvision_minor = int(torchvision.__version__.split(".")[1])
    if torch_major < 2:
        raise ImportError(f"Installed torch version {torch.__version__} is too old. Needs ≥ 2.0.")
    if torchvision_minor < 15:
        raise ImportError(f"Installed torchvision version {torchvision.__version__} is too old. Needs ≥ 0.15.")

    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

except ImportError as e:
    print("[ERROR] torch/torchvision error:", e)
    print("Please install or upgrade with:")
    print("    pip install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113")
    sys.exit(1)

# ─── matplotlib ────────────────────────────────────────────────────────────────
try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib is not installed.")
    print("Please install with:")
    print("    pip install matplotlib")
    sys.exit(1)

# ─── torchinfo (formerly torchsummary) ─────────────────────────────────────────
try:
    from torchinfo import summary
except ImportError:
    print("[ERROR] torchinfo is not installed.")
    print("Please install with:")
    print("    pip install torchinfo")
    sys.exit(1)

# ─── tqdm ───────────────────────────────────────────────────────────────────────
try:
    from tqdm.auto import tqdm
except ImportError:
    print("[ERROR] tqdm is not installed.")
    print("Please install with:")
    print("    pip install tqdm")
    sys.exit(1)

# ─── dbfread ────────────────────────────────────────────────────────────────────
try:
    from dbfread import DBF
except ImportError:
    print("[ERROR] dbfread is not installed.")
    print("Please install with:")
    print("    pip install dbfread")
    sys.exit(1)

# ─── TensorBoard SummaryWriter ─────────────────────────────────────────────────
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("[ERROR] torch.utils.tensorboard (TensorBoard) is not installed.")
    print("Please install with:")
    print("    pip install tensorboard")
    sys.exit(1)

# ─── torchmetrics & mlxtend ─────────────────────────────────────────────────────
try:
    import torchmetrics
    import mlxtend

    # Example version check for mlxtend
    mlxtend_version = mlxtend.__version__.split(".")
    mlxtend_minor = int(mlxtend_version[1])
    if mlxtend_minor < 19:
        raise ImportError(f"Installed mlxtend version {mlxtend.__version__} is too old. Needs ≥ 0.19.")

    print(f"mlxtend version: {mlxtend.__version__}")

except ImportError as e:
    print("[ERROR] torchmetrics or mlxtend error:", e)
    print("Please install or upgrade with:")
    print("    pip install torchmetrics mlxtend")
    sys.exit(1)

# ─── cdsapi ─────────────────────────────────────────────────────────────────────
try:
    import cdsapi
except ImportError:
    print("[ERROR] cdsapi is not installed.")
    print("Please install with:")
    print("    pip install cdsapi")
    sys.exit(1)

# ─── pandas ────────────────────────────────────────────────────────────────────
try:
    import pandas as pd
except ImportError:
    print("[ERROR] pandas is not installed.")
    print("Please install with:")
    print("    pip install pandas")
    sys.exit(1)

# ─── einops ─────────────────────────────────────────────────────────────────────
try:
    from einops import rearrange, repeat
except ImportError:
    print("[ERROR] einops is not installed.")
    print("Please install with:")
    print("    pip install einops")
    sys.exit(1)

# ─── pygrib ─────────────────────────────────────────────────────────────────────
try:
    import pygrib
except ImportError:
    print("[ERROR] pygrib is not installed.")
    print("Please install with:")
    print("    pip install pygrib")
    sys.exit(1)

sys.path.append("src")

device = "cuda" if torch.cuda.is_available() else "cpu"

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

met_weights = {
    key: 1.2 
    for key, description in weather_mapping.items()
    if any(term in description.lower() for term in ("wind", "temperature", "pressure"))
}

priority_features = [key for key in met_weights]

from src.helper_functions import set_seeds, set_data_splits, create_dataloaders
from src.met_transformer.met_utils import WeatherDataset

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

# weather_train_dataloader, weather_val_dataloader, weather_test_dataloader

offsets= range(0, 8)
met_weights_with_offset = {
    f"{feat}_t-{off}":weight
    for feat, weight in met_weights.items()
    for off in offsets
}

from src.met_transformer.met_model import Stormer

stormer = Stormer(img_size=[128, 256],
                  variables=variables,
                  met_weights=met_weights_with_offset,
                  patch_size=2,
                  hidden_size=1024,
                  depth=24,
                  num_heads=16,
                  mlp_ratio=4.0)

from src.met_transformer.met_train import train_step, test_step

# Hyperparameters pulled from the paper
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(stormer.parameters(),lr=3e-4, betas=(0.9, 0.999), weight_decay=1e-2)

X_test, mask_test, y_test, window_test = next(iter(weather_train_dataloader))
output = stormer(X_test.to(device))

result = train_step(model=stormer,
           dataloader=weather_train_dataloader,
           loss_fn=loss_fn,
           optimizer=optimizer,
           device=device,
           lambda_reg=1e-3
)

print(result)