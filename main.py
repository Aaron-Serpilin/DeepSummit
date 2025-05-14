#!/usr/bin/env python
# coding: utf-8

# # Deep Learning Framework for Predicting Himalayan Summit Success 
# 

# ## 0. Setup
# 
# The following section of the notebook covers all relevant imports required for the project. 

# In[1]:


# The following code is only for Google Collab whenever wanting to perform runs there
# !git clone https://github.com/Aaron-Serpilin/DeepSummit.git
# !pip install --upgrade pip
# !pip install -e .


# In[3]:


from torch import nn
from torchvision import transforms

try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[0]) >= 2, "torch version should be 2.+"
    assert int(torchvision.__version__.split(".")[1]) >= 15, "torchvision version should be 0.15+"
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not correct. Installing correct versions.")
    get_ipython().system('pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113')
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[INFO] Couldn't find matplotlib...installing it")
    get_ipython().system('pip install -q matplotlib')
    import matplotlib.pyplot as plt

try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it")
    get_ipython().system('pip install -q torchinfo')
    from torchinfo import summary

try:
    from tqdm.auto import tqdm
except:
    print(f"[INFO] Couldnt't find tqdm... installing it ")
    get_ipython().system('pip install tqdm')
    from tqdm.auto import tqdm

try:
    from torchinfo import summary
except ImportError:
    print("[INFO] Couldn't find torchinfo... installing it")
    get_ipython().system('pip install -q torchinfo')
    from torchinfo import summary

try:
    from dbfread import DBF
except ImportError:
    print("[INFO] Coudln't find dbfread...installing it")
    get_ipython().system('pip install -q dbfread')
    from dbfread import DBF


try:
    from torch.utils.tensorboard import SummaryWriter
except:
    print("[INFO] Couldn't find tensorboard... installing it.")
    get_ipython().system('pip install -q tensorboard')
    from torch.utils.tensorboard import SummaryWriter

try:
    import torchmetrics, mlxtend
    print(f"mlextend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >- 19
except:
    get_ipython().system('pip install -q torchmetrics -U mlxtend')
    import torchmetrics, mlxtend
    print(f"mlextend version: {mlxtend.__version__}")

try:
    import cdsapi
except ImportError:
    print("[INFO] Coudldn't find cdsapi...installing it.")
    get_ipython().system('pip install -q cdsapi')
    import cdsapi

try:
    import pandas as pd
except ImportError:
    print("[INFO] Couldn't find pandas... installing it")
    get_ipython().system('pip install -q pandas')
    import pandas as pd

try:
    from einops import rearrange, repeat
except ImportError:
    print("[INFO] Couldn't find einops... installing it")
    get_ipython().system('pip install -q einops')
    from einops import rearrange, repeat

try:
    import pygrib
except ImportError:
    print("[INFO] Couldn't find pygrib... installing it")
    get_ipython().system('pip install -q pygrib')
    import pygrib


# In[3]:


device = "cuda" if torch.cuda.is_available() else "cpu"
device


# ## 1. Himalayan Data Setup
# 
# The following section covers the retrieval and processing of the tabular data from the Himalayan Database that will use the SAINT architecture to carry out inference. The raw data was obtained through the `get_data.py` file. 

# In[4]:


# We adjust the PYTHONPATH to keep absolute imports
import sys
sys.path.append("src")


# In[5]:


from pathlib import Path

himalayan_train_dir = Path("data/himalayas_data/train")
himalayan_val_dir = Path("data/himalayas_data/val")
himalayan_test_dir = Path("data/himalayas_data/test")

himalayan_train_file = himalayan_train_dir / "train.csv"
himalayan_val_file = himalayan_val_dir / "val.csv"
himalayan_test_file = himalayan_test_dir / "test.csv"


# In[6]:


df_train = pd.read_csv(himalayan_train_file)

print(f"First 10 rows:\n{df_train.head(10)}")
print(f"First training instance:\n{df_train.iloc[0]}")
print(f"Instance shape:\n{df_train.iloc[0].shape}")


# The `load_himalayan_data()` function defined within the `tab_data_setup` file transforms the raw data from .DBF to .csv, filters the relevant features, filters the relevant peaks, develops a data frame with this data, and then creates the corresponding train, val, and test splits by using the `create_dataloaders()` helper function.

# In[7]:


from src.tab_transformer.tab_data_setup import load_himalayan_data

train_dataloader, val_dataloader, test_dataloader = load_himalayan_data()


# ## 2. SAINT Model instantiation
# 
# The following section instantiates the SAINT model architecture based on the hyperparameter selection defined in the relevant paper. This is broken down in the `tab_breakdown.ipynb` file where the core equations and backbone of the architecture is explained. 

# In[8]:


cat_batch, cont_batch, label_batch, cat_mask_batch, cont_mask_batch = next(iter(train_dataloader))

# First instance
cat_instance = cat_batch[0]
cont_instance = cont_batch[0]
label_instance = label_batch[0]

print(f"Categorical instance: {cat_instance}\nCategorical instance shape: {cat_instance.shape}\n")
print(f"Continuous instance: {cont_instance}\nContinuous instance shape: {cont_instance.shape}\n")
print(f"Label instance: {label_instance}\nLabel instance shape: {label_instance.shape}\n")


# In[11]:


from src.tab_transformer.tab_model import SAINT
import numpy as np

categorical_columns = ['SEX', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1', 'SEASON', 'O2USED']
continuous_columns = ['CALCAGE', 'HEIGHTM', 'MDEATHS', 'HDEATHS', 'SMTMEMBERS', 'SMTHIRED']

# Returns the amount of unique values per categorical column
cat_dims = [len(np.unique(df_train[col])) for col in categorical_columns]

# Hyperparameter selection based on default original architecture instantiation
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


# ## 3. SAINT Model Training
# 
# The following section trains the SAINT model with the `TabularDataset` class while saving the corresponding files in the `runs` directory using the `SummaryWriter()` setup.

# In[ ]:


from src.tab_transformer.tab_train import train_step, test_step

loss_fn = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(saint.parameters(),lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)

train_step(model=saint,
           dataloader=train_dataloader,
           loss_fn=loss_fn,
           optimizer=optimizer,
           device=device
)

val_step = test_step(model=saint,
                     dataloader=val_dataloader,
                     loss_fn=loss_fn,
                     device=device)

val_step


# In[12]:


writer = SummaryWriter()


# In[13]:


def create_writer(experiment_name: str,
                  model_name: str,
                  extra: str=None) -> torch.utils.tensorboard.writer.SummaryWriter():

                  from datetime import datetime
                  import os

                  timestamp = datetime.now().strftime("%Y-%m-%d--%H:%M:%S")

                  if extra:
                       log_dir = os.path.join("runs", timestamp, experiment_name, model_name, extra) # Create the log directory path
                  else:
                       log_dir = os.path.join("runs", timestamp, experiment_name, model_name) # Create the log directory path

                  print(f"[INFO] Created SummaryWriter, saving to: {log_dir}")
                  return SummaryWriter(log_dir=log_dir)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'from helper_functions import set_seeds\nfrom src.tab_transformer.tab_train import train\n\nset_seeds(42)\n\n\nresults = train(model=saint,\n      train_dataloader=train_dataloader,\n      val_dataloader=val_dataloader,\n      test_dataloader=test_dataloader,\n      optimizer=optimizer,\n      loss_fn=loss_fn,\n      epochs=10,\n      writer=create_writer(experiment_name="first_run",\n                                   model_name="saint")\n)\n')


# With a `test_acc` of ~92%, the SAINT Model architecture achieves better results than the benchmark Regression model from the Julia project. Therefore, prior to fine tuning the model, and without the incorporation of contextual meteorological data, the model already has a better performance than the baseline. 

# In[ ]:


from helper_functions import plot_loss_curves

plot_loss_curves(results)


# In[ ]:


# get_ipython().run_line_magic('load_ext', 'tensorboard')
# get_ipython().run_line_magic('tensorboard', '--logdir runs')


# ## 4. Era5 Data Setup
# 
# The following section covers the data preparation of the era5 dataset. The raw data was obtained through the `get_data.py` file. 

# The entire data setup from the era5 dataset is condensed into the `load_era5_data` function, which does multiple things. 
# 
# Firstly, it uses`file_to_grib()` to convert all the raw era5 files into .grib files, which facilitate it for processing afterwards. 
# 
# This is followed by `get_variable_mapping()` which obtains the corresponding mapping of the internal grib abbreviations and the actual features from the era5 dataset. This is used to then filter out the desired features when we transform the .grib file into a .csv file. 
# 
# For convenience, we can run `get_variable_mapping()` each time, but in this study case the mapping doesn't change every time. Therefore, we can store the result of the first run in a variable and then merely refer to it, which saves up ~3 minutes of runtime.  
# 
# This mapping is this passed as a parameter to `process_grib_to_csv()` to filter out the desired features and convert the .grib files to .csv, storing them in `data/era5_data/instances/raw_instances`.
# 
# Afterwards, due to the API request limitations, the original requests in `get_data.py` were done in batches of 5 years, leading to each mountain to have 17 different weather data files. Therefore, in order to condense them, we then use `merge_daily_instances()` which condenses all of these files into a single .csv file in `data/era5_data/instances/merged_instances`. 
# 
# The final step of the function is to then use `build_event_instances()` to create the final ML table by the weather and tabular data are matched based on their `date` and `peakid`. It then creates the instances with the target variable from the tabular dataset, and also adds the relevant 7-day context window to analyze the weather trends that affect summit probabilities. 

# In[ ]:


from src.met_transformer.met_data_setup import load_era5_data

ml_table = load_era5_data()
instances_output_path = Path('data/era5_data/instances')
instances_output_path.mkdir(parents=True, exist_ok=True)
ml_table_output_file = instances_output_path / 'ml_table.csv'
ml_table.to_csv(ml_table_output_file, index=False)
print(f"Wrote {len(ml_table)} event instances to {ml_table_output_file}")

