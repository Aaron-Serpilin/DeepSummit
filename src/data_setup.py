import pandas as pd
from pathlib import Path
from dbfread import DBF
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from helper_functions import set_seeds, set_data_splits

# Documentation to run the Himalayan Database on MacOS can be found here: https://www.himalayandatabase.com/crossover.html
### 1. Loading Data ###
himalayan_data_path = Path('data/himalayas_data/database_files')

himalaya_files = {
    "exped": himalayan_data_path / "exped.DBF", 
    "members": himalayan_data_path / "members.DBF",
    "peaks":  himalayan_data_path / "peaks.DBF",
}

himalaya_dataframes = {
   key: pd.DataFrame(iter(DBF(himalaya_files[key], load=True)))
    for key in ['exped', 'members', 'peaks']
}

relevant_dataframe_columns = {
    "exped": ['EXPID', 'PEAKID', 'SMTDATE', 'SEASON', 'SMTMEMBERS', 'SMTHIRED', 'MDEATHS', 'HDEATHS', 'O2USED'],
    "members": ['EXPID', 'PEAKID', 'MEMBID', 'MSUCCESS', 'SEX', 'CALCAGE', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1'],
    "peaks": ['PEAKID', 'PKNAME', 'HEIGHTM']
}

filtered_himalaya_dataframes = {
    key: df[relevant_dataframe_columns[key]]
    for key, df in himalaya_dataframes.items()
    if key in relevant_dataframe_columns
}

eight_k_peak_ids = ['ANN1', 'CHOY', 'DHA1', 'EVER', 'KANG', 'LHOT', 'MAKA', 'MANA'] # Given the weather dataset, we are interested in the 8K peaks. Hence, we will filter our data based on them
subpeak_ids = ['ANNM', 'ANNE', 'KANC', 'KANS', 'LSHR', 'YALU', 'YALW', 'LHOM'] # The Himalayan database only has information on the Nepalese Himalayas. Hence, we will use subpeaks as well to have a more thorough dataset
relevant_ids = set(eight_k_peak_ids + subpeak_ids)

relevant_himalaya_dataframes = {
    key: df[df['PEAKID'].isin(relevant_ids)]
    for key, df in filtered_himalaya_dataframes.items()
}

### 2. Developing DataFrames ###

merged_df = (
    relevant_himalaya_dataframes["members"]
    .merge(relevant_himalaya_dataframes["exped"], on=["EXPID", "PEAKID"], how="left")
    .merge(relevant_himalaya_dataframes["peaks"], on="PEAKID", how="left")
)

unique_key = ['EXPID', 'PEAKID', 'MEMBID'] # these three values together identify a unique climber-experience record
merged_df = merged_df.drop_duplicates(subset=unique_key)

assert merged_df.duplicated(subset=unique_key).sum() == 0, "The dataset contains duplicates that disable our unique identifier on EXPID, PEAKID, MEMBID. Address these duplicates."

merged_df = merged_df[merged_df['MSUCCESS'].notnull()] 
merged_df['Target'] = merged_df['MSUCCESS'].map({True: 1, False:0})
merged_df = merged_df.drop(columns='MSUCCESS') # after the mapping we no longer need this column

### 3. Processing different feature datatypes ###

categorical_columns = ['SEX', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1', 'SEASON', 'O2USED']
numerical_columns = ['CALCAGE', 'HEIGHTM', 'MDEATHS', 'HDEATHS', 'SMTMEMBERS', 'SMTHIRED']

# Label encoding
for col in categorical_columns:
    merged_df[col] = merged_df[col].astype('category').cat.codes

# Normalizing [(value - mean) / std]
scaler = StandardScaler()
merged_df[numerical_columns] = scaler.fit_transform(merged_df[numerical_columns])

### 4. Feature Matrix and Target Vector ###
seed = 42
set_seeds(seed)
feature_columns = categorical_columns + numerical_columns
X = merged_df[feature_columns]
y = merged_df['Target']

### 5. Splits ###

set_data_splits(X, y, himalayan_data_path, 42)

# For reproducibility
output_file = himalayan_data_path / "processed_himalaya_data.csv"

if output_file.exists():
    print("[INFO] Himalaya Data has already been processed")
else: 
    himalayan_data_path.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(output_file, index=False)
