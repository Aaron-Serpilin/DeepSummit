import pandas as pd
from pathlib import Path
from dbfread import DBF

# Documentation to run the Himalayan Database on MacOS can be found here: https://www.himalayandatabase.com/crossover.html
himalayan_data_path = Path('data/himalayas_data')

himalaya_files = {
    "exped": himalayan_data_path / "exped.DBF", 
    # "filters":  himalayan_data_path / "filters.DBF",
    "members": himalayan_data_path / "members.DBF",
    "peaks":  himalayan_data_path / "peaks.DBF",
    # "setup":  himalayan_data_path / "SETUP.DBF"
}

himalaya_dataframes = {
    "exped": pd.DataFrame(iter(DBF(himalaya_files["exped"], load=True))),
    "members": pd.DataFrame(iter(DBF(himalaya_files["members"], load=True))),
    "peaks": pd.DataFrame(iter(DBF(himalaya_files["peaks"], load=True)))
}

relevant_dataframe_columns = {
    "exped": ['EXPID', 'PEAKID', 'SMTDATE', 'SMTTIME', 'SEASON', 'SUCCESS1', 'ASCENT1', 'SMTMEMBERS', 'SMTHIRED', 'MDEATHS', 'HDEATHS', 'O2USED'],
    "members": ['EXPID', 'PEAKID', 'MEMBID', 'MSUCCESS', 'SEX', 'CALCAGE', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1'],
    "peaks": ['PEAKID', 'PKNAME', 'HEIGHTM']
}

filtered_himalaya_dataframes = {
    key: df[relevant_dataframe_columns[key]]
    for key, df in himalaya_dataframes.items()
    if key in relevant_dataframe_columns
}

# Given the weather dataset, we are interested in the 8K peaks. Hence, we will filter our data based on them.
eight_k_peak_ids = ['ANN1', 'CHOY', 'DHA1', 'EVER', 'KANG', 'LHOT', 'MAKA', 'MANA']

# The Himalayan database only has information on the Nepalese Himalayas. Hence, we will use subpeak as well to have a more thorough dataset
subpeak_ids = ['ANNM', 'ANNE', 'KANC', 'KANS', 'LSHR', 'YALU', 'YALW' 'LHOM']

# To display more rows, place the number of rows inside the head method
print(f"[INFO]\nEXPED HEAD: {filtered_himalaya_dataframes['exped'].head()}\nMEMBERS HEAD: {filtered_himalaya_dataframes['members'].head()}\nPEAKS HEAD: {filtered_himalaya_dataframes['peaks'].head()}\n")

eight_k_ids = []

peaks_df = filtered_himalaya_dataframes["peaks"]
for index, row in peaks_df.iterrows():
    if row["HEIGHTM"] > 8000:
        eight_k_ids.append(row["PEAKID"])

print(eight_k_ids)




