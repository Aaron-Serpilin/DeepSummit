import pandas as pd
from pathlib import Path
from dbfread import DBF
from typing import Dict, List, Set

def prepare_himalayas_data(relevant_columns: Dict[str, List[str]],
                           relevant_ids: Set[str]
                           ) -> pd.DataFrame:
    
    """
    Load the three raw DBF tables, subset to only the needed columns,
    filter to only the 8K peaks & sub-peaks, and merge into one DataFrame.

    Args:
        relevant_columns: mapping table name → list of columns to retain
        relevant_ids: set of PEAKID codes to include

    Returns:
        A single merged DataFrame (members ← exped ← peaks), filtered as above.
    """

    data_path = Path("data/himalayas_data/database_files")

    himalayas_files = {
        "exped": data_path / "exped.DBF", 
        "members": data_path / "members.DBF",
        "peaks":  data_path / "peaks.DBF",
    }

    # Raw Data
    raw_dataframe = {
        key: pd.DataFrame(iter(DBF(himalayas_files[key], load=True)))
        for key in ["exped", "members", "peaks"]
    }

    # Column subset of only desired features
    subset_dataframe = {
        key: df[relevant_columns[key]]
        for key, df in raw_dataframe.items()
        if key in relevant_columns
    }

    # Peak subset of only desired PEAKIDs
    for key in ("exped", "members"):
        subset_dataframe[key] = subset_dataframe[key][subset_dataframe[key] ["PEAKID"].isin(relevant_ids)]

    # Merging columns to develop the DataFrame
    df = (
        subset_dataframe["members"]
        .merge(subset_dataframe["exped"],   on=["EXPID","PEAKID"], how="left")
        .merge(subset_dataframe["peaks"],   on=["PEAKID"],        how="left")
    )

    # Dropping duplicate climber events
    df = df.drop_duplicates(subset=["EXPID","PEAKID","MEMBID"])
    assert df.duplicated(subset=["EXPID","PEAKID","MEMBID"]).sum() == 0, "The dataset contains duplicates that disable our unique identifier on EXPID, PEAKID, MEMBID. Address these duplicates."
    return df

def finalize_himalayas_data(df: pd.DataFrame) -> pd.DataFrame:

    """
    Turn MSUCCESS → Target, drop MSUCCESS & any rows missing SMTDATE.

    Args:
        df: the merged DataFrame from prepare_himalayas_data

    Returns:
        Cleaned DataFrame with a binary 'Target' column and no null SMTDATE.
    """

    df = df[df["MSUCCESS"].notnull()].copy()
    df["Target"] = df["MSUCCESS"].map({True: 1, False: 0})
    df = df.drop(columns="MSUCCESS")
    # When analyzing the dataset, 1104 instances lack SMTDATE, out of which 1100 are unsuccessful attempts. Hence, it is a strong indicator of summit failure.
    # These rows will be dropped since there is no way to impute their year, nor date, only the season. Given their class imbalance, their incomplete inclusion might provide unnecessary noise. 
    return df.dropna(subset=["SMTDATE"])

# Documentation to run the Himalayan Database on MacOS can be found here: https://www.himalayandatabase.com/crossover.html
def load_himalayas_data (do_prepare: bool = False,
                         do_finalize: bool = False,
                        ) -> pd.DataFrame | None:
    
    """
    Carries out the entire Himalayas data preparation, and build the corresponding ML-instance .csv file.
    This function however does not request the data as the raw files are obtained from the Himalayan Database. 

    Args:
        do_prepare:  run prepare_himalayas_data (load, filter, merge)
        do_finalize: run finalize_himalayas_data (map & clean)

    Returns:
        The final DataFrame if do_finalize=True, else None.
    """

    if not do_prepare:
        return None
    
    relevant_dataframe_columns = {
        "exped": ['EXPID', 'PEAKID', 'SMTDATE', 'SEASON', 'SMTMEMBERS', 'SMTHIRED', 'MDEATHS', 'HDEATHS', 'O2USED'],
        "members": ['EXPID', 'PEAKID', 'MEMBID', 'MSUCCESS', 'SEX', 'CALCAGE', 'CITIZEN', 'STATUS', 'MO2USED', 'MROUTE1'],
        "peaks": ['PEAKID', 'PKNAME', 'HEIGHTM']
    }

    eight_k_peak_ids = ['ANN1', 'CHOY', 'DHA1', 'EVER', 'KANG', 'LHOT', 'MAKA', 'MANA'] # Given the weather dataset, we are interested in the 8K peaks. Hence, we will filter our data based on them
    subpeak_ids = ['ANNM', 'ANNE', 'KANC', 'KANS', 'LSHR', 'YALU', 'YALW', 'LHOM'] # The Himalayan database only has information on the Nepalese Himalayas. Hence, we will use sub peaks as well to have a more thorough dataset
    relevant_ids = set(eight_k_peak_ids + subpeak_ids)

    merged_df = prepare_himalayas_data(relevant_dataframe_columns, relevant_ids)

    if do_finalize:
        return finalize_himalayas_data(merged_df)
    
    return None
