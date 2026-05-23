import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import re
import pickle

def add_difference_features_corrected(combined_data):
    feature_columns = [col for col in combined_data.columns 
                      if any(col.startswith(family) for family in ['PB', 'V', 'PV', 'BB','VB'])]

    family_time_scales = {}

    for family in ['PB', 'BB']:
        family_features = [col for col in feature_columns if col.startswith(family)]
        family_time_scales[family] = set()
        
        for feature in family_features:
            t_match = re.search(r'_T(\d+)', feature)
            if t_match:
                family_time_scales[family].add(int(t_match.group(1)))

    difference_combinations = [
        (9, 7),
        (10, 9),
        (9, 8),
    ]

    added_features = []

    for family in ['PB', 'VB', 'V', 'PV', 'BB']:
        family_features = [col for col in feature_columns if col.startswith(family)]
        base_features = {}
        for feature in family_features:
            base_match = re.match(r'([A-Z]+\d+)', feature)
            if base_match:
                base_name = base_match.group(1)
                if base_name not in base_features:
                    base_features[base_name] = {}
                t_match = re.search(r'_T(\d+)', feature)
                if t_match:
                    t_value = int(t_match.group(1))
                    base_features[base_name][t_value] = feature

        for base_name, time_dict in base_features.items():
            for t1, t2 in difference_combinations:
                if t1 in time_dict and t2 in time_dict:
                    feat1 = time_dict[t1]
                    feat2 = time_dict[t2]
                    diff_name = f"{base_name}_T{t1}-T{t2}"
                    combined_data[diff_name] = combined_data[feat1] - combined_data[feat2]
                    added_features.append(diff_name)

    # ✅ Always return at the end
    return combined_data, added_features
def process_and_weight_data(df, scaler_path1, scaler_path2, accuracy_file='accuracy_result.csv', diff_shift=10):

    with open(scaler_path1, 'rb') as f:
        scalers1 = pickle.load(f)
    with open(scaler_path2, 'rb') as f:
        scalers2 = pickle.load(f)
    
    # Load accuracy results and get top 100 features
    accuracy_df = pd.read_csv(accuracy_file)
    top_100_features = accuracy_df.nlargest(100, 'accuracy')['feature'].tolist()
    accuracy_dict = dict(zip(accuracy_df['feature'], accuracy_df['accuracy']))
    
    def resample_data(df, apply_diff=False, diff_shift=10):
        df_temp = df.copy()
        time_col = [col for col in df_temp.columns if 'time' in col.lower()][0]
        price_col = [col for col in df_temp.columns if 'price' in col.lower()][0]
        
        df_temp[time_col] = pd.to_datetime(df_temp[time_col])
        df_temp.set_index(time_col, inplace=True)
        
        # --- OHLC resampling for price ---
        price_ohlc = df_temp[price_col].resample('1min').ohlc()
        price_ohlc.columns = [f'Price_{col}' for col in price_ohlc.columns]
        
        # --- Resample other features ---
        feature_dfs = []
        for col in df_temp.columns:
            if col == price_col:
                continue
            if col.startswith('PB'):
                resampled = df_temp[col].resample('1min').mean()
            elif col.startswith('BB'):
                resampled = df_temp[col].resample('1min').last()
            elif col.startswith('V'):
                resampled = df_temp[col].resample('1min').sum()
            else:
                resampled = df_temp[col].resample('1min').mean()
            feature_dfs.append(resampled)
        
        features_resampled = pd.concat(feature_dfs, axis=1)
        result = pd.concat([price_ohlc, features_resampled], axis=1).reset_index()
        result.rename(columns={'index': 'Time'}, inplace=True)

        # --- Apply differencing if requested ---
        if apply_diff:
            diff_cols = {}
            for col in result.columns:
                # Only apply difference to valid feature columns (not Time or Price)
                if col.startswith(('PB', 'BB', 'VB', 'V', 'PV')):
                    diff_col_name = f"{col}_diff"
                    diff_cols[diff_col_name] = result[col] - result[col].shift(diff_shift)
            
            # Replace main result with only diff columns
            result = pd.concat([result[['Time']], pd.DataFrame(diff_cols)], axis=1)

        return result

    def filter_t4_features(df):
        cols_to_keep = []
        for col in df.columns:
            if col in ['Time', 'time', 'TIME'] or col.startswith('Price_'):
                cols_to_keep.append(col)
                continue
            t_matches = re.findall(r'_T(\d+)', col)
            if not t_matches:
                cols_to_keep.append(col)
                continue
            min_t = min(map(int, t_matches))
            if min_t >= 4:
                cols_to_keep.append(col)
        return df[cols_to_keep]

    def scale_features(df, scalers_dict):
        df_scaled = df.copy()
        for col in df_scaled.columns:
            if col in ['Time', 'time', 'TIME'] or col.startswith('Price_'):
                continue
            if col in scalers_dict:
                scaler = scalers_dict[col]
                non_nan_mask = df_scaled[col].notna()
                if non_nan_mask.any():
                    df_scaled.loc[non_nan_mask, col] = scaler.transform(
                        df_scaled.loc[non_nan_mask, col].values.reshape(-1, 1)
                    ).flatten()
        return df_scaled

    processed1 = resample_data(df, apply_diff=False)
    filtered1 = filter_t4_features(processed1)
    scaled1 = scale_features(filtered1, scalers1)
    scaled1_with_diff, _ = add_difference_features_corrected(scaled1)

    # --- Method 2: With difference ---
    processed2 = resample_data(df, apply_diff=True)
    filtered2 = filter_t4_features(processed2)
    scaled2 = scale_features(filtered2, scalers2)

    # --- Merge both dataframes ---
    combined_df = pd.merge(scaled1_with_diff, scaled2, on='Time', how='outer')
    
    # --- Keep only relevant base columns ---
    base_cols = ['Time', 'Price_close', 'Price_open', 'Price_high', 'Price_low']

    # --- Select top 100 features (matching flexible substring) ---
    available_columns = set(combined_df.columns)
    selected_columns = base_cols.copy()

    for feature in top_100_features:
        matches = [col for col in available_columns if feature in col]
        selected_columns.extend(matches)

    selected_columns = list(dict.fromkeys(selected_columns))  # remove duplicates

    final_df = combined_df[selected_columns].copy()

    for col in final_df.columns:
        if col in base_cols:
            continue
        acc = 1.0
        if col in accuracy_dict:
            acc = accuracy_dict[col]
        else:
            for key, val in accuracy_dict.items():
                if key in col:
                    acc = val
                    break
        final_df[col] *= acc
    final_df = final_df[[c for c in final_df.columns if c in base_cols or c in top_100_features or any(f in c for f in top_100_features)]]
    cols_with_many_nans = final_df.columns[final_df.isnull().sum() > 20]
    final_df.drop(columns=cols_with_many_nans, inplace=True)
    final_df = final_df.iloc[20:].reset_index(drop=True)
    return final_df
# Example usage:
def main(dfs):
    result_df = process_and_weight_data(
        df=dfs,
        scaler_path1=r'C:/Users/Kushagra tiwari/Downloads/INTERIIT_PS1/EBY_scalers.pkl',  
        scaler_path2=r'C:/Users/Kushagra tiwari/Downloads/INTERIIT_PS1/EBY_scaler2.pkl',  
        accuracy_file=r'C:/Users/Kushagra tiwari/Downloads/INTERIIT_PS1/eby_top_100_combined_accuracy.csv'
    )
    print("Processed and weighted dataframe shape:", result_df.shape)
    # print("Columns:", result_df.columns.tolist())
    return result_df
if __name__ == "__main__":
    result = main()