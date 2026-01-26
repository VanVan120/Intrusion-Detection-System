import pandas as pd
import glob
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

class IDSDataLoader:
    def __init__(self, config):
        self.config = config
        self.raw_path = config['data']['raw_path']
        self.processed_path = config['data']['processed_path']
        
    def load_and_preprocess(self):
        """Loads data from CSVs, cleans, samples, and saves/returns processed df."""
        if os.path.exists(self.processed_path):
            print(f"Loading existing processed file: {self.processed_path}")
            df = pd.read_csv(self.processed_path)
            return df
            
        print(f"Processing raw data from {self.raw_path}...")
        all_files = glob.glob(os.path.join(self.raw_path, "*.csv"))
        
        if not all_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_path}")

        df_list = []
        for filename in all_files:
            print(f"Reading {os.path.basename(filename)}...")
            try:
                # Remove hardcoded columns to allow dynamic dataset loading
                df_iter = pd.read_csv(filename, index_col=None, header=0, low_memory=False)
                df_list.append(df_iter)
            except Exception as e:
                print(f"Error reading {filename}: {e}")

        if not df_list:
            raise ValueError("No data could be loaded.")

        df = pd.concat(df_list, axis=0, ignore_index=True)
        
        # Clean column names (remove whitespace)
        df.columns = df.columns.str.strip()
        
        # Robust handling for duplicate columns (often added by Pandas as .1)
        # Example: "Fwd Header Length.1"
        cols_to_drop = [c for c in df.columns if c.endswith('.1') and c[:-2] in df.columns]
        if cols_to_drop:
            print(f"Dropping duplicate columns: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)
            
        # Robust Label Detection
        label_col = None
        possible_labels = ['Label', 'label', 'class', 'Class', 'target']
        for col in possible_labels:
            if col in df.columns:
                label_col = col
                break
        
        if label_col:
            if label_col != 'Label':
                print(f"Renaming '{label_col}' to 'Label'...")
                df.rename(columns={label_col: 'Label'}, inplace=True)
        else:
            print("WARNING: Label column not detection. Assuming last column is Label.")
            df.rename(columns={df.columns[-1]: 'Label'}, inplace=True)

        # Robust Binary Encoding
        # If Label is string, map BENIGN/NORMAL -> 0, else -> 1
        if df['Label'].dtype == 'object':
            df['Label'] = df['Label'].astype(str).str.strip().str.upper()
            df['Label'] = df['Label'].apply(lambda x: 0 if x in ['BENIGN', 'NORMAL'] else 1)
            
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        # Robust Feature Type Handling: Ensure all features are numeric
        # This prevents crashes if the dataset contains categorical strings (e.g., Protocol names, IPs)
        # We drop non-numeric columns except Label
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Label' not in numeric_cols:
             # If Label was strings, it should have been converted above. 
             # If it wasn't (because it was already numeric), it's in numeric_cols.
             # If it was converted, it's numeric.
             pass
             
        # Select only numeric columns + Label
        cols_to_keep = [c for c in df.columns if c in numeric_cols or c == 'Label']
        dropped_non_numeric = [c for c in df.columns if c not in cols_to_keep]
        
        if dropped_non_numeric:
            print(f"WARNING: Dropped non-numeric columns: {dropped_non_numeric}")
        
        df = df[cols_to_keep]
        
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        
        # Sample
        fraction = self.config['data'].get('sample_fraction', 0.1)
        df_sample, _ = train_test_split(df, train_size=fraction, stratify=df['Label'], random_state=42)
        
        df_sample.to_csv(self.processed_path, index=False)
        print(f"Saved processed data to {self.processed_path}")
        return df_sample

    def get_data_split(self):
        """Returns X_train, X_test, y_train, y_test using config settings"""
        df = self.load_and_preprocess()
        X = df.drop(columns=['Label'])
        # Remove constant columns
        X = X.loc[:, X.std() > 0]
        y = df['Label']
        feature_names = X.columns.tolist()
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'], 
            random_state=self.config['data']['random_state'], 
            stratify=y
        )
        
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, feature_names
