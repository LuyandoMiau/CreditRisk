"""
This file contains functions for data preparation and preprocessing.
This one should directly handle data from an already existing dataset, a CSV, PICKLE or EXCEL file.
"""

# Impport necessary libraries
import logging
from functools import wraps
from time import perf_counter
from typing import Callable, ParamSpec, TypeVar, Union
import csv
import re
import pandas as pd
import numpy as np
import sklearn
import sqlite3
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union, Literal

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")


def log_execution(func: Callable[P, R]) -> Callable[P, R]:
    """Log when an operation starts, finishes, and how long it takes."""
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        operation = func.__qualname__
        started_at = perf_counter()
        logger.info("Starting %s", operation)
        try:
            result = func(*args, **kwargs)
        except Exception:
            logger.exception("%s failed after %.3f seconds", operation, perf_counter() - started_at)
            raise
        logger.info("%s completed in %.3f seconds", operation, perf_counter() - started_at)
        return result

    return wrapper


"""
This class is responsible for reading data from a CSV, EXCEL or PKL file and storing it in a pandas DataFrame. It also provides a method to create a copy of the data to avoid modifying the original dataset.
"""
class DataReader:
    
    def __init__(self, file_path: str):
        
        """ 
        The init just requires the file path of the dataset to be read. It can be a CSV, EXCEL or PKL file. 
        """
        self.file_path = file_path
        self.data = None
    
    # Called by pandas ONLY for lines with more fields than the header (unquoted/broken commas in text columns)
    def _merge_extra_fields(self, fields):
        cols = self._header_fields
        i_emp, i_desc, i_zip = cols.index('emp_title'), cols.index('desc'), cols.index('zip_code')
        n_left, n_right, n_fixed = i_emp, len(cols) - i_zip, i_desc - i_emp - 1
        left, right, mid = fields[:n_left], fields[-n_right:], fields[n_left:-n_right]

        emp_len = re.compile(r'^(< 1 year|\d+\+? years?|n/a)$')
        homes = {'RENT', 'OWN', 'MORTGAGE', 'OTHER', 'NONE', 'ANY'}
        purposes = {'car', 'credit_card', 'debt_consolidation', 'educational', 'home_improvement', 'house',
                    'major_purchase', 'medical', 'moving', 'other', 'renewable_energy', 'small_business',
                    'vacation', 'wedding'}

        # emp_title ends where emp_length (followed by a valid home_ownership) begins
        k = next((j for j in range(len(mid) - 1) if emp_len.match(mid[j]) and mid[j + 1] in homes), None)
        if k is None:
            return self._reject(fields)
        rest = mid[k + n_fixed:]  # desc ... purpose ... title
        p = next((j for j, t in enumerate(rest) if t in purposes), None)
        if p is None:
            return self._reject(fields)
        fixed = left + [','.join(mid[:k])] + mid[k:k + n_fixed] + [','.join(rest[:p]), rest[p], ','.join(rest[p + 1:])] + right
        return fixed if len(fixed) == len(cols) else self._reject(fields)

    # Keeps a record of lines that could not be realigned instead of dropping them
    def _reject(self, fields):
        self.rejected_lines.append(fields[:12])
        return None
    
    # READS the data from a CSV, EXCEL or PKL file and stores it in the data attribute.
    @log_execution
    def read_data(self):
        """
        Reads data from a CSV, Excel, or Pickle file and stores it in the data attribute.
        
        Uses the Python engine directly to prevent C-parser tokenize errors on malformed lines.
        """
        path = Path(self.file_path)
        ext = path.suffix.lower()

        if ext == '.csv':
            logger.info("Reading CSV file via Python parsing engine: %s", self.file_path)

            # NEW: Header is read first so the bad-line handler knows the expected column layout
            with open(self.file_path, encoding='utf-8-sig') as f:
                self._header_fields = f.readline().rstrip('\r\n').rstrip(';').split(',')
            self.rejected_lines = []

            # Direct python engine read to avoid c_parser_wrapper entirely
            # QUOTE_NONE stops a broken quote from swallowing following rows, and
            # on_bad_lines now repairs rows with extra commas instead of silently skipping them
            self.data = pd.read_csv(
                self.file_path,
                engine='python',
                quoting=csv.QUOTE_NONE,
                on_bad_lines=self._merge_extra_fields,
                encoding='utf-8-sig'
            )

            # --- HEADER & ARTIFACT CLEANUP ---
            # 1. Strip semicolon noise from column names
            self.data.columns = self.data.columns.astype(str).str.rstrip(';').str.strip()

            # 1b. Strip semicolon noise from the VALUES of the last column, and unwrap quoted text columns
            last = self.data.columns[-1]
            self.data[last] = pd.to_numeric(self.data[last].astype(str).str.replace(';', '', regex=False), errors='coerce')
            for c in ['emp_title', 'desc', 'title']:
                self.data[c] = self.data[c].str.replace(r'^"(.*)"$', r'\1', regex=True).str.replace('""', '"', regex=False)
            if self.rejected_lines:
                logger.warning("%d lines could not be re-aligned: %s", len(self.rejected_lines), self.rejected_lines)

            # 2. Remove columns that are entirely empty
            self.data.dropna(how='all', axis=1, inplace=True)

            # 3. Remove artifact index columns ('Unnamed: 0')
            unnamed_cols = [c for c in self.data.columns if c.startswith("Unnamed") or c == ""]
            if unnamed_cols:
                self.data.drop(columns=unnamed_cols, inplace=True)
                logger.info("Dropped artifact columns: %s", unnamed_cols)

        elif ext in ('.xlsx', '.xls'):
            logger.info("Reading Excel file: %s", self.file_path)
            self.data = pd.read_excel(self.file_path)

        elif ext == '.pkl':
            logger.info("Reading pickle file: %s", self.file_path)
            self.data = pd.read_pickle(self.file_path)

        else:
            raise ValueError(f"Unsupported file format '{ext}'. Supported formats: .csv, .xlsx, .xls, .pkl")
        
        logger.info("Loaded dataset with %d rows and %d columns", *self.data.shape)
    
    # CLEAR the specified columns from the dataset =====> ["Unnamed: 0"]
    # this is not required for the preprocessing but it can be useful to remove unnecessary columns.
    @log_execution
    def clear_columns(self, columns_to_clear: List[str]):
        
        """
        Clears the specified columns from the dataset.
        
        Parameters:
        columns_to_clear (list): List of column names to be cleared.
        """
        logger.info("Dropping columns: %s", columns_to_clear)
        self.data.drop(columns=columns_to_clear, inplace=True, errors='ignore')  # Use errors='ignore' to avoid KeyError if column doesn't exist
        logger.info("Dataset now has %d columns", len(self.data.columns))
    
    # CREATES a copy of the data to avoid modifying the original dataset.   
    @log_execution
    def copy_data(self):
        
        """
        Returns a copy of the data to avoid modifying the original dataset.
        """
        return self.data.copy()
    
""" This class is responsible for exploring the data. 
It provides methods to explore the dataset and return some basic information about it.
It also provides a method to explore the dataset using sqlite extension and return some basic information about it.
"""    
class DataExplorerAdvanced:
    
    def __init__(self, data: pd.DataFrame):
        
        """
        The init requires the data to be explored. It should be a pandas DataFrame.
        """
        self.data = data
        
    # EXPLORE the dataset and return some basic information about it.
    @log_execution
    def explore_data(self):
        
        """
        Explores the dataset and returns some basic information about it.
        """
        logger.info("Data shape: %s", self.data.shape)
        logger.info("Data types:\n%s", self.data.dtypes.to_string())
        logger.info("Missing values by column:\n%s", self.data.isnull().sum().to_string())
        logger.info("Data description:\n%s", self.data.describe().to_string())
    
    # EXPLORE the dataset using sqlite extension and return some basic information about it.
    @log_execution
    def save_to_sqlite(self, db_path: str, table_name: str):
        """
        Saves the pandas DataFrame into a physical SQLite database file (.db) on disk.
        Creates the target directory automatically if it does not exist.
        """
        target_path = Path(db_path)
        
        # Ensure the destination directory exists before opening the SQLite connection
        target_path.parent.mkdir(parents=True, exist_ok=True)

        # Connect directly to the disk file path instead of in-memory RAM
        logger.info("Connecting to SQLite database file at: %s", target_path)
        conn = sqlite3.connect(target_path)

        try:
            # Write DataFrame to the database file
            logger.info("Writing %d rows to SQLite table '%s'", len(self.data), table_name)
            self.data.to_sql(table_name, conn, if_exists='replace', index=False)
            logger.info("Successfully saved database file to %s", target_path.resolve())
        finally:
            # Ensure connection is closed cleanly even if an exception occurs
            conn.close()

"""
This class is responsible for preprocessing the data. 
It provides methods to clear unnecessary columns and transform column types based on specified transformation types.
"""
class DataPreprocessor:
    
    def __init__(self, data: pd.DataFrame):
        
        """
        The init requires the data to be preprocessed. It should be a pandas DataFrame.
        """
        self.data = data
    
    # SAVE the column names of the dataset for later use.    
    @log_execution
    def column_names(self):
        
        """
        Returns the column names of the dataset.
        """
        return self.data.columns.tolist()
    
    # CLEAR the specified columns from the dataset =====> ["Unnamed: 0"]
    # this is not required for the preprocessing but it can be useful to remove unnecessary columns.
    @log_execution
    def clear_columns(self, columns_to_clear: List[str]):
        
        """
        Clears the specified columns from the dataset.
        
        Parameters:
        columns_to_clear (list): List of column names to be cleared.
        """
        logger.info("Dropping columns: %s", columns_to_clear)
        if columns_to_clear != []:
            self.data.drop(columns=columns_to_clear, inplace=True)
        logger.info("Dataset now has %d columns", len(self.data.columns))
    
    # This function is responsible for parsing a reference date string into a pandas Timestamp object. It can handle both string and pandas Timestamp inputs.
    @staticmethod
    def parse_reference_date(date_input: Union[str, pd.Timestamp]) -> pd.Timestamp:
        """
        Converts a reference date string (e.g. '2017-12-01') into a pandas Timestamp.
        
        Parameters:
            date_input (str | pd.Timestamp): Date to parse.
            
        Returns:
            pd.Timestamp: Standardized pandas Timestamp object.
        """
        if isinstance(date_input, pd.Timestamp):
            return date_input
        
        try:
            return pd.to_datetime(date_input)
        except Exception as e:
            raise ValueError(f"Could not parse reference date '{date_input}': {e}")

    # Handle some column types transformations for columns
    # But this is a particular case of our dataset, so thez are optional so variables/columns can be different for other cases
    @log_execution
    def transform_columns_types(self, transformation_type: str, columns: List[str], new_column_names: List[str]):
        """
        Transforms a set of columns according to the specified transformation type.
        
        Supported transformation types:
        - String_to_Numeric_specif: Converts specific string representations (e.g. '< 1 year', 'n/a') to integers.
        - String_to_Numeric_specif2: Strips text indicators (e.g. ' months') and converts to integers.
        - String_to_Numeric: Standard conversion of string columns to numeric via pd.to_numeric.
        - String_to_Datetime: Converts date strings into pandas datetime objects using format '%b-%y'.
        - Categorical_to_Numeric: Encodes categorical variables into numerical codes using pandas category codes.
        
        Parameters:
            transformation_type (str): Key identifying which transformation logic to apply.
            columns (List[str]): Original source column names in self.data.
            new_column_names (List[str]): Target column names for transformed outputs.
        """
        logger.info(
            "Applying transformation '%s' to columns: %s -> %s",
            transformation_type,
            columns,
            new_column_names
        )

        # -------------------------------------------------------------------------
        # Case 1: String_to_Numeric_specif (e.g., 'emp_length' -> 'emp_length_int')
        # Extracts numeric values from strings like '< 1 year' or '10+ years', filling n/a with 0
        # -------------------------------------------------------------------------
        if transformation_type == "String_to_Numeric_specif":
            for i, column in enumerate(columns):
                target_col = new_column_names[i]
                self.data[target_col] = (
                    self.data[column]
                        .astype(str)
                        .replace({"< 1 year": "0", "n/a": "0"})
                        .str.extract(r"(\d+)", expand=False)
                        .fillna(0)
                        .astype(int)
                )
                logger.info("Transformed '%s' into new column '%s'", column, target_col)
        
        # -------------------------------------------------------------------------
        # Case 2: String_to_Numeric_specif2 (e.g., 'term' -> 'term_int')
        # Strips ' months' string noise and converts values to integers
        # -------------------------------------------------------------------------
        elif transformation_type == "String_to_Numeric_specif2":
            for i, column in enumerate(columns):
                target_col = new_column_names[i]
                self.data[target_col] = (
                    self.data[column]
                        .astype(str)
                        .str.replace(" months", "", regex=False)
                        .fillna(0)
                        .astype(int)
                )
                logger.info("Transformed '%s' into new column '%s'", column, target_col)

        # -------------------------------------------------------------------------
        # Case 3: String_to_Numeric
        # Standard numeric coercion using pd.to_numeric
        # -------------------------------------------------------------------------
        elif transformation_type == "String_to_Numeric":
            for i, column in enumerate(columns):
                target_col = new_column_names[i]
                self.data[target_col] = pd.to_numeric(self.data[column], errors='coerce')
                logger.info("Converted '%s' to numeric column '%s'", column, target_col)

        # -------------------------------------------------------------------------
        # Case 4: String_to_Datetime (e.g., 'earliest_cr_line' -> 'earliest_cr_line_date')
        # Converts date string into datetime object
        # -------------------------------------------------------------------------
        elif transformation_type == "String_to_Datetime":
            for i, column in enumerate(columns):
                target_col = new_column_names[i]
                self.data[target_col] = pd.to_datetime(self.data[column], format='%b-%y', errors='coerce')
                logger.info("Converted '%s' to datetime column '%s'", column, target_col)

        # -------------------------------------------------------------------------
        # Case 5: Categorical_to_Numeric
        # Encodes categorical variables to numeric category codes
        # -------------------------------------------------------------------------
        elif transformation_type == "Categorical_to_Numeric":
            for i, column in enumerate(columns):
                target_col = new_column_names[i]
                self.data[target_col] = self.data[column].astype('category').cat.codes
                logger.info("Encoded '%s' as numeric column '%s'", column, target_col)

        else:
            raise ValueError(f"Unsupported transformation type: '{transformation_type}'")
         
    # Specific function to get the days and months since the earliest credit line, based on the defined reference date
    @log_execution
    def get_days_months_since_earliest_credit_line(self, defined_reference_date: str):
        
        # Check that the column 'earliest_cr_line_date' exists in the DataFrame
        if 'earliest_cr_line_date' not in self.data.columns:
            raise ValueError("Column 'earliest_cr_line_date' does not exist in the DataFrame. Check in parameters in the part of data_preparation/data_preprocessor/transformations/new_column_names")
        
        # Get the days and months since the earliest credit line, based on the defined reference date
        self.data['days_since_earliest_credit_line'] = defined_reference_date - self.data['earliest_cr_line_date']
        self.data['months_since_earliest_credit_line'] = (self.data['days_since_earliest_credit_line'].dt.days / 30.44).round()
        
        # Also get some descriptive statistics for the new columns
        logger.info(
            "Days since earliest credit line:\n%s",
            self.data['days_since_earliest_credit_line'].describe().to_string(),
        )
        logger.info(
            "Months since earliest credit line:\n%s",
            self.data['months_since_earliest_credit_line'].describe().to_string(),
        )
        
    # Here a function to handle negative values for months_since_earliest_credit_line, in case the reference date is in the future, the negative values will be set to the maximum value of the column, and the user will get notified about it
    @log_execution
    def handle_negative_values_months_since_earliest_credit_line(self):
        if 'months_since_earliest_credit_line' in self.data.columns:
            negative_values_count = (self.data['months_since_earliest_credit_line'] < 0).sum()
            min_value = self.data['months_since_earliest_credit_line'].min()
            logger.info(
                "Minimum months since earliest credit line: %s",
                min_value,
            )
            if negative_values_count > 0:
                max_value = self.data['months_since_earliest_credit_line'].max()
                self.data.loc[self.data['months_since_earliest_credit_line'] < 0, 'months_since_earliest_credit_line'] = max_value
                logger.warning(
                    "Replaced %d negative months-since-earliest-credit-line values with the column maximum (%s)",
                    negative_values_count,
                    max_value,
                )
            else:
                logger.info("No negative months-since-earliest-credit-line values found")
        else:
            logger.info("Skipping negative-value handling; target column is absent")
    
    # Get days and months since issue_date, based on the defined reference date
    @log_execution
    def get_days_months_since_issue_date(self, defined_reference_date: str):
        
        # Check that the column 'issue_d_date' exists in the DataFrame
        if 'issue_d_date' not in self.data.columns:
            raise ValueError("Column 'issue_d_date' does not exist in the DataFrame. Check in parameters in the part of data_preparation/data_preprocessor/transformations/new_column_names")
        
        # Get the days and months since the issue date, based on the defined reference date
        self.data['days_since_issue_date'] = defined_reference_date - self.data['issue_d_date']
        self.data['months_since_issue_date'] = (self.data['days_since_issue_date'].dt.days / 30.44).round()
        
        # Also get some descriptive statistics for the new columns
        logger.info(
            "Days since issue date:\n%s",
            self.data['days_since_issue_date'].describe().to_string(),
        )
        logger.info(
            "Months since issue date:\n%s",
            self.data['months_since_issue_date'].describe().to_string(),
        )
    
    # Function to create the dummy variables for the categorical columns, and drop the original columns, and also drop the first dummy variable to avoid multicollinearity
    @log_execution
    def create_dummy_variables(self, categorical_columns: List[str]):
        """
        Creates dummy variables for the specified categorical columns 
        and keeps all other columns in the dataset.
        """
        logger.info("Creating dummy variables for columns: %s", categorical_columns)
        
        # Pass the full dataset (self.data) and specify columns=categorical_columns
        self.data = pd.get_dummies(
            self.data, 
            columns=categorical_columns, 
            prefix=categorical_columns, 
            prefix_sep=":", 
            dtype=int
        )
        logger.info("Dummy variables created. Dataset now has %d columns", len(self.data.columns))
        
    # Handle empty values in the dataset
    @log_execution
    def handle_empty_values(self, columns: List[str], case: str):
        
        """
        First give a summary of the empty values in the dataset, then handle them accordingly.
        """
        
        # Get the summary of the null values
        def get_null_summary(columns_needed: List[str]) -> pd.DataFrame:
            return pd.DataFrame({
                "null_count": self.data[columns_needed].isna().sum(), # number of empty rows
                "null_pct": (self.data[columns_needed].isna().mean() * 100).round(2), # percentage of the total
                "var_type": self.data[columns_needed].dtypes.astype(str),
                "mean": self.data[columns_needed].mean()
            }).sort_values("null_count", ascending=False)
        
        # Render the summary as plain text so it is readable in a terminal.
        null_summary = get_null_summary(columns_needed=columns)
        print("\nNull-value summary:")
        print(null_summary.to_string())
        
        # TIME TO HANDLE THE EMPTY VALUES
        
        # Case specific: loan_data_2007_2014
        
        if case == "default_loan_data_2007_2014":
        
            # Specific case for the column total_rev_hi_lim
            # ========== total_rev_hi_lim ================
            # if there is no data for the revolving limit, we assume it is equal to the fundded amount
            self.data["total_rev_hi_lim"] = (
                self.data["total_rev_hi_lim"]
                .fillna(self.data["funded_amnt"])
            )
            
            # Specific case for the column annual_inc
            # ========== annual_inc ================
            # Use the mean to fill missing values
            self.data["annual_inc"] = self.data["annual_inc"].fillna(
                self.data["annual_inc"].mean()
            )
            
            # All the other are filled with zeros
            for column in columns:
                if column not in ["total_rev_hi_lim", "annual_inc"]:
                    self.data[column] = self.data[column].fillna(0)
        
        # ====== Other cases ======== 
        
        elif case == "fill_with_zeros": 
            
            # Can be filled with zeros 
            for column in columns:
                self.data[column] = self.data[column].fillna(0)

        elif case == "fill_with_mean":
        
            # Can be filled out with the mean of the column
            for column in columns:
                self.data[column] = self.data[column].fillna(self.data[column].mean())
                
        elif case == "fill_with_min":
                
                    # Can be filled out with the mean of the column
                    for column in columns:
                        self.data[column] = self.data[column].fillna(self.data[column].min())
        
        elif case == "fill_with_max":
                
                    # Can be filled out with the mean of the column
                    for column in columns:
                        self.data[column] = self.data[column].fillna(self.data[column].max())
        


        
        
        
        

        

        
    