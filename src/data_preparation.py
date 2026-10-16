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
from sklearn.model_selection import train_test_split

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
        # Dummies are built from the categorical columns only and concatenated back,
        # so the ORIGINAL columns (e.g. 'grade') are kept for WoE/IV
        dummies = pd.get_dummies(
            self.data[categorical_columns], 
            columns=categorical_columns,  # NEW: forces numeric columns (e.g. term_int) to be encoded too
            prefix=categorical_columns, 
            prefix_sep=":", 
            dtype=int
        )
        self.data = pd.concat([self.data, dummies], axis=1)
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
        
        return self.data

class PDdataPreparation:
    
    def __init__(self, data: pd.DataFrame):
            
            """
            The init requires the data to be preprocessed. It should be a pandas DataFrame.
            """
            self.data = data
            
            # NEW: memory of the aggregation done by add_dummy_varibles_similar_WoE (one entry per original variable)
            self.dummy_groups: Dict[str, Dict[str, List[str]]] = {}  # {original_col: {new_dummy: [dummies that were summed]}}
            self.reference_categories: Dict[str, Optional[str]] = {}  # {original_col: reference dummy}
            # NEW: memory of the fine/coarse classing done for the continuous variables (needed to repeat it identically on the test set)
            self.fine_classes: Dict[str, Dict[str, Any]] = {}  # {original_col: {"upper_edges": {label: upper edge}, "integer": bool, "has_missing": bool}}
            self.coarse_classes: Dict[str, Dict[str, Any]] = {}  # {original_col: {"boundaries": [...], "add_missing": bool, "labels": [...]}}
    
    def dependent_variable_definition(self, dependent_var: str, list_defaulted_values: List[str]):
        
        # get a status overview of the different values of the dependent variable
        status_overview = self.data[dependent_var].value_counts() / self.data[dependent_var].count()
        
        # print it out
        print(f"Status overview of the dependent variable {status_overview}")
        
        # AFTER LOOKING AT THE OVERVIEW, THE USER SHOULD BE GIVEN THE OPPORTUNITY TO INPUT A LIST OF NAMES OF VALUES THAT ARE CONSIDER AS DEFAULT
        # THIS SHOULD BE MADE EASY FOR THE USER SO IT CAN GIVE VALUE BY VALUE AND PYTHON SHOULD TRANSFORM IT INTO A LIST
        
        # For now, we will leave it as something that can be given in the parameters
        
        # transform it in defaulted and non-defaulted
        self.data["defaulted_nondefaulted"] = np.where(
            self.data[dependent_var].isin(list_defaulted_values), 
            0, 1 # 0 if defaulted, 1 if non-defaulted
            )

        # get a status overview of the different values of the dependent variable as defaulted and non defaulted
        defaulted_nondefaulted_overview = self.data["defaulted_nondefaulted"].value_counts()/self.data["defaulted_nondefaulted"].count()
        
        # print it out
        print(f"Status overview of the dependent variable as defaulted and non defaulted {defaulted_nondefaulted_overview}")
        
        # Return statement so that the updated data can be used by the next functions
        return self.data
        
    @log_execution
    def train_test_data_split(
        self, 
        test_size_given: float, 
        random_state: int = 42,
        target_col: str = "defaulted_nondefaulted"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Splits the dataset into training and test sets for inputs (features) and output (target).
        
        Parameters:
            test_size_given (float): Proportion of dataset to include in the test split (e.g. 0.2).
            df (pd.DataFrame, optional): Custom DataFrame to split. If None, uses self.data.
            random_state (int): Controls the shuffling applied to the data before applying the split.
            target_col (str): Target column name representing the dependent variable.
            
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
                (data_inputs_train, data_inputs_test, data_output_train, data_output_test)
        """
        # Use provided DataFrame or default to updated instance data (self.data)
        data_to_split = self.data

        if target_col not in data_to_split.columns:
            raise KeyError(
                f"Target column '{target_col}' not found in DataFrame. "
                "Ensure dependent_variable_definition() was run previously."
            )

        # Define the split between test and train for the output variable and the input variables
        data_inputs_train, data_inputs_test, data_output_train, data_output_test = train_test_split(
            data_to_split.drop(target_col, axis=1), 
            data_to_split[target_col], 
            test_size=test_size_given, 
            random_state=random_state,
            stratify=data_to_split[target_col]  # Stratify to preserve target class proportions
        )
        
        # Print the shapes
        print(
            f"Train data inputs shape is: {data_inputs_train.shape}\n"
            f"Test data inputs shape is: {data_inputs_test.shape}\n"
            f"Train data output shape is: {data_output_train.shape}\n"
            f"Test data output shape is: {data_output_test.shape}"
        )

        return data_inputs_train, data_inputs_test, data_output_train, data_output_test

    @staticmethod    
    def WOE_AND_IV(
            df_inputs: pd.DataFrame, 
            input_col_name: str, 
            df_output: pd.DataFrame, # here the output variable, in our case with the info of the defaults
            discrete: bool
    ):
        
        """
        Calculates the Weight of Evidence (WoE) and the Information Value (IV) of a variable.
        
        ########### Weight of Evidence ###########

        The extent to which a categorical independet variable expalins the difference of a categorical outcome variable.

        WoE(i) = ln(%(y=1)(i)/%(y=0)(i)) = ln(%non-defaulted(i)/defaulted(i))

        ===> more positive = higher propensity of the category to bring us to a non-default result 
        ===> more negative = higher propensity of the category to bring us to a defualted result

        Based on the WoE we can coerce different categories into one (when the WoE is similar and the categories are connected: next to each other sequentially)
        
        ########## Information Value ###########

        This is to check the actual value of a variable by adding up the meaning proportions of all its categories.

        How much information the original independent variable brings with respect to expalining the dependent variable

        Information Value (IV) = summatory[(%non-defaulted - %defaulted)*ln(%non-defaulted/defaulted)]
                                = summatory[(%non-defaulted - %defaulted)*WoE]

        We can use this for pre-selection fo variables

        One could also alternatively use PCA (Princiapl component analysis)

        IV ranges:
        - > 0.02 --> no predictive power
        - 0.02 < IV < 0.1 --> weak predictive power
        - 0.1 < IV < 0.3 --> mid predictive power
        - 0.3 < IV < 0.5 --> strong predictive power
        - 0.5 IV --> suspiciously high predictive power

        Paramaters:
            df_inputs: the df including the input variable we want to check
            input_col_name: the name of the input column we would like to check and that contains the categories in which our analysis will be split
            df_output: here the output variable, in our case with the info of the defaults
            
        Returns:
            A data frame that displays several sumamrizing columns including the WoE and the IV.
        """

        # Let's get a data frame in which we only have the output variable and the grades/ratings
        df = pd.concat([df_inputs[input_col_name], df_output], axis=1) # axis=1 concatenate columns

        # ==================== Let's calculate the Weight of Evidence (WoE) ==========================
        
        # Group by grade and count the rows per grade
        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count()

        # We also need the proportion of defaulted/non-defaulted borrowers per group
        # Let's get the proportion of non_defaults and as we have 1 (non default), 0 (default) we can take the average
        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()

        # Merge the two results into one data frame
        df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(), 
                        df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()],
                        axis=1
                        )
        # MISCELLANEOUS ===================================================
        # Not to get two times grade column
        df = df.iloc[:, [0,1,3]]
        # Change the names of the columns
        df.columns = [df.columns.values[0], 'n_obs', 'prop_nondefaulted']
        # ==================================================================

        # Get the necesary calculations to get the WoE ============>

        # Get the proportion of number of observations
        df['prop_n_obs'] = df.n_obs/df.n_obs.sum()
        # Calculate the number of non-defaulted and defaulted borrowers per group
        df['n_nondefaulted'] = df.prop_nondefaulted * df.n_obs
        df['n_defaulted'] = (1-df.prop_nondefaulted) * df.n_obs
        # Calculate the propensity of non-defaulted and defaulted borrowers
        df['prop_n_nondefaulted'] = df.n_nondefaulted / df.n_nondefaulted.sum()
        df['prop_n_defaulted'] = df.n_defaulted / df.n_defaulted.sum()

        # Get the Weight of Evidence ============>
        df['WoE'] = np.log(df['prop_n_nondefaulted']/df['prop_n_defaulted'])
        
        # Condition for DISCRETE VARIABLES
        if discrete:
            df = df.sort_values(['WoE'])
            df = df.reset_index(drop=True)

        # Get the differences of two subsequent rows, abs() to see the net effect
        df['diff_prop_nondefaulted'] = df.prop_nondefaulted.diff().abs()
        df['diff_WoE'] = df.WoE.diff().abs()

        # ==================== Let's calculate the Information Value (IV) ==========================
        df['IV'] = (df.prop_n_nondefaulted - df.prop_n_defaulted) * df.WoE
        df['IV'] = df['IV'].sum() # All categories get the same value, because IV acts per variable
        
        return df

    sns.set()
    # Function to plot the WoE
    @staticmethod
    def plot_by_woe(
        df_WOE: pd.DataFrame, 
        save_path: str, 
        rotation_of_x_axis_labels: int = 0
    ):
        """Parameters:

            df_WOE: df obtained by the function WOE_AND_IV
            save_path: destination file path (e.g.,
            'reports/figures/woe_income.png')
            rotation_of_x_axis_labels: in case we have too many names in the
            X-axis we adjust this

        Returns:
            A saved plot in high quality at the specified save_path
        """

        # Define the plot variables
        x = np.array(df_WOE.iloc[:, 0]).astype(str)  # Get categories as strings
        y = df_WOE["WoE"]

        # Create figure and set size
        fig, ax = plt.subplots(figsize=(18, 6))

        # Plot data
        ax.plot(x, y, marker="o", linestyle="--", color="k")

        # Set labels and title
        ax.set_xlabel(df_WOE.columns[0])
        ax.set_ylabel("Weight of Evidence")
        ax.set_title(f"Weight of Evidence by {df_WOE.columns[0]}")
        ax.tick_params(axis="x", rotation=rotation_of_x_axis_labels)

        # Add grid for improved readability
        ax.grid(True, linestyle=":", alpha=0.6)

        # Ensure target directory exists before saving
        # Save_path is now treated as a DIRECTORY and the file name is built from the variable name
        output_dir = Path(save_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"WoE_{df_WOE.columns[0]}.png"

        # Save plot in publication quality
        plt.savefig(
            file_path,  # NEW: full file path instead of the folder
            dpi=300,  # High resolution (300 DPI for publication quality)
            bbox_inches="tight",  # Fits labels tightly so text/axis don't get cut off
            pad_inches=0.2,  # Adds clean margin padding around the saved image
            transparent=False,  # Solid white background
            format="png",  # PNG lossless image format
        )
        
        # Close the plot memory context to avoid memory leaks during batch loops
        plt.close(fig)
    
    @log_execution
    def add_dummy_varibles_similar_WoE(
        self,
        df_inputs: pd.DataFrame,
        df_WOE: pd.DataFrame,
        original_col: str,
        max_gap_ratio: float = 0.25,
        max_span_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        
        """
        What this function should do is create aggregated dummy variables depending on the following characteristics
        
        WOE_AND_IV function gets us the WoE and the IV in a df for all the distinct values of a variable in the df
        
        Now, these variables were previously split into dummy categories (e.g. grade into ['grade:A', 'grade:B', 'grade:C', 'grade:D', 'grade:E', 'grade:F', 'grade:G'])
        
        Some categories are relevant for the analisis, but some do not add enough value so they should be combined with others
        
        Lets give an example using the original variable: home_ownership
        
        The dummy variables derived from it were: "home_ownership:RENT", "home_ownership:OTHER", "home_ownership:NONE", "home_ownership:ANY", "home_ownership:OWN", "home_ownership:MORTGAGE"
        
        It was reviewed in a Jupyter Notebook it was determined that this categories were to be combined into one:
        df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_prepr["home_ownership:RENT"],
                                                            df_inputs_prepr["home_ownership:OTHER"],
                                                            df_inputs_prepr["home_ownership:NONE"],
                                                            df_inputs_prepr["home_ownership:ANY"]])
        Leaving us with new variables for home_ownership: ['home_ownership:RENT_OTHER_NONE_ANY', 'home_ownership:MORTGAGE', 'home_ownership:OWN']
        
        And then determining that given that it was the category with the lowest WoE overall, that 'home_ownership:RENT_OTHER_NONE_ANY' was to be set as the reference category
        
        THIS FUNCTION, should find out which categories (dummies) for the variables can be added up (smashed together)
        
        The criteria should be:
        - Categories that are smashed up are to be find next to each other in a escale of WoE
        - The continuous difference in the weight of evidence for the smashed up categories is minimal in comparison to maximal difference between two consecutive categories (and also between the minimal and the maximal) --> Create a rule here
        - The number of categories to be smashed up has no maximal or minimal, but in the end, the original variable must consist of at least two dummies
        - The original variable should not be deleted
        - The function should find gaps, I mean, example a variable has dummies (ordered by WoE from lower to higher value): A, B, C, D, E, F ---> It can match up A-B together, find out that C and D belong to categories fo themselves and then also match up E-F, this is allowed
        - The explicit criteria should be explicitely defined here and comemnted
        
        The inputs of the function should be:
        - The df (in this case it will be in the RunFile the one name as data_inputs, but dont declare it explicitely here)
        - WOE_AND_IV df result
        - The original column from which the dummies were derived --> The function then will look for the dummies generates from this variable and then using the df from WOE_AND_IV function, it see what can be matched up
        
        The output of this function should be 
        - the df having the newly added variable for a specific variable, while deleting the variables that we used in the sum up (not the others).
        - while also returning which is the reference category for that variable
        
        Important, as this will embdedded in a loop for all variables, it is important that the function can rely on "self" and that each iteration contains the adjustments to the df (the matched up newly created dummy category for an specific variable, while deleting the dummy categories that created it)
        
        

        ================== CRITERIA IMPLEMENTED (explicit rules) ==================
        
        Notation: the categories of the variable are ordered by WoE (lowest to highest) -> w_0 <= w_1 <= ... <= w_k
        
        gap_j       = w_j - w_(j-1)               difference between two CONSECUTIVE categories
        max_gap     = max(gap_j)                  biggest difference between two consecutive categories
        total_range = w_k - w_0                   difference between the minimal and the maximal WoE
        span        = w_j - w_(first of group)    difference between the minimal and the current WoE inside a group
        
        Category j is added to the group of category j-1 ONLY IF BOTH conditions hold:
        
        RULE 1 (consecutive difference is minimal compared with the maximal consecutive difference):
                gap_j <= max_gap_ratio * max_gap
        RULE 2 (the whole group stays minimal compared with the min-max difference, this avoids "chaining" A-B, B-C, C-D... into a huge group):
                span <= max_span_ratio * total_range
        
        Otherwise category j opens a new group. This gives these properties:
        - Only categories that are NEXT to each other in the WoE scale are matched (groups are contiguous)
        - Several groups (and several single categories) can coexist: e.g. A-B together, C alone, D alone, E-F together
        - Since max_gap_ratio < 1, the biggest gap is NEVER bridged, so the variable always keeps at least 2 dummies
        - A category with a non finite WoE (no defaults or no non-defaulted observations -> +-inf) is NOT matched, it is kept 
          as it is and a warning is logged, because its WoE can not be compared (the user should review it)
        - The reference category is the group with the lowest WoE (the first one in the ordered scale)
        - The original variable (e.g. 'home_ownership') is never deleted, only the dummies that were summed up
        - max_gap_ratio and max_span_ratio are starting points, they should be calibrated looking at the WoE plots
        
        Parameters:
            df_inputs: the df with the original variable and its dummies, it is NOT modified, a new df is returned
            df_WOE: result of WOE_AND_IV for original_col
            original_col: the original variable from which the dummies ("original_col:category") were derived
            max_gap_ratio: RULE 1 (0 < ratio < 1)
            max_span_ratio: RULE 2
        
        Returns:
            (df_inputs with the aggregated dummies, name of the reference dummy category or None)
        """
        
        # ---- 0. Safety checks ----
        if df_WOE.columns[0] != original_col:
            raise ValueError(f"df_WOE belongs to '{df_WOE.columns[0]}' but original_col is '{original_col}'.")
        if not 0 < max_gap_ratio < 1:
            raise ValueError("max_gap_ratio must be between 0 and 1 (exclusive), otherwise the variable could collapse into one dummy.")

        # The dummies were created as "<original_col>:<category>" (see DataPreprocessor.create_dummy_variables)
        dummy_of = {c: f"{original_col}:{c}" for c in df_WOE[original_col]}
        missing = [d for d in dummy_of.values() if d not in df_inputs.columns]
        if missing:
            raise KeyError(f"Dummy columns not found in df_inputs: {missing}. Were they already aggregated for '{original_col}'?")

        # ---- 1. Order the categories by WoE and separate the ones with a non finite WoE ----
        woe = df_WOE.sort_values("WoE")
        is_finite = np.isfinite(woe["WoE"])
        if (~is_finite).any():
            logger.warning("'%s': categories with non finite WoE are kept as they are (review them): %s",
                           original_col, woe.loc[~is_finite, original_col].tolist())
        woe = woe[is_finite]
        categories = woe[original_col].tolist()
        w = woe["WoE"].to_numpy()

        if len(categories) == 0:
            logger.warning("'%s': no category with a finite WoE, nothing was aggregated", original_col)
            self.dummy_groups[original_col] = {}
            self.reference_categories[original_col] = None
            return df_inputs, None

        # ---- 2. Build the groups of contiguous categories with similar WoE ----
        # NEW: the RULE 1 / RULE 2 logic now lives in _group_similar_woe (shared with coarse_classing), the result is the same as before
        groups = [[categories[i] for i in g] for g in self._group_similar_woe(w, max_gap_ratio, max_span_ratio)]

        # Guarantee at least two dummies (only reachable if all the WoE are identical)
        if len(groups) < 2:
            groups = [[c] for c in categories]

        # ---- 3. Names of the resulting dummies and of the reference category ----
        group_names = [
            dummy_of[g[0]] if len(g) == 1 else f"{original_col}:" + "_".join(str(c) for c in g)
            for g in groups
        ]
        # Groups are ordered by WoE and a group WoE always lies between the WoE of its members, so the first group has the lowest WoE
        reference_category = group_names[0]

        # ---- 4. Apply: add the summed dummies, delete only the dummies that were used in the sum ----
        merged = {name: [dummy_of[c] for c in g] for name, g in zip(group_names, groups) if len(g) > 1}
        if merged:
            new_columns = pd.DataFrame({name: df_inputs[cols].sum(axis=1) for name, cols in merged.items()})
            used_columns = [c for cols in merged.values() for c in cols]
            df_inputs = pd.concat([df_inputs.drop(columns=used_columns), new_columns], axis=1)

        # ---- 5. Keep a memory in self so every loop iteration (and a later re-application) knows what was done ----
        self.dummy_groups[original_col] = merged
        self.reference_categories[original_col] = reference_category
        logger.info("'%s': %d categories -> %d dummies. Aggregated: %s. Reference category: %s",
                    original_col, len(categories), len(groups), merged if merged else "none", reference_category)

        return df_inputs, reference_category

    # =====================================================================================================
    # FINE AND COARSE CLASSING FOR CONTINUOUS VARIABLES
    # Flow per variable:  fine_classing -> WOE_AND_IV(discrete=False) on '<col>_factor' -> plot_by_woe -> coarse_classing
    # =====================================================================================================

    # Shared grouping rule: used by add_dummy_varibles_similar_WoE (discrete) and by coarse_classing (continuous)
    @staticmethod
    def _group_similar_woe(w, max_gap_ratio: Optional[float], max_span_ratio: float) -> List[List[int]]:
        """
        Groups CONTIGUOUS positions of the array w (WoE values, in the order given) following:
        RULE 1: |w_j - w_(j-1)| <= max_gap_ratio * (biggest consecutive gap) --> (skipped if max_gap_ratio is None)
        RULE 2: (max - min WoE of the group including j) <= max_span_ratio * (max - min WoE of all categories)
        For WoE sorted ascending (discrete case) this is exactly the rule described in add_dummy_varibles_similar_WoE.
        Returns a list with the positions of every group.
        """
        
        # Determine the max gap and the total range
        w = np.asarray(w, dtype=float)
        gaps = np.abs(np.diff(w))
        max_gap = gaps.max() if len(gaps) else 0.0
        total_range = w.max() - w.min()

        # Create the groupings ==> This is basically packing categories of a varible by out set of rules (1,2) so that they can be combined into one WoE-significant category
        groups = [[0]]
        group_min = group_max = w[0]
        for j in range(1, len(w)):
            rule_1 = True if max_gap_ratio is None else gaps[j - 1] <= max_gap_ratio * max_gap
            rule_2 = (max(group_max, w[j]) - min(group_min, w[j])) <= max_span_ratio * total_range
            if rule_1 and rule_2:
                groups[-1].append(j)
                group_min, group_max = min(group_min, w[j]), max(group_max, w[j])
            else:
                groups.append([j])
                group_min = group_max = w[j]
        return groups

    # Continuous only: groups with too few observations have an unreliable WoE, so they are merged with a neighbour
    @staticmethod
    def _merge_small_groups(groups, prop_obs, prop_nd, prop_d, min_prop_obs: float) -> List[List[int]]:
        """
        A group with less than min_prop_obs of ALL the observations is merged with its neighbour that has the closest
        WoE (the group WoE is recalculated from prop_n_nondefaulted / prop_n_defaulted). It stops when only 2 groups are left.
        """
        
        # Get the proprotion of non-defaults vs defaults
        def group_woe(g):
            nd, d = prop_nd[g].sum(), prop_d[g].sum()
            return np.log(nd / d) if nd > 0 and d > 0 else np.nan

        # Get the groups gathered by the _group_similar_woe
        groups = [list(g) for g in groups]
        while len(groups) > 2:
            # Get the share of observations
            shares = [prop_obs[g].sum() for g in groups]
            i = int(np.argmin(shares))
            # If the shares are large enoughm break the function
            if shares[i] >= min_prop_obs:
                break
            # Otherwise apply the proportion of non-defaults to defaults
            woes = [group_woe(g) for g in groups]

            # Get the distance between WoE
            def distance(k):
                return abs(woes[k] - woes[i]) if np.isfinite(woes[k]) and np.isfinite(woes[i]) else np.inf

            # Group a group with few observations into its neighboring group
            k = min((k for k in (i - 1, i + 1) if 0 <= k < len(groups)), key=lambda k: (distance(k), -shares[k]))
            lo, hi = sorted((i, k))
            groups[lo] = groups[lo] + groups[hi]
            # Delete the original groups that compund the new group
            del groups[hi]
        return groups

    # Readable and unambiguous names: intervals are (a, b], i.e. 'a-b' means "greater than a and up to b"
    @staticmethod
    def _coarse_labels(boundaries: List[float], integer: bool) -> List[str]:
        f = lambda v: f"{v:.10g}"
        if integer and all(float(b).is_integer() for b in boundaries):
            # integer variable: '38-39' means the values 38 and 39 (as in the original notebook)
            middle = [f(lo + 1) if hi - lo == 1 else f"{f(lo + 1)}-{f(hi)}" for lo, hi in zip(boundaries[:-1], boundaries[1:])]
        else:
            middle = [f"{f(lo)}-{f(hi)}" for lo, hi in zip(boundaries[:-1], boundaries[1:])]
        return [f"<={f(boundaries[0])}"] + middle + [f">{f(boundaries[-1])}"]

    # FINE CLASSING: splits a continuous variable in n_bins intervals of the same width and stores them in '<col>_factor'
    @log_execution
    def fine_classing(
        self,
        df_inputs: pd.DataFrame,
        original_col: str,
        n_bins: int = 50,
        upper_cap: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Adds the column '<original_col>_factor' (ordered categories, one per non-empty interval, in the order of the variable)
        so WOE_AND_IV(discrete=False) and plot_by_woe can be used on it. The df_inputs received is NOT modified.
        
        - n_bins: number of intervals of the same width (like pd.cut(x, n_bins))
        - upper_cap: for variables where one interval holds almost all the data (annual_inc, dti): the n_bins intervals are
          built up to this value and everything above it goes into ONE last interval
        - Missing values (NaN) get their own category 'Missing' (placed at the end), they are never mixed with the intervals
        """
        
        # Original column
        values = df_inputs[original_col]
        # Drop NAs
        non_null = values.dropna()
        # Min and max values
        lo, hi = non_null.min(), non_null.max()
        if lo == hi:
            raise ValueError(f"'{original_col}' is constant ({lo}), it can not be classed.")
        if upper_cap is not None and upper_cap <= lo:
            raise ValueError(f"upper_cap ({upper_cap}) must be bigger than the minimum of '{original_col}' ({lo}).")
        # If the upper_cap is strictly lower than the maximum value of the variable
        if upper_cap is not None and upper_cap < hi:
            edges = np.append(np.linspace(lo, upper_cap, n_bins + 1), hi)
        # If the upper_cap is equal to the maximum value of the variable
        else:
            edges = np.linspace(lo, hi, n_bins + 1)

        # Include_lowest so the minimum belongs to the first interval; empty intervals are removed
        cut = pd.cut(values, bins=edges, include_lowest=True).cat.remove_unused_categories()
        intervals = list(cut.cat.categories)
        labels = [f"({iv.left:.8g}, {iv.right:.8g}]" for iv in intervals]
        # Create the factors
        factor = pd.Series(pd.Categorical.from_codes(cut.cat.codes, categories=labels, ordered=True), index=df_inputs.index)

        # Check for still any NAs
        has_missing = bool(values.isna().any())
        if has_missing:
            factor = factor.cat.add_categories("Missing").fillna("Missing")

        # Memory for coarse_classing
        self.fine_classes[original_col] = {
            "upper_edges": dict(zip(labels, [iv.right for iv in intervals])),
            "integer": bool((non_null % 1 == 0).all()),
            "has_missing": has_missing,
        }
        logger.info("'%s': %d non-empty intervals%s%s", original_col, len(labels),
                    f" (upper cap {upper_cap})" if upper_cap is not None else "", ", plus 'Missing'" if has_missing else "")
        return df_inputs.assign(**{f"{original_col}_factor": factor})

    # COARSE CLASSING: creates the final dummies of a continuous variable, either proposed automatically or from given boundaries
    @log_execution
    def coarse_classing(
        self,
        df_inputs: pd.DataFrame,
        df_WOE: Optional[pd.DataFrame],
        original_col: str,
        df_output: Optional[pd.Series] = None,
        method: Literal["auto", "manual"] = "auto",
        boundaries: Optional[List[float]] = None,
        max_span_ratio: float = 0.15,
        min_prop_obs: float = 0.05,
        add_missing: Optional[bool] = None,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        Creates the dummies '<original_col>:<interval>' from the ORIGINAL variable, so every row belongs to exactly one of them
        (checked). The original variable is kept and the '<original_col>_factor' column is removed. Returns
        (df_inputs with the new dummies, reference category = the dummy with the lowest WoE, or None if df_output is not given).
        
        method="auto":   df_WOE is the WOE_AND_IV result of '<original_col>_factor'. Consecutive fine intervals are grouped with
                         RULE 2 (group span <= max_span_ratio * total WoE range) and then every group with less than
                         min_prop_obs of the observations is merged with the neighbour with the closest WoE.
                         RULE 1 (gap vs biggest gap) is NOT used here: on a smooth trend all the steps are similar to the
                         biggest one, so it would prevent any grouping (34 groups out of 50 in a test).
        method="manual": the boundaries are given (upper limit of every interval but the last one, intervals are (a, b]),
                         df_WOE is not needed. This is also how the SAME classing is repeated on the test set
                         (boundaries = self.coarse_classes[col]["boundaries"], add_missing = self.coarse_classes[col]["add_missing"]).
        """
        
        # In case another method is chosen
        if method not in ("auto", "manual"):
            raise ValueError("method must be 'auto' or 'manual'")
        
        # Retrieve the factor columns got from fine classing and and the original columns
        factor_col = f"{original_col}_factor"
        x = df_inputs[original_col]
        integer = bool((x.dropna() % 1 == 0).all())

        # ---- 1. Boundaries ----
        if method == "auto":
            if df_WOE is None or df_WOE.columns[0] != factor_col or original_col not in self.fine_classes:
                raise ValueError(f"method='auto' needs fine_classing('{original_col}') and the WOE_AND_IV result of '{factor_col}'.")
            woe = df_WOE.reset_index(drop=True)
            numeric = woe[woe[factor_col].astype(str) != "Missing"].reset_index(drop=True) # 'Missing' is never grouped with intervals
            w = numeric["WoE"].where(np.isfinite(numeric["WoE"])).ffill().bfill().to_numpy() # a non finite WoE joins its neighbour
            if np.isnan(w).all():
                logger.warning("'%s': no interval with a finite WoE, coarse classing skipped", original_col)
                return df_inputs, None
            groups = self._group_similar_woe(w, None, max_span_ratio)
            groups = self._merge_small_groups(groups, numeric["prop_n_obs"].to_numpy(), numeric["prop_n_nondefaulted"].to_numpy(),
                                              numeric["prop_n_defaulted"].to_numpy(), min_prop_obs)
            upper_edges = self.fine_classes[original_col]["upper_edges"]
            proposed = [upper_edges[str(numeric[factor_col].iloc[g[-1]])] for g in groups[:-1]]
            # integer variables: the floor gives exactly the same split; others: 4 significant digits to get readable names
            boundaries = [np.floor(b) if integer else float(f"{b:.4g}") for b in proposed]
        elif not boundaries:
            raise ValueError("method='manual' needs the list of boundaries.")
        boundaries = sorted(set(float(b) for b in boundaries))
        if not boundaries:
            logger.warning("'%s': only one interval, nothing to class", original_col)
            return df_inputs, None

        # ---- 2. Dummies (intervals are (a, b], the first one starts at -inf and the last one ends at +inf) ----
        labels = self._coarse_labels(boundaries, integer)
        coarse = pd.cut(x, bins=[-np.inf] + boundaries + [np.inf], labels=labels)
        has_missing = bool(x.isna().any()) if add_missing is None else add_missing
        if has_missing:
            coarse = coarse.cat.add_categories("Missing").fillna("Missing")
        dummies = pd.get_dummies(coarse, prefix=original_col, prefix_sep=":", dtype=int)

        # Every row must be in exactly ONE category, otherwise it would silently fall into the reference category
        not_covered = int((dummies.sum(axis=1) != 1).sum())
        if not_covered:
            raise ValueError(f"'{original_col}': {not_covered} rows belong to no category (missing values without a 'Missing' category?).")
        df_inputs = pd.concat([df_inputs.drop(columns=[factor_col], errors="ignore"), dummies], axis=1)

        # ---- 3. Reference category (lowest WoE of the final categories) ----
        reference_category = self.reference_categories.get(original_col)
        if df_output is not None:
            woe_after = self.WOE_AND_IV(df_inputs=pd.DataFrame({original_col: coarse}, index=x.index),
                                        input_col_name=original_col, df_output=df_output, discrete=False)
            finite = woe_after[np.isfinite(woe_after["WoE"])]
            reference_category = f"{original_col}:{finite.loc[finite['WoE'].idxmin(), original_col]}" if len(finite) else None
            logger.info("'%s': IV before coarse classing: %s | after: %.4f", original_col,
                        f"{df_WOE['IV'].iloc[0]:.4f}" if df_WOE is not None else "n/a", woe_after["IV"].iloc[0])

        # ---- 4. Memory ----
        self.coarse_classes[original_col] = {"boundaries": boundaries, "add_missing": has_missing, "labels": labels}
        self.reference_categories[original_col] = reference_category
        logger.info("'%s': %d categories %s. Reference category: %s", original_col, dummies.shape[1],
                    list(dummies.columns.str.replace(f"{original_col}:", "", regex=False)), reference_category)
        return df_inputs, reference_category