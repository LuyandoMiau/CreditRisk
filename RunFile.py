import logging
import os
import yaml
from pathlib import Path
from src.data_preparation import DataReader, DataExplorerAdvanced, DataPreprocessor

# Set up logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the path to the Parameters.yaml file
def load_parameters(config_path: str = "src/config/Parameters.yaml") -> dict:
    """Load configuration parameters from a YAML file."""
    with open(config_path, "r", encoding="utf-8") as file:
        parameters = yaml.safe_load(file)
    return parameters

def main():
    
    
    """==============================SCRIPTS LOCATION============================="""
    script_dir = Path(__file__).parent
    
    
    """==============================PARAMETERS LOADING============================="""
    # Set it with respect to the script's location
    params = load_parameters("src/config/Parameters.yaml")
    
    
    """===========================DATA PREPARATION PIPELINE=========================="""
    
    # ============================== DataReader ==============================
    
    # 1. Read the file path and parameters from the loaded configuration
    raw_data_path = params["data_preparation"]["data_reader"]["file_path"]
    # 2. Use the function to read the data
    data_reader = DataReader(file_path=raw_data_path)
    data_reader.read_data()
    logger.info(f"Data read from {raw_data_path} with shape: {data_reader.data.shape}")
    # 3. Clear an unnecessary column (e.g., "Unnamed: 0") if it exists
    columns_to_drop = params["data_preparation"]["data_reader"]["columns_to_drop"]
    data_reader.clear_columns(columns_to_clear=columns_to_drop)
    logger.info(f"Columns {columns_to_drop} cleared from the dataset.")
    # 4. Create a copy of the initial DataFrame for further processing
    df_initial = data_reader.copy_data()
    logger.info(f"Initial DataFrame created with shape: {df_initial.shape}")

    # ============================== DataExplorerAdvanced ==============================
    
    # 1. Get the paths for SQLite database and table name from the configuration
    db_path = params["data_exploration"]["data_explorer_advanced_sqlite"]["db_path"]
    table_name = params["data_exploration"]["data_explorer_advanced_sqlite"]["table_name"]
    DataExplorerAdvanced(data=df_initial).explore_data()
    logger.info("Basic data exploration completed, see the logger")
    DataExplorerAdvanced(data=df_initial).save_to_sqlite(db_path=db_path,table_name=table_name)
    logger.info("Data exploration completed and results stored in SQLite database in %s.", db_path)
    
    # ============================== DataPreprocessor ==============================
    
    # 0. Get the defined reference date for calculating days and months since the earliest credit line
    defined_reference_date = params["data_preparation"]["data_preprocessor"]["reference_date"]
    # Parse it to a datetime object if needed, or keep it as a string for later use
    defined_reference_date = DataPreprocessor.parse_reference_date(defined_reference_date)
    
    # 1. Get the list of columns from the df_initial DataFrame
    data_preprocessor = DataPreprocessor(data=df_initial)
    column_names = data_preprocessor.column_names()
    logger.info(f"Columns in the initial DataFrame: {column_names}")
    
    # 2. Transform the columns accordingly
    
    # Get th list with all transformations from the Parameters.yaml file, for example:
    transformations = params["data_preparation"]["data_preprocessor"]["transformations"]
    
    # FIRST COLUMN ---- emp_length_int
    first_trans = transformations[0] 
    trans_type = first_trans["type"] # "String_to_Numeric_specif"
    cols = first_trans["columns"] # ["emp_length"]
    new_cols = first_trans["new_column_names"] # ["emp_length_int"]
    data_preprocessor.transform_columns_types(transformation_type=trans_type, columns=cols, new_column_names=new_cols)
    logger.info(f"Column 'emp_length' transformed to 'emp_length_int' with type {trans_type}.")
    
    # SECOND COLUMN ---- earliest_cr_line_date
    third_trans = transformations[2]
    trans_type = third_trans["type"] # "String_to_Datetime"
    cols = third_trans["columns"] # ["earliest_cr_line"]
    new_cols = third_trans["new_column_names"] # ["earliest_cr_line_date"]
    data_preprocessor.transform_columns_types(transformation_type=trans_type, columns=cols, new_column_names=new_cols)
    logger.info(f"Column 'earliest_cr_line' transformed to 'earliest_cr_line_date' with type {trans_type}.")
    
    # Get days and months since the earliest credit line
    data_preprocessor.get_days_months_since_earliest_credit_line(defined_reference_date=defined_reference_date)
    logger.info(f"Days and months since the earliest credit line calculated using reference date: {defined_reference_date}.")

    # Handle negative values of days and months since the earliest credit line
    data_preprocessor.handle_negative_values_months_since_earliest_credit_line()
    logger.info("Negative values in 'months_since_earliest_credit_line' handled.")
    
    # THIRD COLUMN ---- term
    second_trans = transformations[1]
    trans_type = second_trans["type"] # "String_to_Numeric_specif2"
    cols = second_trans["columns"] # ["term"]
    new_cols = second_trans["new_column_names"] # ["term_int"]
    data_preprocessor.transform_columns_types(transformation_type=trans_type, columns=cols, new_column_names=new_cols)
    logger.info(f"Column 'term' transformed to 'term_int' with type {trans_type}.")
    
    # FOURTH COLUMN ---- issue_d
    fourth_trans = transformations[3]
    trans_type = fourth_trans["type"] # "String_to_Datetime"
    cols = fourth_trans["columns"] # ["issue_d"]
    new_cols = fourth_trans["new_column_names"] # ["issue_d_date"] 
    data_preprocessor.transform_columns_types(transformation_type=trans_type, columns=cols, new_column_names=new_cols)
    logger.info(f"Column 'issue_d' transformed to 'issue_d_date' with type {trans_type}.")
    
    # Get days and months since the issue date
    data_preprocessor.get_days_months_since_issue_date(defined_reference_date=defined_reference_date)
    logger.info(f"Days and months since the issue date calculated using reference date: {defined_reference_date}.")
    


if __name__ == "__main__":
    main()