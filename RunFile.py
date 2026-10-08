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
    data_reader.clear_columns(columns_to_clear=["Unnamed: 0"])
    logger.info(f"Column 'Unnamed: 0' cleared from the dataset.")
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
    

if __name__ == "__main__":
    main()