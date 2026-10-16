import logging
import os
import yaml
from pathlib import Path
from src.data_preparation import DataReader, DataExplorerAdvanced, DataPreprocessor, PDdataPreparation

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
    
    # 3. Create the dummy variables for the categorical columns
    categorical_variables = params["data_preparation"]["data_preprocessor"]["categorical_columns"]
    data_preprocessor.create_dummy_variables(categorical_columns=categorical_variables)
    logger.info("Dummy variables created for categorical columns.")
    
    # 4. Handle NAs
    empty_values_handler = params["data_preparation"]["data_preprocessor"]["handle_empty_values"]
    columns_to_handle = empty_values_handler["columns"]
    case = empty_values_handler["case"]
    preprocessed_data = data_preprocessor.handle_empty_values(columns=columns_to_handle, case=case)
    logger.info(f"NAs handled for the columns {columns_to_handle} using the case {case}.")
    
    # ============================== PDdataPreparation ==============================
    
    # 0. Initialize the class
    PD_data_prep = PDdataPreparation(data=data_preprocessor.data)
    
    # 1. Set up the dependent variable
    pd_dependent_var = params["data_preparation"]["pd_data_preparation"]["dependent_variable"]
    pd_list_default_values = params["data_preparation"]["pd_data_preparation"]["list_defaulted_values"]
    
    PD_data_prep.dependent_variable_definition(dependent_var=pd_dependent_var, list_defaulted_values=pd_list_default_values)
    logger.info(
        f"Summaries of the dependent variable were displayed, being the original values "
        f"{list(preprocessed_data[pd_dependent_var].unique())} and the list of considered default values {pd_list_default_values}"
    )
    
    # 2. Split the data between train and test
    test_data_size = params["data_preparation"]["pd_data_preparation"]["test_data_size"]
    
    # Method uses self.data automatically without parameter conflicts
    data_inputs_train, data_inputs_test, data_output_train, data_output_test = PD_data_prep.train_test_data_split(test_size_given=test_data_size)
    
    # 3. Run WoE and IV for several variables and get their plot
    
    # ===== NOTE: We can switch between TRAIN and TEST variables ====
    data_inputs = data_inputs_train
    data_output = data_output_train
    
    pd_params = params["data_preparation"]["pd_data_preparation"]
    savepathWoE_plots = pd_params["WoE_plot_saving_path"]
    merge_rules = pd_params["WoE_merge_rules"]
    
    # ============ DISCRETE VARIABLES ========================= #
    
    # Define a list for the reference categories:
    reference_categories_dict_discrete = {}
    
    # Loop over every original variable defined in Parameters.yaml
    for col in pd_params["columns_to_check_WoE_discrete"]:
        # WoE and IV
        col_WoE_IV = PD_data_prep.WOE_AND_IV(df_inputs=data_inputs, input_col_name=col, df_output=data_output, discrete=True)
        # Plot before aggregating, so the plot shows all the original categories
        PD_data_prep.plot_by_woe(df_WOE=col_WoE_IV, save_path=savepathWoE_plots, rotation_of_x_axis_labels=0)
        logger.info(f"WoE, IV and WoE plot save for the variable: {col} (IV = {col_WoE_IV['IV'].iloc[0]:.4f})")
        # Aggregate similar dummies. data_inputs is re-assigned so the next iteration starts from the adjusted df
        data_inputs, reference_category = PD_data_prep.add_dummy_varibles_similar_WoE(
            df_inputs=data_inputs, df_WOE=col_WoE_IV, original_col=col,
            max_gap_ratio=merge_rules["max_gap_ratio"], max_span_ratio=merge_rules["max_span_ratio"],
        )
        logger.info(f"Reference category for {col}: {reference_category}")
        reference_categories_dict_discrete[col] = reference_category
        
    logger.info(f"The complete dictionary of reference categories is: {reference_categories_dict_discrete}")
    
    # GET FULL LIST OF COLUMNS OF THE CURRENT UPDATED DF
    # One column per line, with its position (easiest to read)
    print(*[f"{i}: {c}" for i, c in enumerate(data_inputs.columns)], sep="\n")
    
    # ============ CONTINOUS VARIABLES ========================= #
        
    # Define a dictionary for the reference categories:
    reference_categories_dict_continous = {}
    
    # Defaults for all the continuous variables, each variable can override them in Parameters.yaml
    continous_defaults = pd_params["WoE_continous_defaults"]
        
    # Loop over every original variable defined in Parameters.yaml
    for col, options in pd_params["columns_to_check_WoE_continous"].items():
        # Options of this variable = defaults + the ones defined for this variable in Parameters.yaml
        opts = {**continous_defaults, **(options or {})}
        
        # FINE CLASSING: splits the variable in n_bins intervals and stores them in '<col>_factor'
        data_inputs = PD_data_prep.fine_classing(df_inputs=data_inputs, original_col=col, n_bins=opts["n_bins"], upper_cap=opts.get("upper_cap"))
        factor_col = f"{col}_factor"
        # WoE and IV of the fine classes
        col_WoE_IV = PD_data_prep.WOE_AND_IV(df_inputs=data_inputs, input_col_name=factor_col, df_output=data_output, discrete=False)
        # Plot before the coarse classing, so the plot shows all the fine classes
        PD_data_prep.plot_by_woe(df_WOE=col_WoE_IV, save_path=savepathWoE_plots, rotation_of_x_axis_labels=90)
        logger.info(f"WoE, IV and WoE plot save for the variable: {factor_col} (IV = {col_WoE_IV['IV'].iloc[0]:.4f})")
        
        # COARSE CLASSING ("none" = stop here and review the plot first)
        if opts["coarse_method"] == "none":
            logger.info(f"Coarse classing skipped for {col}: review the plot and define coarse_boundaries in Parameters.yaml")
            continue
        # Creates the final dummies. data_inputs is re-assigned so the next iteration starts from the adjusted df
        data_inputs, reference_category = PD_data_prep.coarse_classing(
            df_inputs=data_inputs, df_WOE=col_WoE_IV, original_col=col, df_output=data_output,
            method=opts["coarse_method"], boundaries=opts.get("coarse_boundaries"),
            max_span_ratio=opts["max_span_ratio"], min_prop_obs=opts["min_prop_obs"],
        )
        logger.info(f"Reference category for {col}: {reference_category}")
        reference_categories_dict_continous[col] = reference_category
        
    logger.info(f"The complete dictionary of reference categories is: {reference_categories_dict_continous}")
    
    # GET FULL LIST OF COLUMNS OF THE CURRENT UPDATED DF
    # One column per line, with its position (easiest to read)
    print(*[f"{i}: {c}" for i, c in enumerate(data_inputs.columns)], sep="\n")


if __name__ == "__main__": 
    main()