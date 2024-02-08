from dotenv import load_dotenv
import os
from pymongo import MongoClient
import pandas as pd
import glob

load_dotenv()
cluster_uri = os.environ.get("MONGODB_URI")
client = MongoClient(cluster_uri)
db = client["MSCI446_DB"]

omit_dict = {
    "Gen_Outages": ["forecast_execution_date_ept"],
    "Historical_DA_Prices": ["row_is_current", "version_nbr", "datetime_beginning_utc"],
    "Load_Forecast": ["evaluated_at_utc", "forecast_hour_beginning_utc", "evaluated_at_ept"],
    "Solar_Forecast": ["datetime_beginning_utc"],
    "Wind_Forecast": ["datetime_beginning_utc"]
}


def iterate_over_root(folder_path="ETL/Data") -> None:
    """
    Iterates through each subfolder and CSV file, applying a callback function to each CSV file.

    Args:
        folder_path (str): Path to the folder containing subfolders with CSV files.
    """
    for subdir in os.listdir(folder_path):
        subdir_path = os.path.join(folder_path, subdir)

        # Skip if not a directory
        if not os.path.isdir(subdir_path):
            continue

        # Find all CSV files in the current subfolder
        csv_files = glob.glob(os.path.join(subdir_path, '*.csv'))

        # Print or otherwise use the subdirectory name
        print(f"Processing subdirectory: {subdir}")

        cur_collection = db[subdir]

        cur_omissions = omit_dict[subdir]

        # Example: Print found CSV files
        for csv_file in csv_files:
            etl_process(csv_file, cur_collection, cur_omissions)


def extract(csv_file_path: str, omissions: list) -> pd.DataFrame:
    """
    Extracts data from a CSV file and removes specified columns.

    Args:
        csv_file_path (str): Path to the CSV file.
        omissions (list of str): List of column names to be omitted from the DataFrame.

    Returns:
        pd.DataFrame: DataFrame after removing specified columns.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Remove specified columns
    df.drop(columns=omissions, inplace=True, errors='ignore')

    return df


def transform(data_frame: pd.DataFrame) -> list:
    """
    Converts a pandas DataFrame into a list of dictionaries, suitable for MongoDB insertion.

    Each dictionary in the list represents a row in the DataFrame, with column names as keys.

    Args:
        data_frame (pd.DataFrame): The DataFrame to be transformed.

    Returns:
        list: A list of dictionaries, where each dictionary represents a row from the DataFrame.
    """
    return data_frame.to_dict("records")


def load(data_dicts: list, collection: MongoClient) -> None:
    """
    Inserts a list of dictionaries into a specified MongoDB collection.

    This function uses MongoDB's `insert_many` method for bulk insertion, 
    which is more efficient than inserting documents one at a time.

    Args:
        data_dicts (list): List of dictionaries to be inserted into the MongoDB collection.
        collection (MongoClient.collection): The MongoDB collection where documents will be inserted.
    """
    # Perform bulk insertion of the list of dictionaries into the MongoDB collection
    collection.insert_many(data_dicts)


def etl_process(csv_file: str, collection: MongoClient, omissions: list) -> None:
    """
    Performs the ETL (Extract, Transform, Load) process for a single CSV file.

    The process involves reading the CSV file, optionally removing specified columns,
    transforming the DataFrame into a list of dictionaries, and loading the data into MongoDB.

    Args:
        csv_file (str): Path to the CSV file to be processed.
        collection (MongoClient.collection): The MongoDB collection where data will be loaded.
        omissions (list): List of column names to be omitted from the DataFrame before processing.
    """
    # Extract data from the CSV file, omitting specified columns
    data = extract(csv_file, omissions)

    # Transform the DataFrame into a list of dictionaries
    data_dicts = transform(data)

    # Load the transformed data into the specified MongoDB collection
    load(data_dicts, collection)


def main() -> None:
    """
    Main function to orchestrate the ETL process across multiple CSV files.

    This function calls `iterate_over_root` to process CSV files located within
    a structured folder hierarchy, applying the ETL process to each file.
    """
    # Initiate the ETL process across all CSV files in the specified directory structure
    iterate_over_root()


if __name__ == "__main__":
    main()
