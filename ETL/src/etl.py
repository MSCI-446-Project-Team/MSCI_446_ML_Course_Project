from dotenv import load_dotenv
import os
from pymongo import MongoClient
import pandas as pd
import glob
import argparse


class Etl_Tool():
    load_dotenv()
    __cluster_uri = os.environ.get("MONGODB_URI")
    __omit_dict = {
        "Gen_Outages": ["forecast_execution_date_ept"],
        "Historical_DA_Prices": ["row_is_current", "version_nbr", "datetime_beginning_utc"],
        "Load_Forecast": ["evaluated_at_utc", "forecast_hour_beginning_utc", "evaluated_at_ept"],
        "Solar_Forecast": ["datetime_beginning_utc"],
        "Wind_Forecast": ["datetime_beginning_utc"]
    }

    def __init__(self, root_directory="ETL/Data") -> object:
        self.__client = MongoClient(self.__cluster_uri)
        self.__db = self.__client["MSCI446_DB"]
        self.__root_directory = root_directory

    def iterate_over_root(self) -> None:
        """
        Iterates through each subfolder and CSV file, applying a callback function to each CSV file.

        Args:
            folder_path (str): Path to the folder containing subfolders with CSV files.
        """
        for subdir in os.listdir(self.__root_directory):
            subdir_path = os.path.join(self.__root_directory, subdir)

            # Skip if not a directory
            if not os.path.isdir(subdir_path):
                continue

            # Find all CSV files in the current subfolder
            csv_files = glob.glob(os.path.join(subdir_path, '*.csv'))

            # Print or otherwise use the subdirectory name
            print(f"Processing subdirectory: {subdir}")

            cur_collection = self.__db[subdir]

            cur_omissions = self.__omit_dict[subdir]

            for csv_file in csv_files:
                self.__etl_process(csv_file, cur_collection, cur_omissions)

    def iterate_over_folder(self, folder_name: str) -> None:
        """
        Iterates through each subfolder and CSV file, applying a callback function to each CSV file.

        Args:
            folder_path (str): Path to the folder containing subfolders with CSV files.
        """

        folder_path = self.__get_folder_path(folder_name)
        if not folder_path:
            raise ValueError("Empty folder path implying non-existent folder")

        # Find all CSV files in the current subfolder
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))

        # Print or otherwise use the subdirectory name
        print(f"Processing subdirectory: {folder_name}")

        cur_collection = self.__db[folder_name]

        cur_omissions = self.__omit_dict[folder_name]

        for csv_file in csv_files:
            self.__etl_process(csv_file, cur_collection, cur_omissions)

    def list_sub_folders(self):
        """Lists all subfolders in the root directory."""
        return [f.name for f in os.scandir(self.__root_directory) if f.is_dir()]

    # Your existing methods (iterate_over_root, iterate_over_folder, etc.)

    def process(self, folder_name=None):
        """Determines whether to process a single folder or all subfolders."""
        if folder_name:
            self.iterate_over_folder(folder_name)
        else:
            self.iterate_over_root()

    def clear_all_collections(self) -> None:
        """
        Clears all documents from every collection in the MongoDB database
        without removing the collections themselves.
        """
        # Get a list of all collection names in the database
        collection_names = self.__db.list_collection_names()

        # Iterate over each collection name and delete all documents
        for collection_name in collection_names:
            print(f"Clearing collection: {collection_name}")
            self.__db[collection_name].delete_many({})

        print("All collections have been cleared.")

    def clear_specific_collection(self, collection_name: str) -> None:
        """
        Clears all documents from a specified collection in the MongoDB database
        without removing the collection itself.

        Args:
            collection_name (str): The name of the collection to clear.
        """
        if collection_name not in self.__db.list_collection_names():
            print(f"Collection {collection_name} does not exist.")
            return

        print(f"Clearing collection: {collection_name}")
        self.__db[collection_name].delete_many({})
        print(f"Collection {collection_name} has been cleared.")

    def __get_folder_path(self, folder_name: str) -> str:
        """
        Finds the full path for a folder named `folder_name` starting from `self.__root_directory`.

        Args:
            folder_name (str): The name of the folder to find.

        Returns:
            str: The full path to the folder, if found. Otherwise, returns an empty string.
        """
        for self.root_directory, dirs, files in os.walk(self.__root_directory):
            if folder_name in dirs:
                return os.path.join(self.root_directory, folder_name)
        return ""

    def __extract(self, csv_file_path: str, omissions: list) -> pd.DataFrame:
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

    def __transform(self, data_frame: pd.DataFrame) -> list:
        """
        Converts a pandas DataFrame into a list of dictionaries, suitable for MongoDB insertion.

        Each dictionary in the list represents a row in the DataFrame, with column names as keys.

        Args:
            data_frame (pd.DataFrame): The DataFrame to be transformed.

        Returns:
            list: A list of dictionaries, where each dictionary represents a row from the DataFrame.
        """
        return data_frame.to_dict("records")

    def __load(self, data_dicts: list, collection: MongoClient) -> None:
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

    def __etl_process(self, csv_file: str, collection: MongoClient, omissions: list) -> None:
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
        data = self.__extract(csv_file, omissions)

        # Transform the DataFrame into a list of dictionaries
        data_dicts = self.__transform(data)

        # Load the transformed data into the specified MongoDB collection
        self.__load(data_dicts, collection)


if __name__ == "__main__":
    etl_tool = Etl_Tool()

    parser = argparse.ArgumentParser(description="ETL Pipeline Tool")
    parser.add_argument("-l", "--list", action="store_true",
                        help="List all available sub-folders")
    parser.add_argument("-f", "--folder", type=str,
                        help="Specify a folder name to process only that folder")
    parser.add_argument("-a", "--all", action="store_true",
                        help="Process all sub-folders within the root directory")
    parser.add_argument("-c", "--clear", action="store_true",
                        help="Clear all documents from all collections in the database")
    parser.add_argument("-lc", "--list-collections", action="store_true",
                        help="List all available collections in the database")
    parser.add_argument("-cc", "--clear-collection", type=str,
                        help="Specify a collection name to clear only that collection")

    args = parser.parse_args()

    if args.list:
        print("Available sub-folders:")
        for folder in etl_tool.list_sub_folders():
            print(folder)
    elif args.list_collections:
        print("Available collections:")
        for collection in etl_tool.__db.list_collection_names():
            print(collection)
    elif args.clear:
        etl_tool.clear_all_collections()
    elif args.clear_collection:
        etl_tool.clear_specific_collection(args.clear_collection)
    elif args.folder:
        etl_tool.process(folder_name=args.folder)
    elif args.all:
        etl_tool.iterate_over_root()
    else:
        parser.print_help()
