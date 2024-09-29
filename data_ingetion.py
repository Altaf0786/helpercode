import json
import pandas as pd
import zipfile
import logging
from abc import ABC, abstractmethod
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Define an abstract base class for data ingestion
class DataIngestor(ABC):
    @abstractmethod
    def ingest(self, file_path: Path) -> dict:
        """
        Abstract method for ingesting data. Must be implemented by subclasses.
        """
        pass

    @abstractmethod
    def save(self, dataframes: dict, output_dir: Path):
        """
        Abstract method for saving data. Must be implemented by subclasses.
        """
        pass

# Ingestor for ZIP files
class ZipDataIngestor(DataIngestor):
    def ingest(self, file_path: Path) -> dict:
        """
        Extracts and processes files from a ZIP archive, placing files directly into the data_directory.
        """
        if file_path.suffix != ".zip":
            raise ValueError("The provided file is not a .zip file.")

        # Define the directory for extraction
        dir_path = Path("data_directory")
        dir_path.mkdir(parents=True, exist_ok=True)

        # Extract ZIP file to the directory
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            logging.info("Extracting ZIP file...")
            # Extract files directly into the target directory
            for file in zip_ref.namelist():
                extracted_path = dir_path / Path(file).name
                if not file.endswith('/'):  # Skip directories
                    with zip_ref.open(file) as source, open(extracted_path, 'wb') as target:
                        target.write(source.read())

        # Collect all files in the directory (without subdirectories)
        extracted_files = [f for f in dir_path.iterdir() if f.is_file()]

        logging.debug("Extracted files: %s", extracted_files)

        if not extracted_files:
            logging.error("No files found in the extraction directory.")
            raise FileNotFoundError("No files were extracted.")

        dataframes = {}
        for file in extracted_files:
            logging.debug(f"Processing file: {file}")
            file_ext = file.suffix.lower()
            logging.debug(f"File extension: {file_ext}")

            try:
                if file_ext == ".csv":
                    df = pd.read_csv(file)
                    key = f"{file.stem}{file_ext}"
                    dataframes[key] = df
                elif file_ext in [".xlsx", ".xls"]:
                    df = pd.read_excel(file)
                    key = f"{file.stem}{file_ext}"
                    dataframes[key] = df
                elif file_ext == ".json":
                    with open(file, 'r') as f:
                        json_data = json.load(f)
                    df = pd.json_normalize(json_data)
                    key = f"{file.stem}{file_ext}"
                    dataframes[key] = df
                else:
                    logging.warning(f"Unsupported file format: {file}")
                    continue

                logging.info(f"Successfully processed file: {file}")

            except Exception as e:
                logging.error(f"Error processing file {file}: {e}")

        if not dataframes:
            raise FileNotFoundError("No supported file formats found in the extracted data.")
        
        return dataframes

    def save(self, dataframes: dict, output_dir: Path):
        """
        Save DataFrames to individual files based on their original format.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in dataframes.items():
            file_ext = Path(name).suffix
            output_path = output_dir / f"{Path(name).stem}{file_ext}"
            if file_ext == ".csv":
                df.to_csv(output_path, index=False)
            elif file_ext in [".xlsx", ".xls"]:
                df.to_excel(output_path, index=False)
            elif file_ext == ".json":
                df.to_json(output_path, orient='records', lines=True)
            else:
                logging.warning(f"Unsupported file format for saving: {file_ext}")
                continue
            
            logging.info(f"Saved DataFrame to {output_path}")

# Ingestor for CSV files
class CSVDataIngestor(DataIngestor):
    def ingest(self, file_path: Path) -> dict:
        """
        Reads and processes a CSV file.
        """
        df = pd.read_csv(file_path)
        return {file_path.name: df}

    def save(self, dataframes: dict, output_dir: Path):
        """
        Save DataFrames to CSV files in the specified directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in dataframes.items():
            output_path = output_dir / name
            df.to_csv(output_path, index=False)
            logging.info(f"Saved DataFrame to {output_path}")

# Ingestor for Excel files
class ExcelDataIngestor(DataIngestor):
    def ingest(self, file_path: Path) -> dict:
        """
        Reads and processes an Excel file.
        """
        df = pd.read_excel(file_path)
        return {file_path.name: df}

    def save(self, dataframes: dict, output_dir: Path):
        """
        Save DataFrames to Excel files in the specified directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in dataframes.items():
            output_path = output_dir / f"{Path(name).stem}.xlsx"
            df.to_excel(output_path, index=False)
            logging.info(f"Saved DataFrame to {output_path}")

# Ingestor for JSON files
class JSONDataIngestor(DataIngestor):
    def ingest(self, file_path: Path) -> dict:
        """
        Reads and processes a JSON file.
        """
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        df = pd.json_normalize(json_data)
        return {file_path.name: df}

    def save(self, dataframes: dict, output_dir: Path):
        """
        Save DataFrames to JSON files in the specified directory.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for name, df in dataframes.items():
            output_path = output_dir / f"{Path(name).stem}.json"
            df.to_json(output_path, orient='records', lines=True)
            logging.info(f"Saved DataFrame to {output_path}")

# Factory for creating data ingestors based on file extension
class DataIngestorFactory:
    @staticmethod
    def get_data_ingestor(file_extension: str) -> DataIngestor:
        """
        Factory method to get the appropriate data ingestor based on file extension.
        """
        if file_extension == ".zip":
            return ZipDataIngestor()
        elif file_extension == ".csv":
            return CSVDataIngestor()
        elif file_extension in [".xlsx", ".xls"]:
            return ExcelDataIngestor()
        elif file_extension == ".json":
            return JSONDataIngestor()
        else:
            raise ValueError(f"No ingestor available for file extension: {file_extension}")

# Main execution
if __name__ == "__main__":
    # Update this path to point to your file
    file_path = Path("/Users/mdaltafshekh/Downloads/chhs-cac6d0b5-93d0-4f62-9f05-b3eebb63431f.zip")
    output_dir = Path("data_directory")  # Using the same directory for extraction and output

    file_extension = file_path.suffix

    data_ingestor = DataIngestorFactory.get_data_ingestor(file_extension)

    try:
        dataframes = data_ingestor.ingest(file_path)
        data_ingestor.save(dataframes, output_dir)
        
        # Print the first few rows of each DataFrame
        for name, df in dataframes.items():
            print(f"DataFrame for {name}:")
            print(df.head())
    except Exception as e:
        logging.error("An error occurred: %s", e)
