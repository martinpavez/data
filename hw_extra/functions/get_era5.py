import os

import cdsapi
from dotenv import load_dotenv


def load_environment_variables():
    """Load environment variables from the .env file."""
    load_dotenv()
    cds_api_key = os.getenv('CDS_API_KEY')
    cds_api_url = os.getenv('CDS_API_URL')
    return cds_api_url, cds_api_key



def create_cds_client(cds_api_url, cds_api_key):
    """Create a CDS API client with the provided credentials."""
    return cdsapi.Client(url=cds_api_url, key=cds_api_key)


def build_request(variable, year, pressure_level="500",
                  product_type="monthly_averaged_reanalysis",
                  time="00:00", data_format="netcdf",
                  download_format="unarchived", **extra_params):
    """Build a request dictionary with the given parameters, allowing for extra parameters."""
    request = {
        "product_type": product_type,
        "variable": variable,
        "pressure_level": pressure_level,
        "year": year,
        "month": [
            "01", "02", "03", "04", "05", "06",
            "07", "08", "09", "10", "11", "12"
        ],
        "time": time,
        "data_format": data_format,
        "download_format": download_format
    }
    request.update(extra_params)
    return request


def retrieve_data(client, dataset, request, save_path):
    """Retrieve data from the CDS API and save it in the specified directory."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Crea el directorio si no existe
    client.retrieve(dataset, request).download(target=save_path)
    print(f"Archivo guardado en: {save_path}")

if __name__ == '__main__':

    
    cds_api_url, cds_api_key = load_environment_variables()
    client = create_cds_client(cds_api_url, cds_api_key)
    
    # Parámetros de solicitud
    dataset = "reanalysis-era5-pressure-levels-monthly-means"
    variable = ["geopotential"]
    year = ["1973"]
    
    request = build_request(variable, year)
    
    # Directorio donde se guardará el archivo
    output_dir = "--"
    output_filename = "era5_geopotential_1973.nc"
    save_path = os.path.join(output_dir, output_filename)
    
    
    retrieve_data(client, dataset, request, save_path)
