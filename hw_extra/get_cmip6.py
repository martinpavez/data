import os

import requests
from pyesgf.search import SearchConnection
from tqdm import tqdm

NODE_URL = "https://esgf-data.dkrz.de/esg-search"


def check_availability(project,
                      variable,
                      experiment_id,
                      frequency,
                      grid,
                      institution_id,
                      variant_label,
                      table_id,
                      source_id,
                      data_node
                      ):
    conn = SearchConnection(NODE_URL, distrib=True)
    # Create a search context with the specified parameters
    ctx = conn.new_context(
        project=project,
        variable=variable,
        experiment_id=experiment_id,
        product="model-output",
        frequency=frequency,
        latest=True,
        institution_id=institution_id,
        variant_label=variant_label,
        table_id=table_id,
        source_id=source_id,
        data_node=data_node
    )

    # Perform the search
    search_results = ctx.search()

    # Count the number of hits
    hit_count = ctx.hit_count

    # Get a list of available datasets
    datasets = [result.dataset_id for result in search_results]

    return hit_count, datasets

def get_download_urls(project,
                      variable,
                      experiment_id,
                      frequency,
                      grid,
                      institution_id,
                      variant_label,
                      table_id,
                      source_id,
                      data_node
                      ):
    # ESGF connection setup
    conn = SearchConnection(NODE_URL, distrib=True)

    ctx = conn.new_context(
        project=project,
        variable=variable,
        experiment_id=experiment_id,
        product="model-output",
        frequency=frequency,
        latest=True,
        institution_id=institution_id,
        variant_label=variant_label,
        table_id=table_id,
        source_id=source_id,
        data_node=data_node
    )

    # Perform the search
    search_results = ctx.search()

    total_links = []
    hit_count = int(ctx.hit_count)

    # Print the total number of results
    print(f"Total hits: {hit_count}")
    print(search_results)

    # Check if there are results
    if hit_count > 0:
        # Get the first result
        ds = search_results[0]
        files = ds.file_context().search()  # Find files for the first result
        for f in files:
            download_url = str(f.download_url)  # Extract the download URL
            total_links.append(download_url)  # Add the URL to the list

    return total_links

def download_datasets(url_list, download_dir):

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)  # Create the directory if it doesn't exist

    for url in tqdm(url_list, desc="Downloading"):
        try:
            # Get the file name from the URL
            filename = os.path.basename(url)
            file_path = os.path.join(download_dir, filename)

            # Make a GET request to download the file
            response = requests.get(url, stream=True)
            response.raise_for_status()  # Raise an error if the request fails

            # Save the file in the specified directory
            with open(file_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            print(f"Downloaded: {file_path}")

        except requests.RequestException as e:
            print(f"Failed to download {url}. Error: {e}")

def get_CMIP6_data(project,
                   variable,
                   experiment_id,
                   frequency,
                   grid,
                   institution_id,
                   variant_label,
                   table_id,
                   source_id,
                   data_node,
                   download_dir
                   ):
    # Get download URLs
    download_urls = get_download_urls(
        project=project,
        variable=variable,
        experiment_id=experiment_id,
        frequency=frequency,
        grid=grid,
        institution_id=institution_id,
        variant_label=variant_label,
        table_id=table_id,
        source_id=source_id,
        data_node=data_node
    )
    print(download_urls)

    # Download the files
    download_datasets(download_urls, download_dir)

if __name__ == "__main__":
    project = "CMIP6"
    variable = "psl"
    experiment_id = "ssp126"
    frequency = "mon"
    grid = "gr1"
    institution_id = "EC-Earth-Consortium"
    variant_label = "r1i1p1f1"
    table_id = "Amon"
    
    source_id = "EC-Earth3"
    data_node = "esgf.ceda.ac.uk"
    download_dir = "./data/cmip6"

    get_CMIP6_data(project,
                   variable,
                   experiment_id,
                   frequency,
                   grid,
                   institution_id,
                   variant_label,
                   table_id,
                   source_id,
                   data_node,
                   download_dir)
    # hits, datasets = check_availability(project,
    #                variable,
    #                experiment_id,
    #                frequency,
    #                grid,
    #                institution_id,
    #                variant_label,
    #                table_id,
    #                source_id,
    #                data_node)
    # print(f"Found {hits} datasets")
    # for ds in datasets:
    #     print(ds)