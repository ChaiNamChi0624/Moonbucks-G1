import pandas as pd 
import pycountry
import openrouteservice

client = openrouteservice.Client(key='5b3ce3597851110001cf62482912466b448e4a8ea7b7ffcc05f357a1')

def load_dataset():
    df = pd.read_csv("../moonbucks_dataset.csv")
    df = df.drop(["country", "phone_number_1", "phone_number_2", "fax_1", "fax_2", "email_1", "email_2", "website", "facebook", "twitter", "instagram", "pinterest", "youtube"], axis=1)
    df = df.rename(columns={"state": "country_code"})
    df["country"] = df.apply(lambda row: pycountry.countries.get(alpha_2=row["country_code"]).name, axis=1)
    df = df[["name", "url", "street_address", "city", "zip_code", "country_code", "country", "open_hours", "latitude", "longitude"]]
    return df

def get_country_dict(df):
    country_names = df["country"].unique()
    country_codes = df["country_code"].unique()
    countries = {}
    for name, code in zip(country_names, country_codes):
        countries[code] = {"name": name, "n_stores": len(df.loc[df["country_code"] == code])}
    return countries

def get_local_stores(df, country_code=None, country_name=None):
    if country_code != None:
        local_df = df.loc[df["country_code"] == country_code].reset_index(drop=True)
    elif country_name != None:
        local_df = df.loc[df["country"] == country_name].reset_index(drop=True)
    else:
        raise Exception("Please enter either country_code or country_name.")
    return local_df

def select_distribution_center(local_stores):
    """
    Return 
    - center_idx: index of distribution center in local_store dataframe
    - center_dist: total distance needed
    - routes_mat: route infomation matrix
    """
    n_stores = len(local_stores)
    dist_mat = [[0] * n_stores for _ in range(n_stores)]
    routes_mat = [[0] * n_stores for _ in range(n_stores)]
    
    # Construct the distance matrix by computing the distance between every store in local_stores
    # - Assuming route A -> B and route B -> A take the same driving route
    for i in range(n_stores):
        for j in range(i):
            # Get coordinates of stores
            coord_i = (local_stores.iloc[i].longitude, local_stores.iloc[i].latitude)
            coord_j = (local_stores.iloc[j].longitude, local_stores.iloc[j].latitude)
            
            # Obtain the driving route from point i to point j, return a json that contains route infos            
            routes = client.directions((coord_i, coord_j))
            dist = routes['routes'][0]['summary']['distance'] / 1000    # Get the driving distance
            
            # Fill in the distance matrix
            dist_mat[i][j] = dist
            dist_mat[j][i] = dist
            
            # Fill in this matrix with route info
            routes_mat[i][j] = routes
            routes_mat[j][i] = routes
    
    # Select the distribution center among all stores (store that takes the least distance to every store)
    center_dist = float("inf")
    center_idx = None
    for i in range(n_stores):
        current_dist_sum = sum(dist_mat[i]) * 2 # multiply by 2 to include two-way travel
        if current_dist_sum < center_dist:
            center_dist = current_dist_sum
            center_idx = i
    
    return center_idx, center_dist, routes_mat
