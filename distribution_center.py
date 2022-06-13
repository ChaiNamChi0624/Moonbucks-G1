from geopy import distance as geopy_dist
from util import load_dataset, get_local_stores
from algorithms import held_karp_tsp, nearest_center, nearest_neighbour_tsp
import folium
import math
import json
import requests
import polyline
import os
import warnings
from mapbox import Directions
from collections import OrderedDict


class DistributionCenter:
    def __init__(self, country_code=None, country_name=None):
        df = load_dataset()
        self.local_store_df = get_local_stores(
            df, country_code=country_code, country_name=country_name
        )
        self.cost = None
        self.delivery_route = None
        self.center_idx = None
        self.folium_map = None

    def solve(self, method="held-karp"):
        distance_mat = self.__build_distance_matrix()
        self.center_idx = self.__find_center()
        geodesic_cost, path = self.__solve_tsp(distance_mat, method)

        start_from = path.index(self.center_idx)
        self.delivery_route = path[start_from:] + path[0 : start_from + 1]
        path_dict, self.cost = self.__fetch_delivery_routes_info(distance_mat)
        self.folium_map = self.__build_map(distance_mat, path_dict)

    def plot(self):
        return self.folium_map

    def info(self):
        return {"cost": self.cost, "delivery_route": self.delivery_route, "distribution_center": self.local_store_df.iloc[self.center_idx].to_dict()}

    def __build_distance_matrix(self):
        n = len(self.local_store_df)
        distance_mat = [[0 for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1):
                if i == j:
                    continue

                latlong_i = (
                    self.local_store_df.iloc[i]["latitude"],
                    self.local_store_df.iloc[i]["longitude"],
                )
                latlong_j = (
                    self.local_store_df.iloc[j]["latitude"],
                    self.local_store_df.iloc[j]["longitude"],
                )

                geodesic_dist = geopy_dist.geodesic(latlong_i, latlong_j).km
                distance_mat[i][j] = distance_mat[j][i] = geodesic_dist

        return distance_mat

    def __solve_tsp(self, distance_mat, method):
        if method == "held-karp":
            cost, path = held_karp_tsp(distance_mat)
        elif method == "nearest-neighbour":
            cost, path = nearest_neighbour_tsp(distance_mat, self.center_idx)
        else:
            cost = 0
            path = list(range(len(distance_mat)))
        
        return cost, path

    def __find_center(self):
        coordinates = list(self.local_store_df[["latitude", "longitude"]].to_records(index=False))
        center_idx = nearest_center(coordinates=coordinates)

        return center_idx

    def __fetch_delivery_routes_info(self, distance_mat):
        with open("./mapbox_api_key.txt") as f:
            api_key = f.readline()
        os.environ["MAPBOX_ACCESS_TOKEN"] = api_key
        service = Directions()

        path_dict = OrderedDict()
        delivery_cost = 0   # x1 for driving route, x2 for geodesic distance

        for i in range(len(self.delivery_route)):
            if i == len(self.delivery_route) - 1:
                break

            origin_idx = self.delivery_route[i]
            dest_idx = self.delivery_route[i + 1]

            latlong_origin = (
                self.local_store_df.iloc[origin_idx]["latitude"],
                self.local_store_df.iloc[origin_idx]["longitude"],
            )
            latlong_dest = (
                self.local_store_df.iloc[dest_idx]["latitude"],
                self.local_store_df.iloc[dest_idx]["longitude"],
            )

            response = service.directions(
                [latlong_origin[::-1], latlong_dest[::-1]], profile="mapbox/driving"
            )

            if response.status_code != 200:
                raise Exception("Failed to retrieve routes from Mapbox API")

            response_json = response.json()

            if response_json["code"] == "NoRoute":
                warnings.warn("No route found, use geodesic distance")
                cur_geodesic_dist = distance_mat[origin_idx][dest_idx]
                path_dict[(origin_idx, dest_idx)] = {
                    "driving_distance": None,
                    "geodesic_distance": cur_geodesic_dist,
                    "duration": None,
                    "path": None,
                    "code": "NoRoute",
                }
                delivery_cost += cur_geodesic_dist * 2
            else:
                cur_driving_dist = response_json["routes"][0]["distance"] / 1000  # km
                cur_path = response_json["routes"][0]["geometry"]  # polyline string
                cur_dur = response_json["routes"][0]["duration"] / 60  # minutes
                cur_geodesic_dist = distance_mat[origin_idx][dest_idx]
                path_dict[(origin_idx, dest_idx)] = {
                    "driving_distance": cur_driving_dist,
                    "geodesic_distance": cur_geodesic_dist,
                    "duration": cur_dur,
                    "path": cur_path,
                    "code": "Ok",
                }
                delivery_cost += cur_driving_dist * 1

        return path_dict, delivery_cost

    def __build_map(self, distance_mat, path_dict):
        def construct_address(store_info):
            address = ""
            for i, key in enumerate(("street_address", "zip_code", "city", "country")):
                try:
                    isnan = math.isnan(store_info[key])
                except:
                    isnan = False  # If input to math.isnan is not a float
                if store_info[key] != "0" and not isnan:
                    address += (
                        f"{store_info[key]}, " if i != 3 else f"{store_info[key]}."
                    )
            return address

        def construct_openhours(store_info):
            try:
                isnan = math.isnan(store_info["open_hours"])
            except:
                isnan = False  # If input to math.isnan is not a float

            if not isnan:
                openhours = {}
                for openhour in store_info["open_hours"].split(", "):
                    day, time = openhour.split(" : ")
                    openhours[day] = time
                return openhours
            return None

        def construct_store_tooltip(store_info, storetype):
            address = construct_address(store_info)
            openhours = construct_openhours(store_info)
            storename = store_info["name"]
            url = store_info["url"]
            tooltip = f"""
                <p><i>{storetype}</i></p>
                <h3>{storename}</h3>
                <hr>
                <p><b>Address</b> : {address}</p>
                <p><b>LatLong</b> : ({store_info["latitude"]}, {store_info["longitude"]})</p>
                <table>
                    <tr><th>Opening Hours</th></tr>
                    {"".join(f"<tr><td>{day}</td><td>{openhours[day]}</td></tr>" for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"] if day in openhours) if openhours != None else f"<tr><td>{'Not available'}</td><td></td></tr>"}
                </table>
                <br>
                <p><b>Check us out on :</b><br><a href="{url}">{url}</a></p>
            """
            return tooltip

        def fetch_country_geojson(country_name):
            res = requests.get(
                f"https://nominatim.openstreetmap.org/search?country={country_name}&polygon_geojson=1&format=json"
            )
            country_geojson = json.loads(res.content.decode())[0]["geojson"]
            return country_geojson

        folium_map = folium.Map(
            location=(
                self.local_store_df.iloc[self.center_idx]["latitude"],
                self.local_store_df.iloc[self.center_idx]["longitude"],
            ),
            control_scale=True,
        )
        folium.GeoJson(
            fetch_country_geojson(country_name=self.local_store_df.iloc[0]["country"]),
            name=self.local_store_df.iloc[0]["country"],
            style_function=lambda x: {"fillOpacity": 0.1},
            show=False,
        ).add_to(folium_map)

        for idx, store_info in self.local_store_df.iterrows():
            latlong_store = (store_info["latitude"], store_info["longitude"])
            if idx == self.center_idx:
                center_tooltip = construct_store_tooltip(
                    store_info, storetype="Moonbucks' Distribution Center"
                )
                folium.Marker(
                    location=latlong_store,
                    popup=folium.Popup(center_tooltip, min_width=300, max_width=300),
                    tooltip=folium.Tooltip(center_tooltip),
                    icon=folium.Icon(color="red", icon="building-o", prefix="fa"),
                ).add_to(folium_map)
            else:
                store_tooltip = construct_store_tooltip(
                    store_info, storetype="Moonbucks' Branch"
                )
                folium.Marker(
                    location=latlong_store,
                    popup=folium.Popup(store_tooltip, min_width=300, max_width=300),
                    tooltip=folium.Tooltip(store_tooltip),
                    icon=folium.Icon(color="green", icon="home", prefix="glyphicon"),
                ).add_to(folium_map)

        for idx, key in enumerate(path_dict):
            origin_idx, dest_idx = key
            if path_dict[key]["code"] == "NoRoute":
                cur_geodesic_distance = path_dict[key]["geodesic_distance"]

                path_popup = f"""
                    <h5>From : <i><u>{self.local_store_df.iloc[origin_idx]["name"]}</u>&nbsp;<small>({self.local_store_df.iloc[origin_idx]["latitude"]}, {self.local_store_df.iloc[origin_idx]["longitude"]})</small></i> <br>To : <i><u>{self.local_store_df.iloc[dest_idx]["name"]}</u>&nbsp;<small>({self.local_store_df.iloc[dest_idx]["latitude"]}, {self.local_store_df.iloc[dest_idx]["longitude"]})</small></i></h5>
                    <p style='color: red'><b>No driving route found, geodesic distance is used</b></p>
                    <b>Driving distance : </b> - <br> 
                    <b>Geodesic distance : </b> {round(cur_geodesic_distance, 2)} km &nbsp;<small>(geodesic_dist × 2 is applied to cost)</small><br> 
                    <b>Duration : </b> -
                """
                latlong_origin = (
                    self.local_store_df.iloc[origin_idx]["latitude"],
                    self.local_store_df.iloc[origin_idx]["longitude"],
                )
                latlong_dest = (
                    self.local_store_df.iloc[dest_idx]["latitude"],
                    self.local_store_df.iloc[dest_idx]["longitude"],
                )

                fg = folium.FeatureGroup(name=f"{idx + 1} : {self.local_store_df.iloc[origin_idx]['name']} → {self.local_store_df.iloc[dest_idx]['name']}")
                folium.PolyLine(
                    [latlong_origin, latlong_dest],
                    color="green", weight="4", dash_array="10"
                ).add_child(folium.Popup(path_popup, max_width=400)).add_to(fg)
                fg.add_to(folium_map)

            else:
                cur_duration = path_dict[key]["duration"]
                cur_driving_distance = path_dict[key]["driving_distance"]
                cur_geodesic_distance = path_dict[key]["geodesic_distance"]
                cur_path_coords = polyline.decode(path_dict[key]["path"], geojson=True)
                cur_path_geojson = {
                    "type": "LineString",
                    "coordinates": cur_path_coords,
                }

                path_popup = f"""
                    <h5>From : <i><u>{self.local_store_df.iloc[origin_idx]["name"]}</u>&nbsp;<small>({self.local_store_df.iloc[origin_idx]["latitude"]}, {self.local_store_df.iloc[origin_idx]["longitude"]})</small></i> <br>To : <i><u>{self.local_store_df.iloc[dest_idx]["name"]}</u>&nbsp;<small>({self.local_store_df.iloc[dest_idx]["latitude"]}, {self.local_store_df.iloc[dest_idx]["longitude"]})</small></i></h5>
                    <b>Driving distance : </b> {round(cur_driving_distance, 2)} km &nbsp;<small>(driving_dist × 1 is applied to cost)</small><br>
                    <b>Geodesic distance : </b> {round(cur_geodesic_distance, 2)} km <br>
                    <b>Duration : </b> {round(cur_duration, 2)} minutes
                """
                folium.GeoJson(
                    cur_path_geojson,
                    style_function=lambda x: {"color": "green", "weight": 4},
                    highlight_function=lambda x: {"fillColor": "#c30010", "color": "#c30010", "fillOpacity": 1, "weight": 4},
                    name=f"{idx + 1} : {self.local_store_df.iloc[origin_idx]['name']} → {self.local_store_df.iloc[dest_idx]['name']}",
                ).add_child(folium.Popup(path_popup, max_width=400)).add_to(folium_map)

        folium.LayerControl().add_to(folium_map)
        return folium_map
    