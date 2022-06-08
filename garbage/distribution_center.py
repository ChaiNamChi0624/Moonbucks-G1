from util import load_dataset, get_local_stores, get_country_dict
from geopy import distance as geopy_dist
from mapbox import Directions
import polyline
import os
import folium
import math


class SelectDistribCenter:
    def __init__(self, country_code=None, country_name=None):
        df = load_dataset()
        self.local_store_df = get_local_stores(df, country_code=country_code, country_name=country_name)
        self.__coordinates = self.__get_coordinates()

        with open("mapbox_api_key.txt") as f:
            api_key = f.readline()
        os.environ['MAPBOX_ACCESS_TOKEN'] = api_key
        self.__service = Directions()

        self.distrib_center_idx, self.direction_info = self.__find_center()
        self.folium_map = self.__build_map()
        
    
    def info(self):
        return self.local_store_df.iloc[self.distrib_center_idx].to_dict()
        # return a python dictionary of selected distribution center

    def plot(self):
        return self.folium_map
        # return folium map object

    def __find_center(self):
        n = len(self.__coordinates)

        direction_matrix = [[{"dist": None, "path": None, "duration": None, "code": "Ok"} for _ in range(n)] for _ in range(n)]

        for i in range(n):
            for j in range(i+1, n):
                cur_distance, cur_duration, cur_path = self.__distance_func(self.__coordinates[i], self.__coordinates[j])
                direction_matrix[i][j]["dist"] = direction_matrix[j][i]["dist"] = cur_distance
                direction_matrix[i][j]["path"] = direction_matrix[j][i]["path"] = cur_path
                direction_matrix[i][j]["duration"] = direction_matrix[j][i]["duration"] = cur_duration

                if cur_duration == None and cur_path == None:
                    direction_matrix[i][j]["code"] = direction_matrix[j][i]["code"] = "NoRoute"
        
        min_index = -1
        min_distance = -1
        for i in range(n):
            total_distance = sum(entry_dict["dist"] if entry_dict["dist"] != None else 0 for entry_dict in direction_matrix[i])
            if total_distance < min_distance or min_index < 0:
                min_distance = total_distance
                min_index = i


        direction_matrix[min_index][min_index]["code"] = "Center"
        return min_index, direction_matrix[min_index]

    def __build_map(self):
        # given distrib_center_idx, direction_info
        # folium use (lat, long)
        center_coord = self.__coordinates[self.distrib_center_idx][::-1]
        center_info = self.info()
        address = ", ".join(center_info[key] for key in ('street_address', 'zip_code', 'city', 'country') if key in center_info and (center_info[key] != "0" or not math.isnan(center_info[key])))
        open_hours = center_info["open_hours"].split(", ")
        # open_hours = [{open_hour.split(" : ")[0], open_hour.split(" : ")[1]} for open_hour in open_hours]
        # [["Monday", ], ["Tuesday", ], ["Wednesday", ], ["Thursday", ], ["Friday", ], ["Saturday", ], ["Sunday", ]]
        # for item in open_hours:
        #     item[0]
        # open_hours = sorted(open_hours, key=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

        # print(open_hours)

        center_tooltip = f"""
            <h3>{center_info['name']}</h3>
            <p><b>Address</b> : {address}</p>
            <table>
                <tr><th>Opening Hours</th></tr>
                {"".join(f"<tr><td>{open_hour}</tr></td>" for open_hour in open_hours)}
            </table>
        """
        folium_map = folium.Map(location=center_coord, control_scale=True)

        folium.Marker(location=center_coord, tooltip=folium.Tooltip(center_tooltip), icon=folium.Icon(color="green", icon="building-o", prefix="fa")).add_to(folium_map)
        
        for idx, path_info in enumerate(self.direction_info):
            if idx != self.distrib_center_idx:
                cur_dist = path_info["dist"]
                cur_dur = path_info["duration"]
                cur_path = {"type": "LineString", "coordinates": polyline.decode(path_info["path"], geojson=True)}
                folium.GeoJson(cur_path).add_child(folium.Popup(f"<b>cur_dist : </b> {cur_dist} km <br><b>cur_dur : </b> {cur_dur} minutes", max_width=300)).add_to(folium_map)

                # folium.Marker(location=center_coord, tooltip=folium.Tooltip("abc"), icon=folium.Icon(color="red", icon="shopping-basket", prefix="fa")).add_to(folium_map)
        return folium_map

    def __get_coordinates(self):
        # return in (long, lat) format
        coordinates = []
        for i in range(len(self.local_store_df)):
            latitude = self.local_store_df["latitude"][i]
            longitude = self.local_store_df["longitude"][i]

            coordinates.append((longitude, latitude))

        return coordinates

    def __distance_func(self, coord1, coord2):
        # mapbox use (long, lat) format
        # geopy use (lat, long) format
        response = self.__service.directions([coord1, coord2], profile='mapbox/driving')

        if response.status_code != 200:
            raise Exception("Failed to retrieve routes from Mapbox API")

        response_json = response.json()

        if response_json["code"] == "NoRoute":
            print("No route found, use geodesic distance")

            dist = geopy_dist.distance(coord1[::-1], coord2[::-1]).km
            dur = path = None

        else:
            dist = response_json["routes"][0]["distance"] / 1000    # km
            dur = response_json["routes"][0]["duration"] / 60       # minute
            path = response_json["routes"][0]["geometry"]           # polyline string

        return dist, dur, path

      