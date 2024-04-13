from meteostat import Stations

stations = Stations()
stations = stations.nearby(31.4,  121.4667 ,8)
station = stations.fetch(1)

print(station)