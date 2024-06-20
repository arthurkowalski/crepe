import requests
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Constantes
constante_solaire = 1361  # W/m^2, valeur moyenne au niveau de la Terre
rayon_astre = 6371  # km, par exemple le rayon de la Terre
sigma = 5.670e-8  # Constante de Stefan-Boltzmann en W/m^2/K^4
tau = 5 # Constante de temps thermique en heures

rayon_astre_m = rayon_astre * 1000

# Grille sphérique pour représenter la surface de l'astre
phi = np.linspace(0, 2 * np.pi, 6)
theta = np.linspace(0, np.pi, 3)
phi, theta = np.meshgrid(phi, theta)

x = rayon_astre_m * np.sin(theta) * np.cos(phi)
y = rayon_astre_m * np.sin(theta) * np.sin(phi)
z = rayon_astre_m * np.cos(theta)

# Angle d'incidence des rayons solaires
normal = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
sun_vector = np.array([1, 0, 0])

# Initialisation des températures avec des floats
initial_temperature = 288.0  # Température initiale en Kelvin
temperature = np.full(phi.shape, initial_temperature, dtype=float)




def get_power_data(lat, lon, start_date, end_date):
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_UP&community=RE&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
    response = requests.get(url)
    data = response.json()
    return data

def calculate_albedo(data):
    down_radiation = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
    up_radiation = data['properties']['parameter']['ALLSKY_SFC_SW_UP']

    down_values = list(down_radiation.values())
    up_values = list(up_radiation.values())

    average_down = sum(down_values) / len(down_values)
    average_up = sum(up_values) / len(up_values)

    albedo = average_up / average_down
    return albedo

def extract_coordinates_long_lat(phi, theta, rayon_astre_m):
    # Initialiser une liste pour stocker les coordonnées
    coordinates = []

    # Parcourir chaque point de la grille
    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            x_central = rayon_astre_m * np.sin(theta[i, j]) * np.cos(phi[i, j])
            y_central = rayon_astre_m * np.sin(theta[i, j]) * np.sin(phi[i, j])
            z_central = rayon_astre_m * np.cos(theta[i, j])

            # Convertir les coordonnées cartésiennes en latitude et longitude
            latitude = np.arcsin(z_central / rayon_astre_m) * 180 / np.pi
            longitude = np.arctan2(y_central, x_central) * 180 / np.pi

            coordinates.append((latitude, longitude))

    return coordinates

def main():
    # Entrer les coordonnées GPS
    all_coordinates = extract_coordinates_long_lat(phi, theta, rayon_astre_m)
    for coord in all_coordinates:
        latitude, longitude = coord

    # Dates de début et de fin
        start_date = "20230101"
        end_date = "20231231"
    
        # Récupérer les données depuis l'API
        data = get_power_data(latitude, longitude, start_date, end_date)
    
        # Calculer l'albédo
        albedo = calculate_albedo(data)
    
        print(f"Albédo local à {latitude}, {longitude} : {albedo:.2f}")

if __name__ == "__main__":
    main()
