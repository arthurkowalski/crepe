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
phi = np.linspace(0, 2 * np.pi, 60)
theta = np.linspace(0, np.pi, 30)
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

# Fonction pour calculer la puissance reçue en fonction du temps
def calc_power(time):
    # Calcul de l'angle de rotation en fonction du temps
    angle_rotation = (time / 24) * 2 * np.pi  # Conversion du temps en angle
    rotation_matrix = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation), np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])
    sun_vector_rotated = np.dot(rotation_matrix, sun_vector)

    # Calculer l'angle d'incidence (cosinus de l'angle)
    cos_theta_incidence = np.clip(np.dot(normal.T, sun_vector_rotated), 0, 1).T

    coef_reflexion = 0.3
    # Puissance reçue par unité de surface (W/m^2)
    puissance_recue = constante_solaire * cos_theta_incidence * (1 - coef_reflexion)

    return puissance_recue

# Fonction pour mettre à jour les températures en tenant compte de l'inertie thermique
def update_temperature(time, dt=1):
    global temperature
    puissance_recue = calc_power(time)
    equilibrium_temperature = (puissance_recue / sigma) ** 0.25
    temperature += (equilibrium_temperature - temperature) * dt / tau
    return temperature

# Fonction pour mettre à jour le graphique
def update_plot(time):
    updated_temp = update_temperature(time)
    ax.clear()
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(updated_temp / np.max(updated_temp)), rstride=1, cstride=1, alpha=0.9, linewidth=0)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Température de la surface à t = {time:.1f} h')
    fig.canvas.draw_idle()

# Fonction pour convertir les coordonnées GPS en puissance reçue et température
def gps_to_power_and_temp(latitude, longitude, time):
    # Convertir latitude et longitude en radians
    lat_rad = np.radians(90 - latitude)  # Conversion pour theta
    lon_rad = np.radians(longitude)  # Conversion pour phi

    # Calculer les coordonnées sphériques correspondantes
    x_gps = rayon_astre_m * np.sin(lat_rad) * np.cos(lon_rad)
    y_gps = rayon_astre_m * np.sin(lat_rad) * np.sin(lon_rad)
    z_gps = rayon_astre_m * np.cos(lat_rad)

    # Trouver l'indice de la grille le plus proche des coordonnées calculées
    distances = np.sqrt((x - x_gps)**2 + (y - y_gps)**2 + (z - z_gps)**2)
    idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
    puissance_recue = calc_power(time)
    puissance = puissance_recue[idx]
    updated_temp = update_temperature(time)
    temp = updated_temp[idx]

    return puissance, temp

# # # Exemple d'utilisation
# # latitude = 0  # Exemple de latitude
# # longitude = -180  # Exemple de longitude
# # time = 0
# # puissance, temp = gps_to_power_and_temp(latitude, longitude, time)
# # print(f"Puissance reçue à {time} heure à la latitude {latitude}° et longitude {longitude}°: {puissance:.2f} W/m^2")
# # print(f"Température locale à {time} heure à la latitude {latitude}° et longitude {longitude}°: {temp:.2f} K")

# Création de la figure et de l'axe
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Initialisation du graphique
update_plot(0)

# Création du slider
ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_slider, 'Time (h)', 0, 48, valinit=0, valstep=1)

# Liaison du slider à la fonction de mise à jour
time_slider.on_changed(update_plot)

plt.show()

# # def extract_coordinates_long_lat(phi, theta, rayon_astre_m):
# #     # Initialiser une liste pour stocker les coordonnées
# #     coordinates = []
# #
# #     # Parcourir chaque point de la grille
# #     for i in range(phi.shape[0]):
# #         for j in range(phi.shape[1]):
# #             x_central = rayon_astre_m * np.sin(theta[i, j]) * np.cos(phi[i, j])
# #             y_central = rayon_astre_m * np.sin(theta[i, j]) * np.sin(phi[i, j])
# #             z_central = rayon_astre_m * np.cos(theta[i, j])
# #             coordinates.append((x_central, y_central, z_central))
# #
# #     return coordinates

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

# Utiliser la fonction pour extraire les coordonnées centrales
coordinates = extract_coordinates_long_lat(phi, theta, rayon_astre_m)


# Exemple d'affichage des premières coordonnées
print((coordinates))
