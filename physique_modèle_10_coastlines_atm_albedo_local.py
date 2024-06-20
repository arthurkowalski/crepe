import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
import shapefile
import pandas as pd
from get_coo_frontieres import liste_coo_frontieres

# Charger les données SHP
sf = shapefile.Reader("data/ne_10m_coastline.shp")
shapes = sf.shapes()
cooo = liste_coo_frontieres()

list_albedo = []
for i in range (1,10):
    # Charger les données d'albédo
    csv_file_path = f'data/albedo0{i}.csv'
    albedo_data = pd.read_csv(csv_file_path)
    if i == 1:
        latitudes = albedo_data['Latitude/Longitude'].values
        longitudes = albedo_data.columns[1:].astype(float)
    # Convertir les données CSV en grille d'albédo
    albedo_grid = albedo_data.set_index('Latitude/Longitude').to_numpy()
    list_albedo.append(albedo_grid)
for i in range(10,13):
    csv_file_path = f'data/albedo{i}.csv'
    albedo_data = pd.read_csv(csv_file_path)
    # Convertir les données CSV en grille d'albédo
    albedo_grid = albedo_data.set_index('Latitude/Longitude').to_numpy()
    list_albedo.append(albedo_grid)


# Constantes
constante_solaire = 1361  # W/m^2, valeur moyenne au niveau de la Terre
rayon_astre = 6371  # km, par exemple le rayon de la Terre
sigma = 5.670e-8  # Constante de Stefan-Boltzmann en W/m^2/K^4
epaisseur_atmosphere = 600  # km, approximative thickness of Earth's atmosphere

rayon_astre_m = rayon_astre * 1000
epaisseur_atmosphere_m = epaisseur_atmosphere * 1000

# Grille sphérique pour représenter la surface de l'astre
phi = np.linspace(0, 2 * np.pi, 60)
theta = np.linspace(0, np.pi, 30)
phi, theta = np.meshgrid(phi, theta)

x = rayon_astre_m * np.sin(theta) * np.cos(phi)
y = rayon_astre_m * np.sin(theta) * np.sin(phi)
z = rayon_astre_m * np.cos(theta)

x_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.sin(theta) * np.cos(phi)
y_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.sin(theta) * np.sin(phi)
z_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.cos(theta)

# Créer une grille pour stocker les valeurs d'albédo
albedo_grid_mapped = np.zeros_like(x)

# Angle d'incidence des rayons solaires
normal = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
sun_vector = np.array([1, 0, 0])

def update_sun_vector(mois):
    angle_inclinaison = np.radians(23 * np.cos(2 * np.pi * mois / 12))
    rotation_matrix_saison = np.array([
        [np.cos(angle_inclinaison), 0, np.sin(angle_inclinaison)],
        [0, 1, 0],
        [-np.sin(angle_inclinaison), 0, np.cos(angle_inclinaison)]
    ])
    sun_vector_rotated = np.dot(rotation_matrix_saison, sun_vector)
    return sun_vector_rotated

# Fonction pour calculer la puissance reçue et la température en fonction du temps
def calc_power_temp(time, mois):
    # Calcul de l'angle de rotation en fonction du temps
    angle_rotation = (time / 24) * 2 * np.pi  # Conversion du temps en angle
    rotation_matrix = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation), np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])

    sun_vector_rotated = np.dot(rotation_matrix, update_sun_vector(mois))

    # Calculer l'angle d'incidence (cosinus de l'angle)
    cos_theta_incidence = np.clip(np.dot(normal.T, sun_vector_rotated), 0, 1).T

    # Mapper les valeurs d'albédo sur la grille
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            lon, lat = np.degrees(phi[i, j]), 90 - np.degrees(theta[i, j])
            if lon > 180:
                lon -= 360
            albedo_grid_mapped[i, j] = get_albedo(lat, lon, mois)

    coef_reflexion =  albedo_grid_mapped
    # Puissance reçue par unité de surface (W/m^2)
    puissance_recue = constante_solaire * cos_theta_incidence * (1 - coef_reflexion)

    # Calculer la température en utilisant la loi de Stefan-Boltzmann
    temperature = (puissance_recue / sigma) ** 0.25

    return puissance_recue, temperature

# Fonction pour mettre à jour le graphique
def update_plot(time, mois=3):
    puissance_recue, _ = calc_power_temp(time, mois)
    ax.clear()



    # Plot the coastlines
    for shape in shapes:
        points = np.array(shape.points)
        points = points[::300]
        lon = points[:, 0]
        lat = points[:, 1]
        if len(lon) < 2 or len(lat) < 2:
            continue
        x_coast, y_coast, z_coast = project_to_sphere(lon, lat, rayon_astre_m + 100000)
        ax.plot(x_coast, y_coast, z_coast, color='black', zorder=5)


    # Plot the sphere surface
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(puissance_recue / np.max(puissance_recue)), rstride=1, cstride=1, linewidth=1)

    # Uncomment and plot additional coordinates if necessary
    # for i in range(len(cooo)):
    #     ax.plot(cooo[i][0], cooo[i][1], cooo[i][2], color='black', zorder=5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Distribution de la puissance radiative reçue par l\'astre à t = {time:.1f} h (mois : {mois})')
    fig.canvas.draw_idle()

# Wrapper pour le slider qui ne passe qu'une valeur à update_plot
def slider_update(val):
    update_plot(time_slider.val, current_month[0])

# Fonction pour définir la saison (ou mois dans ce cas)
def set_mois(mois):
    current_month[0] = mois
    update_sun_vector(current_month[0])
    time_slider.reset()
    update_plot(0, current_month[0])


# Fonction pour projeter les coordonnées sur une sphère
def project_to_sphere(lon, lat, radius=1):
    lon = np.radians(lon)
    lat = np.radians(lat)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

def get_albedo(lat, lon, mois):
    lat_idx = (np.abs(latitudes - lat)).argmin()
    lon_idx = (np.abs(longitudes - lon)).argmin()
    return list_albedo[mois-1][lat_idx, lon_idx]

# Créer une grille pour stocker les valeurs d'albédo
albedo_grid_mapped = np.zeros_like(x)

# Création de la figure et de l'axe
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Initialisation du graphique
current_month = [3]
update_plot(0, current_month[0])

# Création du slider
ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_slider, 'Time (h)', 0, 48, valinit=0, valstep=1)

# Liaison du slider à la fonction de mise à jour
time_slider.on_changed(slider_update)



# Création des axes et boutons pour chaque mois
mois_labels = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
btn_mois = []

for i, mois in enumerate(mois_labels):
    ax_mois = plt.axes([0.1, 0.95 - i * 0.07, 0.1, 0.04])
    btn = Button(ax_mois, mois)
    btn.on_clicked(lambda event, m=i + 1: set_mois(m))
    btn_mois.append(btn)

# Affichage de la figure
plt.show()
