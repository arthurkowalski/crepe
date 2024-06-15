import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# Constantes
constante_solaire = 1361  # W/m^2, valeur moyenne au niveau de la Terre
rayon_astre = 6371  # km, par exemple le rayon de la Terre
sigma = 5.670e-8  # Constante de Stefan-Boltzmann en W/m^2/K^4


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



# Fonction pour calculer la puissance reçue et la température en fonction du temps
def calc_power_temp(time, saison):
    if saison == 'été':
        angle_inclinaison = np.radians(23)
    elif saison == 'hiver':
        angle_inclinaison = np.radians(-23)
    else:
        angle_inclinaison = 0

    # Calcul de l'angle de rotation en fonction du temps
    angle_rotation = (time / 24) * 2 * np.pi  # Conversion du temps en angle
    rotation_matrix = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation), np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])

    # Rotation supplémentaire pour l'inclinaison solaire
    rotation_matrix_saison = np.array([
        [1, 0, 0],
        [0, np.cos(angle_inclinaison), -np.sin(angle_inclinaison)],
        [0, np.sin(angle_inclinaison), np.cos(angle_inclinaison)]
    ])

    rotation_matrix_total = np.dot(rotation_matrix, rotation_matrix_saison)
    sun_vector_rotated = np.dot(rotation_matrix_total, sun_vector)

    # Calculer l'angle d'incidence (cosinus de l'angle)
    cos_theta_incidence = np.clip(np.dot(normal.T, sun_vector_rotated), 0, 1).T

    coef_reflexion = 0
    # Puissance reçue par unité de surface (W/m^2)
    puissance_recue = constante_solaire * cos_theta_incidence * (1 - coef_reflexion)

    # Calculer la température en utilisant la loi de Stefan-Boltzmann
    temperature = (puissance_recue / sigma) ** 0.25

    return puissance_recue, temperature

# Fonction pour mettre à jour le graphique
def update_plot(time, saison):
    puissance_recue, _ = calc_power_temp(time, saison)
    ax.clear()
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(puissance_recue / np.max(puissance_recue)), rstride=1, cstride=1, alpha=0.9, linewidth=0)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Distribution de la puissance radiative reçue par l\'astre à t = {time:.1f} h (Saison : {saison})')
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
    puissance_recue, temperature = calc_power_temp(time, 'printemps')
    puissance = puissance_recue[idx]
    temp = temperature[idx]

    return puissance, temp


latitude = 0  # Exemple de latitude
longitude = 0  # Exemple de longitude
time = 0
puissance, temp = gps_to_power_and_temp(latitude, longitude, time)
print(f"Puissance reçue à {time} heure à la latitude {latitude}° et longitude {longitude}°: {puissance:.2f} W/m^2")
print(f"Température locale à {time} heure à la latitude {latitude}° et longitude {longitude}°: {temp:.2f} K")

# dA=rayon_astre_m**2*np.sin(theta)*0.10471975511965977**2
# surface_tot = np.sum(dA)
# puissance_totale = np.sum(puissance_recue*dA)

# puissance_moyenne = puissance_totale / surface_tot
# print(f"Puissance totale reçue: {puissance_totale:.2f} W")
# print(f"Puissance moyenne reçue: {puissance_moyenne:.2f} W/m^2")

# Création de la figure et de l'axe
# # # # fig = plt.figure(figsize=(10, 7))
# # # # ax = fig.add_subplot(111, projection='3d')
# # # #
# # # # # Initialisation du graphique
# # # # update_plot(0, 'printemps')  # Saison par défaut
# # # #
# # # # # Création du slider
# # # # ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
# # # # time_slider = Slider(ax_slider, 'Time (h)', 0, 48, valinit=0, valstep=0.1)
# # # #
# # # # # Création de la liste déroulante pour la saison
# # # # ax_season = plt.axes([0.25, 0.07, 0.50, 0.03], facecolor='lightgoldenrodyellow')
# # # # season_dropdown = plt.widgets.Dropdown(ax_season, 'Saison', ['printemps', 'été', 'automne', 'hiver'])
# # # #
# # # # # Liaison du slider et de la liste déroulante à la fonction de mise à jour
# # # # time_slider.on_changed(lambda val: update_plot(time_slider.val, season_dropdown.label.get_text()))
# # # # season_dropdown.on_change(lambda label: update_plot(time_slider.val, season_dropdown.label.get_text()))
# # # #
# # # # plt.show()

# Création de la figure et de l'axe
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Initialisation du graphique
update_plot(0, 'été')

# Création du slider
ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_slider, 'Time (h)', 0, 48, valinit=0, valstep=0.1)

# Liaison du slider à la fonction de mise à jour
time_slider.on_changed(update_plot)

plt.show()
