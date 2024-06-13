import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button


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

def update_sun_vector(mois):
    # if saison == 'été':
    #     angle_inclinaison = np.radians(-23)
    # elif saison == 'hiver':
    #     angle_inclinaison = np.radians(+23)
    # else:
    #     angle_inclinaison = 0

    angle_inclinaison = np.radians(23 * np.cos(2*np.pi*mois/12))
    print(angle_inclinaison)
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

    # rotation_matrix_total = np.dot(rotation_matrix, rotation_matrix_saison)
    sun_vector_rotated = np.dot(rotation_matrix, update_sun_vector(mois))

    # Calculer l'angle d'incidence (cosinus de l'angle)
    cos_theta_incidence = np.clip(np.dot(normal.T, sun_vector_rotated), 0, 1).T

    coef_reflexion = 0
    # Puissance reçue par unité de surface (W/m^2)
    puissance_recue = constante_solaire * cos_theta_incidence * (1 - coef_reflexion)

    # Calculer la température en utilisant la loi de Stefan-Boltzmann
    temperature = (puissance_recue / sigma) ** 0.25

    return puissance_recue, temperature

# Fonction pour mettre à jour le graphique
def update_plot(time, mois=3):
    puissance_recue, _ = calc_power_temp(time, mois)
    ax.clear()
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.viridis(puissance_recue / np.max(puissance_recue)), rstride=1, cstride=1, alpha=0.9, linewidth=0)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Distribution de la puissance radiative reçue par l\'astre à t = {time:.1f} h (mois : {mois})')
    fig.canvas.draw_idle()

# Wrapper pour le slider qui ne passe qu'une valeur à update_plot
def slider_update(val):
    update_plot(time_slider.val, current_month[0])

# Création de la figure et de l'axe
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Initialisation du graphique
current_month = [3]
update_plot(0, current_month[0])

# Création du slider
ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_slider, 'Time (h)', 0, 48, valinit=0, valstep=0.1)

# Liaison du slider à la fonction de mise à jour
time_slider.on_changed(slider_update)

# # # # # Boutons pour changer la saison
# # # # ax_ete = plt.axes([0.7, 0.9, 0.1, 0.04])
# # # # btn_ete = Button(ax_ete, 'Été')
# # # # ax_hiver = plt.axes([0.7, 0.85, 0.1, 0.04])
# # # # btn_hiver = Button(ax_hiver, 'Hiver')
# # # # ax_printemps = plt.axes([0.7, 0.8, 0.1, 0.04])
# # # # btn_printemps = Button(ax_printemps, 'Printemps')
# # # # ax_automne = plt.axes([0.7, 0.75, 0.1, 0.04])
# # # # btn_automne = Button(ax_automne, 'Automne')
# # # #
# # # # def set_saison(saison):
# # # #     current_season[0] = saison
# # # #     update_sun_vector(current_season[0])
# # # #     time_slider.reset()
# # # #     update_plot(0, current_season[0])
# # # #
# # # # btn_ete.on_clicked(lambda event: set_saison('été'))
# # # # btn_hiver.on_clicked(lambda event: set_saison('hiver'))
# # # # btn_printemps.on_clicked(lambda event: set_saison('printemps'))
# # # # btn_automne.on_clicked(lambda event: set_saison('automne'))



# Fonction pour définir la saison (ou mois dans ce cas)
def set_mois(mois):
    current_month[0] = mois
    print(mois)
    update_sun_vector(current_month[0])
    time_slider.reset()
    update_plot(0, current_month[0])

# Création des axes et boutons pour chaque mois
mois_labels = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
btn_mois = []

for i, mois in enumerate(mois_labels):
    ax_mois = plt.axes([0.1, 0.95 - i*0.07, 0.1, 0.04])
    btn = Button(ax_mois, mois)
    btn.on_clicked(lambda event, m=i+1: set_mois(m))
    btn_mois.append(btn)

# Affichage de la figure
plt.show()


