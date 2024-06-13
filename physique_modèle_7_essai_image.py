import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider, Button
from PIL import Image

# Load the image
image = Image.open('wrld.jpg')
image = image.resize((60, 30))  # Resize the image to map correctly

# Convert the image to a NumPy array
image = np.array(image)

# Constants
constante_solaire = 1361  # W/m^2, average value at Earth's level
rayon_astre = 6371  # km, for example, Earth's radius
sigma = 5.670e-8  # Stefan-Boltzmann constant in W/m^2/K^4
epaisseur_atmosphere = 100  # km, approximate thickness of Earth's atmosphere

rayon_astre_m = rayon_astre * 1000
epaisseur_atmosphere_m = epaisseur_atmosphere * 1000

# Spherical grid to represent the surface of the star
phi = np.linspace(0, 2 * np.pi, image.shape[1])
theta = np.linspace(0, np.pi, image.shape[0])
phi, theta = np.meshgrid(phi, theta)

x = rayon_astre_m * np.sin(theta) * np.cos(phi)
y = rayon_astre_m * np.sin(theta) * np.sin(phi)
z = rayon_astre_m * np.cos(theta)

# Atmospheric layer
x_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.sin(theta) * np.cos(phi)
y_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.sin(theta) * np.sin(phi)
z_atmosphere = (rayon_astre_m + epaisseur_atmosphere_m) * np.cos(theta)

# Angle of incidence of solar rays
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

def calc_power_temp(time, mois):
    angle_rotation = (time / 24) * 2 * np.pi  # Time to angle conversion
    rotation_matrix = np.array([
        [np.cos(angle_rotation), -np.sin(angle_rotation), 0],
        [np.sin(angle_rotation), np.cos(angle_rotation), 0],
        [0, 0, 1]
    ])
    sun_vector_rotated = np.dot(rotation_matrix, update_sun_vector(mois))
    cos_theta_incidence = np.clip(np.dot(normal.T, sun_vector_rotated), 0, 1).T
    coef_reflexion = 0
    puissance_recue = constante_solaire * cos_theta_incidence * (1 - coef_reflexion)
    temperature = (puissance_recue / sigma) ** 0.25
    return puissance_recue, temperature

def update_plot(time, mois=3):
    ax.clear()
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=image / 255., alpha=1.0, linewidth=0)
    # ax.plot_surface(x, y, z, rstride=1, cstride=1,facecolors=image / 255., edgecolor='black', alpha=1.0, linewidth=1)
    # ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=image / 255., edgecolor='black', alpha=1.0, linewidth=10, shade=False)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=image / 255.)

    # ax.plot_surface(x_atmosphere, y_atmosphere, z_atmosphere, color='blue', alpha=0.1, linewidth=0)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title(f'Solar radiation power distribution on the star at t = {time:.1f} h (month: {mois})')
    fig.canvas.draw_idle()

def slider_update(val):
    update_plot(time_slider.val, current_month[0])

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
current_month = [3]
update_plot(0, current_month[0])

ax_slider = plt.axes([0.25, 0.02, 0.50, 0.03], facecolor='lightgoldenrodyellow')
time_slider = Slider(ax_slider, 'Time (h)', 0, 48, valinit=0, valstep=0.1)
time_slider.on_changed(slider_update)

def set_mois(mois):
    current_month[0] = mois
    update_sun_vector(current_month[0])
    time_slider.reset()
    update_plot(0, current_month[0])

mois_labels = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
btn_mois = []

for i, mois in enumerate(mois_labels):
    ax_mois = plt.axes([0.1, 0.95 - i * 0.07, 0.1, 0.04])
    btn = Button(ax_mois, mois)
    btn.on_clicked(lambda event, m=i + 1: set_mois(m))
    btn_mois.append(btn)

plt.show()
