import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import shapefile

# Charger les données SHP
sf = shapefile.Reader("ne_10m_coastline.shp")
shapes = sf.shapes()

# Rayon de la Terre en mètres
earth_radius = 6371000

# Initialiser une figure 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Fonction pour projeter les coordonnées sur une sphère
def project_to_sphere(lon, lat, radius=1):
    lon = np.radians(lon)
    lat = np.radians(lat)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return x, y, z

# Parcourir toutes les formes et les tracer
for shape in shapes:
    # Obtenir les coordonnées des points de la forme
    points = np.array(shape.points)
    points = points[::300]  # Réduire la résolution en prenant un échantillon de chaque 300 points
    lon = points[:, 0]
    lat = points[:, 1]

    # Vérifier si le nombre de points est suffisant après réduction
    if len(lon) < 2 or len(lat) < 2:
        continue

    # Projet des coordonnées sur une sphère avec le rayon de la Terre
    x, y, z = project_to_sphere(lon, lat, earth_radius)

    # Tracer la forme en utilisant un trait continu
    ax.plot(x, y, z, color='black')

# Réglages des axes et du titre
ax.set_title('Côtes en 3D sur une sphère')

# Masquer les grilles
ax.grid(False)

# Afficher le tracé
plt.show()
