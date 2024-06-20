import numpy as np
import shapefile

# Fonction pour projeter les coordonnées sur une sphère
def project_to_sphere(lon, lat, radius=1):
    lon = np.radians(lon)
    lat = np.radians(lat)
    x = radius * np.cos(lat) * np.cos(lon)
    y = radius * np.cos(lat) * np.sin(lon)
    z = radius * np.sin(lat)
    return (x, y, z)


def liste_coo_frontieres():
    rayon_astre = 6371
    rayon_astre_m = rayon_astre * 1000
    # Charger les données SHP
    sf = shapefile.Reader("data/ne_10m_coastline.shp")
    shapes = sf.shapes()

    coo_frontieres = []

    for shape in shapes:
        points = np.array(shape.points)
        points = points[::300]
        lon = points[:, 0]
        lat = points[:, 1]
        if len(lon) < 2 or len(lat) < 2:
            continue
        coo_line = project_to_sphere(lon, lat, 6371*1000 + 100000)
        coo_frontieres.append(coo_line)

    return(coo_frontieres)