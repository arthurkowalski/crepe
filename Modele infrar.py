import numpy as np

# Constantes
S0 = 340  # Constante solaire en W/m²
alpha = 0.3  # Albedo moyen de la Terre
latitude = 0  # Latitude en degrés
day_of_year = 195  # Jour de l'année (1 à 365)
hour_angle = 0  # Angle horaire en degrés (0 à midi solaire)

# Conversion en radians
latitude_rad = np.radians(latitude)
hour_angle_rad = np.radians(hour_angle)

# Calcul de la déclinaison solaire
declination = 23.44 * np.sin(np.radians((360 / 365) * (day_of_year - 81)))
declination_rad = np.radians(declination)

# Calcul de l'angle d'incidence
cos_theta = (np.sin(latitude_rad) * np.sin(declination_rad) +
             np.cos(latitude_rad) * np.cos(declination_rad) * np.cos(hour_angle_rad))

# Puissance solaire reçue par unité de surface en un point
P_recue_point = S0 * cos_theta

# Puissance solaire absorbée par unité de surface en un point
P_absorbee_point = P_recue_point * (1 - alpha)

# Puissance émise par unité de surface en un point (équivalent à P_absorbee_point)
P_emise_point = P_absorbee_point

print(f"Puissance émise par rayonnement infrarouge en un point : {P_emise_point:.2f} W/m²")
