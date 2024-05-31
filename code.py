import numpy as np
import matplotlib.pyplot as plt

# Données de CO2 (ppm) et températures moyennes globales (anomalies en °C)
# Sources :
# - CO2: NOAA (National Oceanic and Atmospheric Administration)
# - Températures: NASA (GISS Surface Temperature Analysis), NOAA

years = np.array([1850, 1860, 1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940,
                  1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])

co2_levels = np.array([285, 287, 288, 290, 291, 292, 294, 295, 296, 297,
                       300, 316, 325, 338, 354, 369, 390, 414])

temperature_anomalies = np.array([-0.3, -0.3, -0.3, -0.2, -0.3, -0.1, -0.4, -0.3, -0.1, 0.0,
                                  -0.2, -0.1, -0.1, 0.1, 0.2, 0.4, 0.6, 1.0])

# Création du graphique
plt.figure(figsize=(10, 6))
plt.scatter(co2_levels, temperature_anomalies, color='blue', label='Données')
plt.plot(co2_levels, temperature_anomalies, color='blue')

# Ajout des labels et du titre
plt.xlabel('Niveaux de CO2 (ppm)')
plt.ylabel('Anomalies de température (°C)')
plt.title('Anomalies de température en fonction des niveaux de CO2')
plt.legend()
plt.grid(True)

# Affichage du graphique
plt.show()
