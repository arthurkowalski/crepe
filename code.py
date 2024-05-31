import numpy as np
import matplotlib.pyplot as plt

# Les années jusqu'en 2100
years = np.arange(1850, 2101, 10)

# Niveaux de CO2 et anomalies de température jusqu'en 2020 (valeurs réelles)
co2_levels = np.array([285, 287, 288, 290, 291, 292, 294, 295, 296, 297, 300, 316, 325, 338, 354, 369, 390, 414])
temperature_anomalies = np.array([-0.3, -0.3, -0.3, -0.2, -0.3, -0.1, -0.4, -0.3, -0.1, 0.0, -0.2, -0.1, -0.1, 0.1, 0.2, 0.4, 0.6, 1.0])

# Création du graphique initial
plt.figure(figsize=(10, 6))
plt.scatter(co2_levels, temperature_anomalies, color='blue', label='Données')
plt.plot(co2_levels, temperature_anomalies, color='blue')

# Ajout des labels et du titre
plt.xlabel('Niveaux de CO2 (ppm)')
plt.ylabel('Anomalies de température (°C)')
plt.title('Anomalies de température en fonction des niveaux de CO2')
plt.legend()
plt.grid(True)

# Scénarios futurs avec des niveaux de CO2 et d'anomalies de température fictifs
future_co2_scenarios = [[420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590, 600, 610],
                        [400, 410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590],
                        [390, 395, 400, 405, 410, 415, 420, 425, 430, 435, 440, 445, 450, 455, 460, 465, 470, 475, 480, 485]]

future_temperature_scenarios = [[1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
                                [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7],
                                [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5]]

for i in range(len(future_co2_scenarios)):
    plt.plot(future_co2_scenarios[i], future_temperature_scenarios[i], label=f'Scénario {i+1}')

# Affichage du graphique
plt.legend()
plt.show()
