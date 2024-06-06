# # # import numpy as np
# # # import matplotlib.pyplot as plt

# # # # Définir les durées de vie des GES en années
# # # lifetime_CO2 = 100  # Durée de vie approximative moyenne
# # # lifetime_CH4 = 12
# # # lifetime_N2O = 114

# # # # Définir le temps (en années)
# # # years = np.arange(0, 200, 1)

# # # # Modèle de décroissance exponentielle
# # # def decay_concentration(initial_concentration, lifetime, years):
# # #     return initial_concentration * np.exp(-years / lifetime)

# # # # Concentrations initiales arbitraires
# # # initial_concentration = 100

# # # # Calculer les concentrations au fil du temps
# # # concentration_CO2 = decay_concentration(initial_concentration, lifetime_CO2, years)
# # # concentration_CH4 = decay_concentration(initial_concentration, lifetime_CH4, years)
# # # concentration_N2O = decay_concentration(initial_concentration, lifetime_N2O, years)

# # # # Tracer les résultats
# # # plt.figure(figsize=(10, 6))
# # # plt.plot(years, concentration_CO2, label='CO2 (100 ans)')
# # # plt.plot(years, concentration_CH4, label='CH4 (12 ans)')
# # # plt.plot(years, concentration_N2O, label='N2O (114 ans)')
# # # plt.xlabel('Années')
# # # plt.ylabel('Concentration relative (%)')
# # # plt.title('Décroissance des concentrations de GES dans l\'atmosphère')
# # # plt.legend()
# # # plt.grid(True)
# # # plt.show()

# # import numpy as np
# # import scipy.integrate as integrate

# # # Pouvoirs radiatifs (W m^-2 kg^-1)
# # a_CO2 = 1.37e-5  # Référence pour CO2
# # a_CH4 = 3.63e-4  # Pour CH4
# # a_N2O = 3.03e-3  # Pour N2O

# # # Durée de vie atmosphérique (années)
# # lifetime_CO2 = 100  # Utilisation d'une grande valeur pour simplification
# # lifetime_CH4 = 12
# # lifetime_N2O = 114

# # # Période de temps pour le calcul (années)
# # time_period = 100

# # # Concentration après émission pour un gaz donné
# # def concentration(t, lifetime):
# #     return np.exp(-t / lifetime)

# # # Intégrale de la concentration pour la période donnée
# # def integrated_concentration(lifetime, time_period):
# #     result, _ = integrate.quad(concentration, 0, time_period, args=(lifetime,))
# #     return result

# # # Calcul du PRG pour un gaz donné
# # def calculate_prg(a_gas, lifetime_gas, a_CO2, lifetime_CO2, time_period):
# #     int_conc_gas = integrated_concentration(lifetime_gas, time_period)
# #     int_conc_CO2 = integrated_concentration(lifetime_CO2, time_period)
# #     prg = (a_gas * int_conc_gas) / (a_CO2 * int_conc_CO2)
# #     return prg

# # # Calcul des PRG pour CH4 et N2O
# # prg_CH4 = calculate_prg(a_CH4, lifetime_CH4, a_CO2, lifetime_CO2, time_period)
# # prg_N2O = calculate_prg(a_N2O, lifetime_N2O, a_CO2, lifetime_CO2, time_period)

# # print(f"PRG du CH4 sur {time_period} ans: {prg_CH4:.2f}")
# # print(f"PRG du N2O sur {time_period} ans: {prg_N2O:.2f}")

# a = 0.31 
# K = 5.67e-8 
# P = 342
# #T = ((1-a)*P/4*K)**(1/4)
# T=(390/K)**0.25-273
# print (T)

import numpy as np
import matplotlib.pyplot as plt

# Constantes
sigma = 5.67e-8  # Constante de Stefan-Boltzmann en W/m^2/K^4
S0 = 1361  # Constante solaire en W/m^2 (énergie reçue au sommet de l'atmosphère)
albedo = 0.3  # Albedo moyen de la Terre (fraction de l'énergie solaire réfléchie)

def temperature(latitude):
    """
    Calcule la température à la surface de la Terre en fonction de la latitude.
    
    Args:
    latitude (float): Latitude en degrés (-90 à 90).
    
    Returns:
    float: Température en degrés Celsius.
    """
    # Conversion de la latitude en radians
    latitude_rad = np.radians(latitude)
    
    # Puissance solaire reçue par unité de surface en fonction de la latitude
    # Correction par l'angle d'incidence des rayons du soleil
    S = S0 * (1 - albedo) * np.cos(latitude_rad)
    
    # Équilibre radiatif : P = sigma * T^4
    # => T = (P / sigma)^(1/4)
    # Convertir la température de K à °C
    T = (S / sigma)**0.25 - 273.15
    
    return T

# Générer les données pour la latitude de -90 à 90 degrés
latitudes = np.linspace(-90, 90, 180)
temperatures = [temperature(lat) for lat in latitudes]

# Tracer les résultats
plt.figure(figsize=(10, 6))
plt.plot(latitudes, temperatures, label='Température de surface')
plt.title("Température de surface en fonction de la latitude")
plt.xlabel("Latitude (degrés)")
plt.ylabel("Température (°C)")
plt.legend()
plt.grid(True)
plt.show()
