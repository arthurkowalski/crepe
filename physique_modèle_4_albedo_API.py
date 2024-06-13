import requests
import json

def get_power_data(lat, lon, start_date, end_date):
    url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters=ALLSKY_SFC_SW_DWN,ALLSKY_SFC_SW_UP&community=RE&longitude={lon}&latitude={lat}&start={start_date}&end={end_date}&format=JSON"
    response = requests.get(url)
    data = response.json()
    return data

def calculate_albedo(data):
    down_radiation = data['properties']['parameter']['ALLSKY_SFC_SW_DWN']
    up_radiation = data['properties']['parameter']['ALLSKY_SFC_SW_UP']

    down_values = list(down_radiation.values())
    up_values = list(up_radiation.values())

    average_down = sum(down_values) / len(down_values)
    average_up = sum(up_values) / len(up_values)

    albedo = average_up / average_down
    return average_down, average_up, albedo

def main():
    # Entrer les coordonnées GPS
    latitude = float(input("Entrez la latitude: "))
    longitude = float(input("Entrez la longitude: "))

    # Dates de début et de fin
    start_date = "20230101"
    end_date = "20231231"

    # Récupérer les données depuis l'API
    data = get_power_data(latitude, longitude, start_date, end_date)

    # Calculer les moyennes et l'albédo
    average_down, average_up, albedo = calculate_albedo(data)

    # Afficher les résultats
    print(f"Moyenne de l'irradiance descendante (kW-hr/m^2/day): {average_down:.2f}")
    print(f"Moyenne de l'irradiance ascendante (kW-hr/m^2/day): {average_up:.2f}")
    print(f"Albédo: {albedo:.2f}")

if __name__ == "__main__":
    main()