# pylint: disable=missing-module-docstring

import sys
import urllib.parse
import requests

BASE_URI = "https://weather.lewagon.com"


def search_city(query):
    '''Look for a given city. If multiple options are returned, have the user choose between them.
       Return one city (or None)
    '''
    # url = urllib.parse.urljoin(BASE_URI, '/geo/1.0/direct')

    url = BASE_URI + f'/geo/1.0/direct?q={query}'
    # response = requests.get(url).json()

    cities = requests.get(url, params={'q': query, 'limit': 5}).json()

    if not cities:
        print(f"Sorry, OpenWeather does not know about {query}!")
        return None

    if len(cities) == 1:
        return cities[0]

    for i, city in enumerate(cities):
        print(f"{i + 1}, {city['name']}, {city['country']}")

    index = int(input("Multiple matches found, which city did you mean?\n> ")) - 1

    return cities[index]


def weather_forecast(lat, lon):
    '''Return a 5-day weather forecast for the city, given its latitude and longitude.'''
    # url = BASE_URI + f'/data/2.5/forecast?lat={lat}&lon={lon}'
    # response = requests.get(url).json()
    # day_list = response['list']
    # return day_list

    url = urllib.parse.urljoin(BASE_URI, 'data/2.5/forecast')
    forecasts = requests.get(url, params={'lat': lat,  'lon': lon, 'units': 'metric'}).json()['list']

    return forecasts[::8]
    # return forecasts



def main():
    '''Ask user for a city and display weather forecast'''
    query = input("City?\n> ")
    city = search_city(query)

    if city:
        daily_forcasts = weather_forecast(city['lat'], city['lon'])

        # for i in range(0, len(daily_forcasts), 8):
        #     date = daily_forcasts[i]['dt_txt'][:-8]
        #     weather = daily_forcasts[i]['weather'][0]['main']
        #     temp = round(daily_forcasts[i]['main']['temp_max'])
        #     print(f"{date}: {weather} {temp}°C")

        for forecast in daily_forcasts:
            max_temp = round(forecast['main']['temp_max'])
            print(f"{forecast['dt_txt'][:10]}: {forecast['weather'][0]['main']} ({max_temp}°C)")

if __name__ == '__main__':
    try:
        # print(weather_forecast(48.8588897, 2.3200410217200766))
        print(search_city("London"))
        # while True:
        #     main()
    except KeyboardInterrupt:
        print('\nGoodbye!')
        sys.exit(0)
