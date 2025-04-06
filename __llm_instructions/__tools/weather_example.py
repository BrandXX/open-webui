"""
title: Keyless Weather
author: spyci
author_url: https://github.com/open-webui
funding_url: https://github.com/open-webui
version: 0.1.1
"""

import os
import requests
import urllib.parse
import datetime


def get_city_info(city: str):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={urllib.parse.quote(city)}&count=1&language=en&format=json"
    response = requests.get(url)

    if response.status_code == 200:
        try:
            data = response.json()["results"][0]
            return data["latitude"], data["longitude"], data["timezone"]
        except (KeyError, IndexError):
            print(f"City '{city}' not found")
            return None
    else:
        print(f"Failed to retrieve data for city '{city}': {response.status_code}")
        return None


wmo_weather_codes = {
    "0": "Clear sky",
    "1": "Mainly clear, partly cloudy, and overcast",
    "2": "Mainly clear, partly cloudy, and overcast",
    "3": "Mainly clear, partly cloudy, and overcast",
    "45": "Fog and depositing rime fog",
    "48": "Fog and depositing rime fog",
    "51": "Drizzle: Light, moderate, and dense intensity",
    "53": "Drizzle: Light, moderate, and dense intensity",
    "55": "Drizzle: Light, moderate, and dense intensity",
    "56": "Freezing Drizzle: Light and dense intensity",
    "57": "Freezing Drizzle: Light and dense intensity",
    "61": "Rain: Slight, moderate and heavy intensity",
    "63": "Rain: Slight, moderate and heavy intensity",
    "65": "Rain: Slight, moderate and heavy intensity",
    "66": "Freezing Rain: Light and heavy intensity",
    "67": "Freezing Rain: Light and heavy intensity",
    "71": "Snow fall: Slight, moderate, and heavy intensity",
    "73": "Snow fall: Slight, moderate, and heavy intensity",
    "75": "Snow fall: Slight, moderate, and heavy intensity",
    "77": "Snow grains",
    "80": "Rain showers: Slight, moderate, and violent",
    "81": "Rain showers: Slight, moderate, and violent",
    "82": "Rain showers: Slight, moderate, and violent",
    "85": "Snow showers slight and heavy",
    "86": "Snow showers slight and heavy",
    "95": "Thunderstorm: Slight or moderate",
    "96": "Thunderstorm with slight and heavy hail",
    "99": "Thunderstorm with slight and heavy hail",
}


def fetch_weather_data(base_url, params):
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            return f"Error fetching weather data: {data['message']}"
        return data
    except requests.RequestException as e:
        return f"Error fetching weather data: {str(e)}"


def format_date(date_str, date_format="%Y-%m-%dT%H:%M", output_format="%I:%M %p"):
    dt = datetime.datetime.strptime(date_str, date_format)
    return dt.strftime(output_format)


class Tools:
    def __init__(self):
        self.citation = True
        pass

    def get_future_weather_week(self, city: str) -> str:
        """
        Get the weather for the next week for a given city.
        :param city: The name of the city to get the weather for.
        :return: The current weather information or an error message.
        """
        if not city:
            return """The location has not been defined by the user, so weather cannot be determined."""

        city_info = get_city_info(city)
        if not city_info:
            return """Error fetching weather data"""

        lat, lng, tmzone = city_info
        print(f"Latitude: {lat}, Longitude: {lng}, Timezone: {tmzone}")

        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "daily": [
                "weather_code",
                "temperature_2m_max",
                "temperature_2m_min",
                "uv_index_max",
                "precipitation_probability_max",
                "wind_speed_10m_max",
            ],
            "current": "temperature_2m",
            "timezone": tmzone,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "forecast_days": 7,
        }

        data = fetch_weather_data(base_url, params)
        if isinstance(data, str):
            return data

        formatted_timestamp = format_date(data["current"]["time"])
        data["daily"]["time"][0] += " (Today)"

        mapped_data = {
            date: {
                "weather_description": wmo_weather_codes[
                    str(data["daily"]["weather_code"][i])
                ],
                "temperature_max_min": f'{data["daily"]["temperature_2m_max"][i]} {data["daily_units"]["temperature_2m_max"]} / {data["daily"]["temperature_2m_min"][i]} {data["daily_units"]["temperature_2m_min"]}',
                "uv_index_max": f'{data["daily"]["uv_index_max"][i]} {data["daily_units"]["uv_index_max"]}',
                "precipitation_probability_max": f'{data["daily"]["precipitation_probability_max"][i]} {data["daily_units"]["precipitation_probability_max"]}',
                "max_wind_speed": f'{data["daily"]["wind_speed_10m_max"][i]} {data["daily_units"]["wind_speed_10m_max"]}',
            }
            for i, date in enumerate(data["daily"]["time"])
        }

        return f"""
Give a weather description for the next week, include the time of the data ({formatted_timestamp} {data['timezone_abbreviation']} in {city}):
Show a standard table layout of each of these days: {mapped_data}
Include a one sentence summary of the week at the end."""

    def get_current_weather(self, city: str) -> str:
        """
        Get the current weather for a given city.
        :param city: The name of the city to get the weather for.
        :return: The current weather information or an error message.
        """
        if not city:
            return """The location has not been defined by the user, so weather cannot be determined."""

        city_info = get_city_info(city)
        if not city_info:
            return """Error fetching weather data"""

        lat, lng, tmzone = city_info
        print(f"Latitude: {lat}, Longitude: {lng}, Timezone: {tmzone}")

        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lng,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "wind_speed_10m",
                "weather_code",
            ],
            "timezone": tmzone,
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "forecast_days": 1,
        }

        data = fetch_weather_data(base_url, params)
        if isinstance(data, str):
            return data

        formatted_timestamp = format_date(data["current"]["time"])
        data["current"]["weather_code"] = wmo_weather_codes[
            str(data["current"]["weather_code"])
        ]
        formatted_data = ", ".join(
            [
                f"{x} ({data['current_units'][x]}) = '{data['current'][x]}'"
                for x in data["current"].keys()
            ]
        ).replace("weather_code", "weather_description")

        return f"""
Give a weather description, include the time of the data ({formatted_timestamp} {data['timezone_abbreviation']} in {city}):
Include this data: [{formatted_data}]
Ensure you mention the real temperature and the "feels like"(apparent_temperature) temperature. Convert all numbers to integers.
Keep response as brief as possible."""
