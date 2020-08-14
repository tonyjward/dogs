import pandas as pd
import datetime
from darksky.api import DarkSky, DarkSkyAsync
from darksky.types import languages, units, weather
import datetime
import pickle
import traceback
import os
from datetime import datetime as dt
from dateutil.relativedelta import relativedelta
from decouple import config

API_KEY = config('DARK_SKY_API_KEY')

data_dir = '/home/d14xj1/repos/greyhound/data/weather'

# date ranges for weather collection
start_date = dt(2002, 1, 1, 12) 
end_date = dt(2020, 4,1,12)


def get_weather_data():
    
    # establish connection to dark sky
    darksky = DarkSky(API_KEY)

    stadium_locations = pd.read_csv(os.path.join(data_dir, 'stadium_locations.csv'))

    # list of timestamps to collect data for
    time_stamps = []
    current_date = start_date
    while current_date < end_date:
        time_stamps.append(current_date)
        current_date += relativedelta(days = +1)

    # TODO: REMOVE THIS
    # time_stamps = time_stamps[:5]

    # get weather data
    weather_forecast = {}
    i = 0
    for index, row in stadium_locations.iterrows():
        try:
            print(f"Getting data for stadium number {i} / {stadium_locations.shape[0]}: {row['stadium_name']}")
            forecast_list = []
            for time_stamp in time_stamps:
                forecast = darksky.get_time_machine_forecast(row['latitude'], row['longitude'],                                              
                                                            extend=False, # default `False`
                                                            lang=languages.ENGLISH, # default `ENGLISH`
                                                            values_units=units.AUTO, # default `auto`
                                                            exclude=[weather.MINUTELY, weather.ALERTS], # default `[]`,
                                                            timezone='UTC', # default None - will be set by DarkSky API automatically
                                                            time=time_stamp)
                forecast_list.append(forecast)
            weather_forecast[row['stadium_name']] = forecast_list
        except Exception:
            print(f"We failed to get weather data for {row['stadium_name']}")
            traceback.print_exc()
        finally:
            i += 1

    # save weather data
    pickle.dump(weather_forecast, open(os.path.join(data_dir, 'weather_forecast.p'),"wb"))

    print('Weather data has been obtained')

if __name__ == '__main__':
    get_weather_data()







