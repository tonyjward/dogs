import os
import pickle
import traceback
import pandas as pd
import logging
import datetime
from datetime import timezone
import psycopg2
from psycopg2 import Error

DATA_DIR = '/home/d14xj1/repos/greyhound/data/weather'
log_location = '/home/d14xj1/repos/greyhound/logs/'
log_level = logging.INFO

from greyhound import create_weather, insert_weather

def populate_weather():

    # Set up Logging
    now_date = datetime.datetime.now(timezone.utc)
    now = now_date.strftime("%Y-%m-%d %H:%M:%S")

    logging.basicConfig(filename = os.path.join(log_location, 'weather.log'), 
                        level = log_level,
                        filemode='w')

    msg = f'At {now} we kicked off a populate_weather job'
    print(msg)
    logging.info(msg)

    # first create the weather table
    create_weather()

    # load required files
    weather_forecast = pickle.load(open(os.path.join(DATA_DIR,'weather_forecast.p'), "rb"))
    stadium_locations = pd.read_csv(os.path.join(DATA_DIR, 'stadium_locations.csv'))

    # inserting data
    for stadium_name, forecast_list in weather_forecast.items():
        print(f"Inserting data for {stadium_name}")
        
        stadium_id = stadium_locations.loc[stadium_locations['stadium_name'] ==stadium_name, 'stadium_id'].values[0]
        
        for forecast in forecast_list: # forecast list has 6300 forecasts
            for hourly_forecast in forecast.hourly.data: 
                
                # each forecast has 24 hourly_forecasts
                insert_weather(stadium_id = stadium_id, 
                            date_time = hourly_forecast.time,
                            precip_type = hourly_forecast.precip_type, 
                            precip_probability = hourly_forecast.precip_probability, 
                            precip_intensity = hourly_forecast.precip_intensity, 
                            temperature = hourly_forecast.temperature, 
                            humidity = hourly_forecast.humidity, 
                            pressure = hourly_forecast.pressure)

    # Closing Message
    now_date = datetime.datetime.now(timezone.utc)
    now = now_date.strftime("%Y-%m-%d %H:%M:%S")

    msg = f'At {now} we finished populating the database with weather data:'
    print(msg)
    logging.info(msg)

def add_24_hours():
    try:
        msg = "Attempting to add the 24 hours variable to the weather table"
        print(msg)
        logging.info(msg)
        
        # Connect to database
        connect_str = "dbname='greyhounds' user='postgres' host='localhost' password='postgres'"
        conn_psql = psycopg2.connect(connect_str)
        cursor = conn_psql.cursor()

        cursor.execute("""
        ALTER TABLE weather ADD COLUMN date_time_plus24 timestamptz;
        UPDATE weather set date_time_plus24 = date_time + interval '24 hours';
        CREATE INDEX idx_weather_inverse ON weather (stadium_id, date_time, date_time_plus24 desc);
        CLUSTER weather USING idx_weather_inverse;
        """)

        conn_psql.commit()
        msg = "The 24 hours variable has been added to the weather table"
        print(msg)
        logging.info(msg)

    except (Exception, psycopg2.DatabaseError) as error:
        msg = "Error whilst adding 24 hours column to weather table"
        print(msg)
        logging.exception(msg)

    finally:
        if(conn_psql):
            cursor.close()
            conn_psql.close()
            print("PostgreSQL connection is closed")

def recent_weather():
    try:
        msg = "Attempting to create the recent_weather_1hourly table"
        print(msg)
        logging.info(msg)
        
        # Connect to database
        connect_str = "dbname='greyhounds' user='postgres' host='localhost' password='postgres'"
        conn_psql = psycopg2.connect(connect_str)
        cursor = conn_psql.cursor()

        cursor.execute("""
        DROP TABLE IF EXISTS recent_weather_1hourly;
        CREATE TABLE recent_weather_1hourly AS
        SELECT r.race_id, w.temperature, w.humidity,
        100 * w.precip_probability * w.precip_intensity as precip_combined,
        DATE_PART('day', r.date_time - w.date_time) * 24 + DATE_PART('hour', r.date_time - w.date_time) as hours_ago

        FROM races r
        LEFT JOIN weather w ON
            r.stadium_id = w.stadium_id AND
            r.date_time > w.date_time AND
            r.date_time < w.date_time_plus24;
        CREATE INDEX idx_recent_weather_1hourly ON recent_weather_1hourly(race_id);
        """)

        conn_psql.commit()
        msg = "The recent_weather_1hourly table has been created"
        print(msg)
        logging.info(msg)


    except (Exception, psycopg2.DatabaseError) as error:
        msg = "Error whilst creating the recent_weather_1hourly table"
        print(msg)
        logging.exception(msg)

    finally:
        if(conn_psql):
            cursor.close()
            conn_psql.close()
            print("PostgreSQL connection is closed")
    
def aggregate_weather():
    try:
        msg = "Attempting to create the recent_weather_banded table"
        print(msg)
        logging.info(msg)
        
        # Connect to database
        connect_str = "dbname='greyhounds' user='postgres' host='localhost' password='postgres'"
        conn_psql = psycopg2.connect(connect_str)
        cursor = conn_psql.cursor()

        cursor.execute("""
        DROP TABLE IF EXISTS recent_weather_banded;
        CREATE TABLE recent_weather_banded AS
        SELECT race_id, hours_ago_b as hours_ago, avg(temperature) as temperature, sum(precip_combined) as precip_combined, 
        avg(humidity) as humidity
        FROM
        (SELECT *, CASE WHEN hours_ago = 0 THEN 'a. 0'
                        WHEN hours_ago = 1 THEN 'b. 1'
                        WHEN hours_ago BETWEEN  2 AND 3 THEN 'c. 2-3'
                        WHEN hours_ago BETWEEN  4 AND 6 THEN 'd. 4-6'
                        WHEN hours_ago BETWEEN  7 AND 12 THEN 'e. 7-12'
                        WHEN hours_ago BETWEEN  13 AND 18 THEN 'f. 13-18'
                        WHEN hours_ago BETWEEN  19 AND 24 THEN 'g. 19-24'
                        END as hours_ago_b
        FROM recent_weather_1hourly) subquery
        GROUP BY race_id, hours_ago_b;
        CREATE INDEX idx_recent_weather_banded ON recent_weather_banded(race_id);
        """)

        conn_psql.commit()
        msg = "The recent_weather_banded table has been created"
        print(msg)
        logging.info(msg)

    except (Exception, psycopg2.DatabaseError) as error:
        msg = "Error whilst creating the recent_weather_banded table"
        print(msg)
        logging.exception(msg)
        
    finally:
        if(conn_psql):
            cursor.close()
            conn_psql.close()
            print("PostgreSQL connection is closed")


if __name__ == '__main__':
    populate_weather()
    add_24_hours()
    recent_weather()
    aggregate_weather()
