import os
import pickle
import glob
import itertools
import re
import traceback
import pandas as pd
from sys import argv
import logging
import datetime
from datetime import timezone

from greyhound import delete_all, create_all, insert_stadium, insert_dog, insert_race, insert_position, insert_comparison

DATA_DIR = '/home/d14xj1/repos/greyhound/data'
log_location = '/home/d14xj1/repos/greyhound/logs/'
log_level = logging.INFO

def populate_database(search_term):

    # Set up Logging
    now_date = datetime.datetime.now(timezone.utc)
    now = now_date.strftime("%Y-%m-%d %H:%M:%S")
    logging.basicConfig(filename = os.path.join(log_location, search_term + '.log'), 
                        level = log_level,
                        filemode='w')
    path_name = os.path.join(log_location, 'database_failure_' + search_term + '.txt')
    
    try:
        if os.path.exists(path_name):
            os.remove(path_name)
    except:
        msg = f"Error deleteing log file {path_name}"
        print(msg)
        logging.exception(msg)

    msg = f'At {now} we kicked off a populate_database job with search term {search_term}'
    print(msg)
    logging.info(msg)

    # prepare database by deleting all tables and starting again
    delete_data = input('Would you like to delete all the data in database (y/n)? ')
    if delete_data == 'y':
        msg = 'Attempting to delete all the data now'
        print(msg)
        logging.info(msg)
        delete_all()
        create_all()
    else:
        msg = 'We are appending to the existing data in the database (rather than deleting everything first)'
        print(msg)
        logging.info(msg)

    # keep track of failures with their error messages
    failed_race_id = []
    failed_error = []

    # regex for extracting time string
    regex_time = re.compile('\d+:\d+')
    regex_race_no = re.compile('\d+')

    # occasionally a race won't have data on the time of race
    # in these situations we want to insert the time for the previous race
    # here we initialise this time
    last_time_string = '12:00'
    last_race_no = 1

    pickle_paths = glob.glob(DATA_DIR + '/race_data_UK-United-Kingdom_' + search_term + '*')

    for path in pickle_paths:

        # load races for given path
        race_data = pickle.load(open(path, "rb"))

        # update user with progress
        msg = f"Inserting {len(race_data)} races from {path}"
        logging.info(msg)
        print(msg)
    
        for specific_race in race_data:

            try:

                msg = f"Inserting data for race id : {specific_race['race_id']}"
                logging.debug(msg)

                # grab_race_data
                # first we need to sort the race results by box order
                # This is so that we always get the same pairwise comparisons
                specific_race_data = specific_race['results_df'].sort_values(by = 'box').reset_index(drop = True)

                # Handle Missing Data
                try:
                    # extract time string
                    time_string = regex_time.findall(specific_race['race_no_time'])[0]
                    # update last obvserved time
                    last_time_string = time_string 
                except IndexError:
                    msg = f"No time data is available for race_id {specific_race['race_id']}. Using last seen time of {last_time_string}"
                    logging.exception(msg)
                    time_string = last_time_string

                try:
                    # extract time string
                    race_no = int(regex_race_no.findall(specific_race['race_no_time'])[0])
                    # update last obvserved time
                    last_race_no = race_no
                except ValueError:
                    msg = f"The race number could not be converted to integer for {specific_race['race_id']}"
                    logging.exception(msg)
                    print(msg) 
                except IndexError:
                    print(f"No race number is available for race_id {specific_race['race_id']}. Using last seen race number of {last_race_no} plus 1")
                    race_no = last_race_no + 1

                # insert stadium
                insert_stadium(specific_race['stadium_id'], specific_race['stadium_name'])

                # insert race
                insert_race(race_id = specific_race['race_id'], 
                        stadium_id = specific_race['stadium_id'],
                        race_name = specific_race['race_name'],
                        race_no = race_no,
                        date_time = specific_race['date'] + ' ' + time_string,
                        going = specific_race['going'],
                        grade = specific_race['grade'],
                        distance = specific_race['distance'])

                # insert dogs and positions      
                fin_regex = re.compile('\d')      

                for index, row in specific_race_data.iterrows():

                    insert_dog(row['dog_id'], row['name'])

                    insert_position(race_id = specific_race['race_id'],
                                    dog_id = row['dog_id'],
                                    fin = fin_regex.findall(row['fin'])[0],
                                    time = row['time'],
                                    dist = row['dist'],
                                    stime = row['stime'],
                                    box = row['box'],
                                    posts = row['posts'],
                                    sp = row['sp'],
                                    kg = row['kg'],
                                    comment = row['comment'])
                # insert comparisons

                # work out the 15 combinations of box positions that we will compare
                race_positions = list(specific_race_data.index)
                combinations = list(itertools.combinations(race_positions,2))

                for combination in combinations:

                    insert_comparison(race_id = specific_race['race_id'],
                                    dog_A_id = specific_race_data.loc[combination[0]]['dog_id'],
                                    dog_B_id = specific_race_data.loc[combination[1]]['dog_id'],
                                    time_A = specific_race_data.loc[combination[0]]['time'],
                                    time_B = specific_race_data.loc[combination[1]]['time'],
                                    stime_A = specific_race_data.loc[combination[0]]['stime'],
                                    stime_B = specific_race_data.loc[combination[1]]['stime'])
            except:
                msg = f"Populate_database failed for {specific_race['race_id']}"
                logging.exception(msg)
                error_message = traceback.format_exc()
                print(error_message)
                failed_race_id.append(specific_race['race_id'])
                failed_error.append(error_message)

    # save failures for later inspection
    try:
        failures_df = pd.DataFrame(list(zip(failed_race_id, failed_error)), columns = ['race_id', 'error'])
        failures_df.to_csv(path_name, index = False)
    except:
        msg = 'Error saving failures to disk'
        print(msg)
        logging.exception(msg)

    # Closing Message
    now_date = datetime.datetime.now(timezone.utc)
    now = now_date.strftime("%Y-%m-%d %H:%M:%S")
    msg = f'At {now} we finished populating the database with search term: {search_term}'
    print(msg)
    logging.info(msg)


if __name__ == '__main__':
    script, search_term = argv
    populate_database(search_term)
