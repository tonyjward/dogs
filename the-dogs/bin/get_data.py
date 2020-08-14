
from selenium.webdriver.common.keys import Keys

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import time
import pickle
import os
from sys import argv
import datetime
from datetime import timezone

from decouple import config
import logging
 
from greyhound import start_selenium, get_driver, get_login_creds, check_popups, login, search, get_race_pages, get_race_links, get_race_data
from greyhound import deallocate_vm

'''
Example usage

python -m get_data data 'UK United Kingdom' All Jun 2020 Jun 2020
python -m get_data data 'UK United Kingdom' Henlow Jun 2020 Jun 2020
'''

# Logging level and location
log_location = '/home/d14xj1/repos/greyhound/logs/log_getdata.log'
log_level = logging.INFO


def main(country, stadium, start_month, start_year, end_month, end_year):
    
    # Set up Logging
    now_date = datetime.datetime.now(timezone.utc)
    now = now_date.strftime("%Y-%m-%d %H:%M:%S")

    logging.basicConfig(filename = log_location, level = log_level)
    
    msg = f'''At {now} we kicked off a get_data job with details 
                country: {country}
                stadium: {stadium} 
                start_month: {start_month}
                start_year: {start_year}
                end_month: {end_month}
                end_year: {end_year}'''
    print(msg)
    logging.info(msg)
    
    # If race data already exists do nothing
    suffix = '_' + country.replace(' ','-') + '_' + stadium.replace(' ','-') + '_' + start_month + '_' + start_year + '_' + end_month + '_' + end_year + '.p'
    path_data = os.path.join(data_dir, 'race_data' + suffix)

    if os.path.exists(path_data):
        msg = f"Race data exists for {country}, {stadium}, {start_month}, {start_year}, {end_month}, {end_year}"
        print(msg)
        logging.info(msg)
    else:
        try:
            start_selenium()
            driver = get_driver()
            username, password = get_login_creds()
            login(driver, username, password)

            search(driver, 
                    country = country, 
                    stadium = stadium,
                    start_month = start_month, 
                    start_year = start_year, 
                    end_month = end_month, 
                    end_year = end_year
                )

            path_pages = os.path.join(data_dir, 'race_pages' + suffix)
            path_links = os.path.join(data_dir, 'race_links' + suffix)
            path_failures = os.path.join(data_dir, 'failed_links' + suffix)

            # Race pages
            if os.path.exists(path_pages):
                race_pages = pickle.load(open(path_pages, "rb"))
                msg = f"Race pages already exist and so were loaded from file"
                print(msg)
                logging.info(msg)
            else: 
                race_pages = get_race_pages(driver)
                pickle.dump(race_pages, open(path_pages, "wb"))

            # Race links
            if os.path.exists(path_links):
                race_links = pickle.load(open(path_links, "rb"))
                msg = f"Race links already exist and so were loaded from file"
                print(msg)
                logging.info(msg)
            else: 
                race_links = get_race_links(driver, race_pages)
                if len(race_links) == 0:
                    msg = "We couldn't find any links for that race"
                    print(msg)
                    logging.warning(msg)
                else:
                    pickle.dump(race_links, open(path_links, "wb"))

            # Race data
            race_data, failures, failed_links = get_race_data(driver, username, password, race_links)
            if len(race_data) == 0:
                msg = "We couldn't find any race data for those links"
                print(msg)
                logging.warning(msg)
            else:       
                pickle.dump(race_data, open(path_data, "wb"))
            if failures:
                msg = "We found some failed links and have stored them for inspection"
                print(msg)
                logging.warning(msg)
                pickle.dump(failed_links, open(path_failures, "wb"))

            # Shutdown and deallocate vm
            group_name = config('GROUP_NAME')
            vm_name = config('VM_NAME')
            msg = f'Shutting down vm: {vm_name}'
            print(msg)
            logging.info(msg)
            time.sleep(2)
            deallocate_vm(group_name, vm_name)
        except Exception as error:
            msg = f'''get_data encountered an error whilst searching for race with details 
                country: {country}
                stadium: {stadium} 
                start_month: {start_month}
                start_year: {start_year}
                end_month: {end_month}
                end_year: {end_year}'''
            print(msg)
            logging.exception(msg)
     
if __name__ == '__main__':
    script, data_dir, country, stadium, start_month, start_year, end_month, end_year = argv
    main(country, stadium, start_month, start_year, end_month, end_year)
