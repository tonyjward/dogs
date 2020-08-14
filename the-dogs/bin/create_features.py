import logging
import datetime
from datetime import timezone
import os

from greyhound.features.min_time_long import min_time_long
from greyhound.features.race_winner import race_winner
from greyhound.features.runners import runners
from greyhound.features.running_style import running_style
from greyhound.features.pairwise_comparisons import pairwise_comparisons


DATA_DIR = '/home/d14xj1/repos/greyhound/data'
log_location = '/home/d14xj1/repos/greyhound/logs/'
log_level = logging.INFO

def create_features():
    "Create all features for use in modelling"
    
    # Set up Logging
    logging.basicConfig(filename = os.path.join(log_location, 'create_features.log'), 
                        level = log_level,
                        filemode='w')
    now_date = datetime.datetime.now(timezone.utc)
    now = now_date.strftime("%Y-%m-%d %H:%M:%S")
    msg = f'At {now} we kicked off a create_features job'
    print(msg)
    logging.info(msg)

    # create features
    # min_time_long()
    # race_winner()
    # runners()
    # running_style()
    pairwise_comparisons()

if __name__ == '__main__':
    create_features()