import numpy as np

def create_features(df):
    """ Creates features"""
    
    X = df.drop(['race_id','date_time', 'winning_box', 'favourite'], axis = 1)
    Y = df['winning_box'].to_numpy()
    
    return X, Y 

def trim_text(string, char = '_'):
    try:
        return string[:string.rindex(char)]
    except:
        return string

def features_used(columns, not_features = ['winning', 'race', 'favourite', 'date']):
    features_used = set([trim_text(string) for string in columns])

    for string in not_features:
        features_used.remove(string)

    features_used_str = str(features_used)

    # remove unwanted characters
    unwanted_chars = ['{', '}', "'"]
    for char in unwanted_chars:
        features_used_str = features_used_str.replace(char, '')

    return features_used_str