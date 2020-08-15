import numpy as np

def create_features(df):
    """ Creates features"""
    
    X = df.drop(['date_time', 'winning_box', 'benchmark'], axis = 1)
    Y = df['winning_box'].to_numpy()
    
    return X, Y 

def trim_text(string, char = '_'):
    try:
        return string[:string.rindex(char)]
    except:
        return string

def features_used(columns, not_features = ['winning', 'benchmark', 'date']):
    features_used = set([trim_text(string) for string in columns])

    for string in not_features:
        features_used.remove(string)

    features_used_str = str(features_used)

    # remove unwanted characters
    unwanted_chars = ['{', '}', "'"]
    for char in unwanted_chars:
        features_used_str = features_used_str.replace(char, '')

    return features_used_str

def accuracy_error(preds, train_data):
    "A custom loss metric for multi-class accuracy to be used in lightgbm cross validation"    
    
    preds = preds.reshape(-1, 6, order = 'F')
    preds_class = preds.argmax(axis = 1)
    labels = train_data.get_label()
    model_correct = (preds_class == labels)
    accuracy = model_correct.mean()
    
    return 'accuracy', accuracy, True