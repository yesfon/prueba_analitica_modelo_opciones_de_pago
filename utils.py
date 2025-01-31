import os

def kaggle_format(data, labels):
    data['ID'] = data.iloc[:, 0].astype(str) + '#' + \
               data.iloc[:, 1].astype(str) + '#' + \
               data.iloc[:, 2].astype(str)
    data['var_rpta_alt'] = labels
    data.drop(data.columns[:-2], axis=1, inplace=True)
    os.makedirs('logs', exist_ok=True)
    data.to_csv('logs/submission.csv', index=False)