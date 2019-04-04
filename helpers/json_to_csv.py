import json
import os
import pandas as pd

path_to_intents = os.path.join('..', 'data', 'raw')
intents = os.listdir(path_to_intents)

print('\nloading data...\n')
data = {}
for intent in intents:
    train = {}
    validate = {}
    with open(os.path.join(path_to_intents, intent,'train_'+intent+'_full.json'), encoding='latin-1') as f:
        train = json.load(f)
    
    with open(os.path.join(path_to_intents, intent,'validate_'+intent+'.json'), encoding='latin-1') as f:
        validate = json.load(f)
    
    data[intent] = train[intent] + validate[intent]
    print("\tloaded", len(data[intent]), "entries for", intent)

print('\nlooking for entities...\n')
for intent in intents:
    entities = set()

    for row in data[intent]:
        for item in row['data']:
            if 'entity' in item:
                entities.add(item['entity'])

    print("\tFor", intent, "entities are:")
    [print("\t\t", x) for x in list(entities)]
    data[intent + '_entities'] = list(entities)

print('\nmaking dataframes...\n')
# print(pd.DataFrame([x['data'] for x in data['AddToPlaylist']]).head())
dfs = {}
for intent in intents:
    rows = []
    for row in data[intent]:
        _row = {}
        _row['text'] = []
        for item in row['data']:
            _row['text'].append(item['text'])

            for entity in data[intent + '_entities']:
                if 'entity' in item:
                    _row[item['entity']] = item['text']
        
        _row['text'] = ' '.join([x.strip() for x in _row['text']])
        rows.append(_row)

    dfs[intent] = pd.DataFrame(
        data = rows,
        columns=(['text'] + data[intent + '_entities'])
        )

    dfs[intent].to_csv(os.path.join(path_to_intents, intent,intent + '.csv'))
    print("\twrote dataframe to " + intent + '.csv')