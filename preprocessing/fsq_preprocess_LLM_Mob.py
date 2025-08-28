import os
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
import argparse 

NUM_CONTEXT_STAY = 5 

def preprocess_fsq(city):
    BASE_DIR = os.path.join('./data/fsq', city)
    DATASET_PATH = os.path.join(BASE_DIR, f'dataSet_foursquare_{city}.csv')
    OUTPUT_DIR = BASE_DIR 

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(DATASET_PATH)
    df = df.sort_values(['user_id', 'start_day', 'start_min'])

    train_data_list, valid_data_list, test_samples = [], [], []

    for uid, user_data in df.groupby('user_id'):
        user_data = user_data.sort_values(['start_day', 'start_min'])

        non_test, test = train_test_split(user_data, test_size=0.2, shuffle=False)

        train, valid = train_test_split(non_test, test_size=0.125, shuffle=False)

        train_data_list.append(train)
        valid_data_list.append(valid)

        test_records = test.to_dict('records')
        for i in range(len(test_records) - NUM_CONTEXT_STAY):
            context = test_records[i:i+NUM_CONTEXT_STAY]
            target = test_records[i+NUM_CONTEXT_STAY]

            test_samples.append({
                'user_X': uid,
                'start_min_X': [r['start_min'] for r in context],
                'weekday_X': [r['weekday'] for r in context],
                'X': [r['location_id'] for r in context],  
                'start_min_Y': target['start_min'],      
                'weekday_Y': target['weekday'],           
                'Y': target['location_id']               
            })

    train_data = pd.concat(train_data_list)
    valid_data = pd.concat(valid_data_list)

    train_data.to_csv(os.path.join(OUTPUT_DIR, f'fsq_train_{city}.csv'), index=False,
                      columns=['id','user_id','location_id','latitude','longitude','start_day','start_min','weekday'])
    valid_data.to_csv(os.path.join(OUTPUT_DIR, f'fsq_valid_{city}.csv'), index=False,
                      columns=['id','user_id','location_id','latitude','longitude','start_day','start_min','weekday'])

    with open(os.path.join(OUTPUT_DIR, f'fsq_testset_{city}.pk'), 'wb') as f:
        pickle.dump(test_samples, f)

    print(f"数据集划分完成 ({city.upper()}): \n"
          f"- 训练集: {len(train_data)}条\n"
          f"- 验证集: {len(valid_data)}条\n"
          f"- 测试样本: {len(test_samples)}组")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Foursquare dataset for a given city after initial processing.")
    parser.add_argument('--city', type=str, required=True, choices=['tky', 'nyc'],
                        help='The city dataset to process (tky or nyc)')
    args = parser.parse_args()

    preprocess_fsq(city=args.city)