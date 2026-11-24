#!/usr/bin/env python

import argparse
import os
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


def train(data_dir, model_file):
    data = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    data = data.sort_values('QueryId', kind='stable')
    features = [column for column in data.columns if column not in ('QueryId', 'Label')]

    queries = np.random.default_rng(42).permutation(data['QueryId'].unique())
    valid_mask = data['QueryId'].isin(queries[:len(queries) // 5])
    train_data = data.loc[~valid_mask]
    valid_data = data.loc[valid_mask]
    valid_pool = Pool(valid_data[features], valid_data['Label'], group_id=valid_data['QueryId'])

    params = dict(
        loss_function='YetiRank',
        eval_metric='NDCG:top=10;type=Exp',
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        thread_count=4,
        allow_writing_files=False,
    )
    model = CatBoostRanker(**params)
    model.fit(
        train_data[features], train_data['Label'], group_id=train_data['QueryId'],
        eval_set=valid_pool, early_stopping_rounds=100, use_best_model=True, verbose=100,
    )
    params['iterations'] = model.tree_count_
    model = CatBoostRanker(**params)
    model.fit(data[features], data['Label'], group_id=data['QueryId'], verbose=100)
    model.save_model(model_file)


def predict(data_dir, model_file, submission_file):
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    model = CatBoostRanker()
    model.load_model(model_file)
    scores = model.predict(test[model.feature_names_], thread_count=4)
    submission = pd.DataFrame({'ObjectId': test['ObjectId'], 'Score': scores})
    submission.to_csv(submission_file, index=False)


def main():
    parser = argparse.ArgumentParser(description='Learning to rank homework solution')
    parser.add_argument('--train', action='store_true', help='обучить модель и сохранить её в model_file')
    parser.add_argument('--model_file', required=True, help='файл модели')
    parser.add_argument('--submission_file', help='куда записать сабмишн для Kaggle')
    parser.add_argument('data_dir', help='папка с данными соревнования')
    args = parser.parse_args()

    start = timer()

    if args.train:
        train(args.data_dir, args.model_file)
    else:
        if args.submission_file is None:
            parser.error('без --train нужен --submission_file')
        predict(args.data_dir, args.model_file, args.submission_file)

    print(f'finished, elapsed = {timer() - start:.3f}')


if __name__ == '__main__':
    main()
