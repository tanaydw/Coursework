import numpy as np
import pandas as pd
import xgboost as xgb

from tqdm import tqdm
from datetime import datetime


def apk(actual, predicted, k=10):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    if not actual:
        return 0.0

    return score / min(len(actual), k)


def calculate_mapk(valid_df, pred_df):
    score = 0
    test_bikers = pred_df['biker_id'].tolist()

    for biker in test_bikers:
        actual = valid_df[(valid_df['biker_id'] == biker) & (valid_df['like'] == 1)]['tour_id'].tolist()
        predicted = pred_df[(pred_df['biker_id'] == biker)]['tour_id'].tolist()[0].split()
        score += apk(actual, predicted)

    return score / len(test_bikers)


def predict_validation_score(pred, data_content):
    X_val = data_content.train_df.loc[data_content.split_point + 1:]
    test_bikers = X_val['biker_id'].unique()
    X_val['pred'] = pred
    pred_df = pd.DataFrame(test_bikers, columns=['biker_id'])
    pred_df['tour_id'] = ''

    for idx, _ in pred_df.iterrows():
        temp_df = X_val[X_val['biker_id'] == pred_df.loc[idx, 'biker_id']]
        temp_df = temp_df.sort_values(by='pred', ascending=False)
        pred_df.loc[idx, 'tour_id'] = ' '.join(temp_df['tour_id'].tolist())

    return calculate_mapk(X_val, pred_df)


def train_xgbr(data_content):
    print('Training XGBoost...', flush=True)
    model = xgb.XGBRanker(**data_content.best_params)

    model.fit(data_content.X_train, data_content.y_train, group=data_content.train_group, verbose=False,
              eval_set=[(data_content.X_valid, data_content.y_valid)], eval_group=[data_content.valid_group])

    print('Model trained...', flush=True)
    pred = model.predict(data_content.X_valid)
    score = predict_validation_score(pred, data_content)
    print('Validation Score = ' + str(score), flush=True)

    return model


def inference(pred, data_content):
    X_val = data_content.test_df
    test_bikers = X_val['biker_id'].unique()
    X_val['pred'] = pred
    pred_df = pd.DataFrame(test_bikers, columns=['biker_id'])
    pred_df['tour_id'] = ''

    for idx, _ in pred_df.iterrows():
        temp_df = X_val[X_val['biker_id'] == pred_df.loc[idx, 'biker_id']]
        temp_df = temp_df.sort_values(by='pred', ascending=False)
        pred_df.loc[idx, 'tour_id'] = ' '.join(temp_df['tour_id'].tolist())

    return pred_df


def training_and_inference(data_content):
    data_content.xgbr = train_xgbr(data_content)
    data_content.pred = data_content.xgbr.predict(data_content.X_test)
    data_content.pred_df = inference(data_content.pred, data_content)
    data_content.pred_df.to_csv('ME17B180_ME17B039_' + str(data_content.submission_number) + '.csv', index=False)
    data_content.submission_number += 1

    return data_content
