# -*- coding: UTF-8 -*-
from joblib import dump, load
import pandas as pd
# 載入模型
RandomForestModel = load('./app/model/best_random_forest_model_4.joblib')

def predict(input):
    pred = RandomForestModel.predict(input)[0]
    input_df = pd.DataFrame(input, columns=['CTR', 'KTE_kurt(IR)', 'SE_mean', 'KTE_skew(IR)'])
    print(pred)
    return pred
