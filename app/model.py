# -*- coding: UTF-8 -*-
from joblib import load
import pandas as pd
# 載入模型
RandomForestModel = load('./app/model/best_random_forest_model_4.joblib')

def predict(input):
    input_df = pd.DataFrame(input, columns=['CTR', 'KTE_kurt(IR)', 'SE_mean', 'KTE_skew(IR)'])
    pred = RandomForestModel.predict(input_df)[0]
    print(pred)
    return pred