# -*- coding: UTF-8 -*-
import app.model as model
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def postInput():
    # 取得前端傳來的數值
    data = request.get_json()
    CTR = data['CTR']
    KTE_kurt = data['KTE_kurt(IR)']
    SE_mean = data['SE_mean']
    KTE_skew = data['KTE_skew(IR)']
    x = [CTR, KTE_kurt, SE_mean, KTE_skew]
    print(x)
    input = np.array([x])
    print(input)
    # 預測
    result = model.predict(input)

    return jsonify({'result': str(result)})