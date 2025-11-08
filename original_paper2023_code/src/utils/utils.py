import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from src.utils.paths import get_model_dir


def plotCurves(train_MSE, train_rank, test_MSE, EPOCH, name):
    exp_MSE = 0.00021
    print(f"best test MSE: {np.min(test_MSE):.12f};   experiment MSE error: 0.000015794911;   data variance: "
          f"0.045936428010")
    exp_MSE = np.full(len(train_MSE), exp_MSE)
    
    # Get model directory using centralized path utility
    clean_name = name.split('.')[0]
    model_dir = get_model_dir(clean_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    plt.plot(train_MSE[1:], 'r', label="train_MSE")
    plt.plot(train_rank[1:], 'y', label="train_rank")
    plt.plot(test_MSE[1:], 'b', label="test")
    plt.plot(exp_MSE[1:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig(str(model_dir / 'curve.png'))

    plt.clf()

    plt.plot(train_MSE[EPOCH // 2:], 'r', label="train")
    plt.plot(test_MSE[EPOCH // 2:], 'b', label="test")
    plt.plot(exp_MSE[EPOCH // 2:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    plt.savefig(str(model_dir / 'curve2.png'))

def savePrediction(data, prediction, std, mean, name):
    result = np.zeros([len(data.y), 2])
    prediction = prediction * std + mean
    prediction = prediction.cpu().detach().numpy()
    result[:, 0] = prediction[0:len(data.y), 0]  # pre
    result[:, 1] = np.asarray([d * std + mean for d in data.y.cpu()]).flatten()

    df = pd.DataFrame(result)

    # Get model directory using centralized path utility
    clean_name = name.split('.')[0]
    model_dir = get_model_dir(clean_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Fix for newer pandas versions
    with pd.ExcelWriter(str(model_dir / 'Prediction.xlsx'), engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='page_1', float_format='%.9f')


def printInstance(data, prediction, std, mean):
    result = np.zeros([len(data.y), 2])
    prediction = prediction * std + mean
    prediction = prediction.cpu().detach().numpy()
    result[:, 0] = prediction[0:len(data.y), 0]  # pre
    result[:, 1] = np.asarray([d * std + mean for d in data.y.cpu()])
    print('w:')
    print(data.x[0])
    print('a_upper:')
    print(data.edge_attr[0][:, 0])
    print('a_lower:')
    print(data.edge_attr[0][:, 1])
    print('gt:')
    print(result[0, 1])
    print('prediction')
    print(prediction[0])