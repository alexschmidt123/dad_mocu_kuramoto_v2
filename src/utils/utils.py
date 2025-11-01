import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plotCurves(train_MSE, train_rank, test_MSE, EPOCH, name, output_dir='../models/'):
    exp_MSE = 0.00021
    print(f"best test MSE: {np.min(test_MSE):.12f};   experiment MSE error: 0.000015794911;   data variance: "
          f"0.045936428010")
    exp_MSE = np.full(len(train_MSE), exp_MSE)
    plt.plot(train_MSE[1:], 'r', label="train_MSE")
    plt.plot(train_rank[1:], 'y', label="train_rank")
    plt.plot(test_MSE[1:], 'b', label="test")
    plt.plot(exp_MSE[1:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    name = name.split('.')[0]
    output_dir = output_dir if output_dir.endswith('/') else output_dir + '/'
    # If name is empty or None, save directly to output_dir; otherwise create subfolder
    if name and name.strip() and name != '__USE_OUTPUT_DIR__':
        save_path = output_dir + name + '/'
        # Ensure directory exists
        import os
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(save_path + 'curve.png')
    else:
        plt.savefig(output_dir + 'curve.png')

    plt.clf()

    plt.plot(train_MSE[EPOCH // 2:], 'r', label="train")
    plt.plot(test_MSE[EPOCH // 2:], 'b', label="test")
    plt.plot(exp_MSE[EPOCH // 2:], '-', label="experiment")
    plt.xlabel('epoch')
    plt.ylabel('Mean Square Error')
    plt.legend()
    if name and name.strip() and name != '__USE_OUTPUT_DIR__':
        plt.savefig(save_path + 'curve2.png')
    else:
        plt.savefig(output_dir + 'curve2.png')

def savePrediction(data, prediction, std, mean, name, output_dir='../models/'):
    result = np.zeros([len(data.y), 2])
    prediction = prediction * std + mean
    prediction = prediction.cpu().detach().numpy()
    result[:, 0] = prediction[0:len(data.y), 0]  # pre
    result[:, 1] = np.asarray([d * std + mean for d in data.y.cpu()]).flatten()

    df = pd.DataFrame(result)
    output_dir = output_dir if output_dir.endswith('/') else output_dir + '/'
    name = name.split('.')[0] if name else ''

    # If name is empty or None, save directly to output_dir; otherwise create subfolder
    if name and name.strip() and name != '__USE_OUTPUT_DIR__':
        save_path = output_dir + name + '/'
        # Ensure directory exists
        import os
        os.makedirs(save_path, exist_ok=True)
        excel_path = save_path + 'Prediction.xlsx'
    else:
        excel_path = output_dir + 'Prediction.xlsx'

    # Fix for newer pandas versions
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
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