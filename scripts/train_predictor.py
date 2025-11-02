import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.utils import *
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import Data, DataLoader
import numpy as np
import copy
import argparse
import os


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        dim = 32
        self.lin0 = torch.nn.Linear(1, dim)

        nn = Sequential(Linear(2, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean')
        self.gru = GRU(dim, dim)

        self.set2set = Set2Set(dim, processing_steps=3)
        self.lin1 = torch.nn.Linear(2 * dim, dim)
        self.lin2 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out.view(-1)


def computeRankLoss(prediction, edge_attr, use_l2 = True):
    grads = torch.autograd.grad(outputs=prediction, inputs=edge_attr,
                                      grad_outputs=torch.ones(prediction.size()).cuda(), create_graph=True)
    grads = grads[0]
    lower_grads = F.relu(grads[:, 0])
    upper_grads = F.relu(-1*grads[:, 1])
    if use_l2:
        rank_loss = lower_grads.square().sum() + upper_grads.square().sum()
    else:
        rank_loss = lower_grads.sum() + upper_grads.sum()
    return rank_loss


def getArg():
    parser = argparse.ArgumentParser(description='Message Passing for MOCU Prediction')
    parser.add_argument('--pretrain', default='.',
                        help='pretrained model name')
    parser.add_argument('--name', default='MP_Graph',
                        help='file name to save the model')
    parser.add_argument('--data_path', default='',
                        help='')
    parser.add_argument('--EPOCH', default=400, type=int,
                        help='EPOCH to train')
    parser.add_argument('--test_only', action='store_true',
                        help='output test result only')
    parser.add_argument('--debug', action='store_true',
                        help='print debug information')
    parser.add_argument('--Constrain_weight', default=0.0001, type=float,
                        help='rank loss weight')
    parser.add_argument('--multiple_model', action='store_true',
                        help='use multiple models for test')
    parser.add_argument('--output_dir', type=str, default='../models/',
                        help='Output directory for models, plots, and statistics')

    args = parser.parse_args()
    return args


def loadData(test_only, data_path, pretrain, name, output_dir):

    print('Preparing data...')
    data_list = torch.load(data_path)

    if test_only:
        model_paths = pretrain.split('+')
        pretrain = model_paths[0]
        statistics = torch.load(output_dir + pretrain + '/statistics.pth')
        mean = statistics['mean']
        std = statistics['std']
        for d in data_list:
            d.y = (d.y - mean) / std
        data_test = data_list[0:len(data_list)]
        train_loader = []
        test_loader = DataLoader(data_test, batch_size=128, shuffle=False)

    else:
        mean = np.asarray([d.y[0][0] for d in data_list]).mean()
        std = np.asarray([d.y[0][0] for d in data_list]).std()
        for d in data_list:
            d.y = (d.y - mean) / std
        data_train = data_list[0:int(0.96 * len(data_list))]
        data_test = data_list[int(0.96 * len(data_list)):len(data_list)]
        train_loader = DataLoader(data_train, batch_size=128, shuffle=True)
        test_loader = DataLoader(data_test, batch_size=128, shuffle=False)
        torch.save({'mean': mean, 'std': std}, output_dir + name + '/statistics.pth')

    return train_loader, test_loader, [std, mean]


def main():

    print(sys.argv)
    args = getArg()
    EPOCH = args.EPOCH if not args.test_only else 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure output directory exists and normalize path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir_str = str(output_dir) + '/' if not str(output_dir).endswith('/') else str(output_dir)
    
    # Model folder: if name is provided and not empty, append it; otherwise use output_dir directly
    # This allows run.sh to pass the exact timestamp folder path
    # Special marker "__USE_OUTPUT_DIR__" or empty string means use output_dir directly
    use_output_dir_directly = (not args.name or args.name.strip() == '' or args.name == '__USE_OUTPUT_DIR__')
    
    if not use_output_dir_directly:  # Name is provided and meaningful
        model_dir = output_dir_str + args.name
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_path_prefix = output_dir_str + args.name
        load_data_output_dir = output_dir_str
        load_data_name = args.name
        print(f"[train_predictor] Using name '{args.name}': saving to {model_path_prefix}/")
    else:
        # No name provided or empty - use output_dir directly (run.sh passes exact timestamp folder)
        model_dir = output_dir_str.rstrip('/')
        model_path_prefix = output_dir_str.rstrip('/')
        # Extract folder name from path for loadData
        model_path_prefix_path = Path(model_path_prefix)
        load_data_output_dir = str(model_path_prefix_path.parent) + '/'
        load_data_name = model_path_prefix_path.name
        print(f"[train_predictor] No name provided - using output_dir as model folder: {model_path_prefix}")
        print(f"[train_predictor] Statistics will be saved to: {load_data_output_dir}{load_data_name}/statistics.pth")
    
    train_loader, test_loader, [std, mean] = loadData(args.test_only, args.data_path, args.pretrain, load_data_name, load_data_output_dir)
    print('Making Model...')
    with torch.backends.cudnn.flags(enabled=False):
        model = Net().cuda()  # Create model FIRST
    
        if args.pretrain != '.':
            if args.multiple_model:
                model_paths = args.pretrain.split('+')
                models = []
                for model_path in model_paths:
                    m = Net().cuda()
                    m.load_state_dict(torch.load(output_dir_str + model_path + '/model.pth'))
                    models.append(m)
            else:
                # Load pretrained weights into the model
                model.load_state_dict(torch.load(output_dir_str + args.pretrain + '/model.pth'))

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                            factor=0.7, patience=5, min_lr=0.00001)

        test_MSE = np.zeros(EPOCH)
        train_MSE = np.zeros(EPOCH)
        train_rank = np.zeros(EPOCH)
        # start training
        for epoch in range(EPOCH):
            if not args.test_only:
                train_MSE_step = []
                train_rank_step = []
                model.train()
                for data in train_loader:  # for each training step
                    # train
                    data_ = copy.deepcopy(data)
                    data_.edge_attr.requires_grad = True
                    lr = scheduler.optimizer.param_groups[0]['lr']
                    data_ = data_.to(device)
                    optimizer.zero_grad()
                    prediction = model(data_).unsqueeze(dim=1)  # [batch_size, 1]
                    mseLoss = F.mse_loss(prediction, data_.y)
                    rankLoss = args.Constrain_weight*computeRankLoss(prediction, data_.edge_attr)
                    loss = mseLoss + rankLoss
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                    train_MSE_step.append(mseLoss.item() * std * std)
                    train_rank_step.append(rankLoss.item() * std * std)

                train_MSE[epoch] = (sum(train_MSE_step) / len(train_MSE_step))
                train_rank[epoch] = (sum(train_rank_step) / len(train_rank_step))
                # Print progress every 10 epochs or at the end
                if (epoch + 1) % 10 == 0 or epoch == EPOCH - 1:
                    print('Epoch %d/%d | LR: %.6f | Train MSE: %.6f | Train Rank: %.6f' % 
                          (epoch + 1, EPOCH, lr, train_MSE[epoch], train_rank[epoch]))

            # test
            model.eval()
            error = 0
            for data in test_loader:
                data = data.to(device)
                if args.multiple_model:
                    predictions = 0
                    for model in models:
                        model.eval()
                        predictions = predictions + model(data).unsqueeze(dim=1)
                    prediction = predictions/len(models)
                else:
                    prediction = model(data).unsqueeze(dim=1)
                error += (prediction * std - data.y * std).square().sum().item()  # MSE
            loss = error / len(test_loader.dataset)
            test_MSE[epoch] = loss
            # Print test result every 10 epochs or at the end
            if (epoch + 1) % 10 == 0 or epoch == EPOCH - 1:
                print('         | Test MSE: %.6f' % loss)
            if epoch > 5 and loss < min(test_MSE[0:epoch]):
                torch.save(model.state_dict(), model_path_prefix + '/model.pth')
                if (epoch + 1) % 10 != 0:
                    print('         | Test MSE: %.6f (best, saved)' % loss)

    # plot and save
    # Use load_data_name (extracted from folder path) instead of args.name (which might be __USE_OUTPUT_DIR__)
    plot_name = load_data_name if use_output_dir_directly else args.name
    plotCurves(train_MSE, train_rank, test_MSE, EPOCH, plot_name, model_path_prefix + '/')

    # save some prediction result
    savePrediction(data, prediction, std, mean, plot_name, model_path_prefix + '/')

    if args.debug:
        printInstance(data, prediction, std, mean)


if __name__ == '__main__':
    main()
