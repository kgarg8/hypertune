from comet_ml import Experiment
from comet_ml import Optimizer
import torch
import torch.nn as nn
import torch.optim as optimizer
import numpy as np
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt

experiment = Experiment(
    api_key="random",
    project_name="krishna-rl",
    workspace="krishna-garg",
)

config = {
    "algorithm": "grid",

    # Declare your hyperparameters in the Vizier-inspired format:
    "parameters": {
        "hidden_size_1": {"type": "discrete", "values": [8]},#, 10, 12]},
        "hidden_size_2": {"type": "discrete", "values": [8, 10, 12]},
        "hidden_size_3": {"type": "discrete", "values": [8]},#, 10, 12]},
        "learning_rate": {"type": "discrete", "values": [0.01]},#, 0.001]},
        "num_epochs":    {"type": "discrete", "values": [25]},
    },

    # Declare what we will be optimizing, and how:
    "spec": {
      "metric": "test_loss",
      "objective": "minimizer",
      "seed": 1
    },
}

comet_opt = Optimizer(config, project_name="krishna-rl", api_key="TZad3Xo6GtBmTrHONXh6bL0zw")
experiment.log_parameters(config["parameters"])

class NN(nn.Module):
    def __init__(self, input_size, h1, h2, h3, num_output):
        super(NN,self).__init__()
        self.fc1 = nn.Linear(input_size,h1)
        self.fc2 = nn.Linear(h1,h2)
        self.fc3 = nn.Linear(h2, h3)
        self.fc4 = nn.Linear(h3,num_output)

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def train_new_model(hyper_params):
    df = pd.read_csv('dataset.csv')
    index = df['n-class-0-latency'].index[df['n-class-0-latency'].apply(np.isnan)]
    df = df.drop(index)
    print(df.head())
    train_x = df[0:800][['ar-0', 'ar-1', 'ar-2', 'ar-3', 'pr-0', 'pr-1', 'pr-2', 'pr-3']].to_numpy()
    test_x = df[800:][['ar-0', 'ar-1', 'ar-2', 'ar-3', 'pr-0', 'pr-1', 'pr-2', 'pr-3']].to_numpy()
    
    print('train_x')
    print(df.shape)
    
    train_y = df[0:800][['n-class-0-latency', 'n-class-1-bw', 'n-class-2-bw']].to_numpy()
    test_y = df[800:][['n-class-0-latency', 'n-class-1-bw', 'n-class-2-bw']].to_numpy()
    
    for epoch in range(hyper_params["num_epochs"]):
        model = NN(8, hyper_params["hidden_size_1"],hyper_params["hidden_size_2"],hyper_params["hidden_size_3"],3)
        prediction = model(torch.from_numpy(train_x).float())
        loss_func = nn.MSELoss()
        opt = optimizer.Adam(model.parameters(), lr=hyper_params["learning_rate"])
        loss = loss_func(prediction.float(), torch.from_numpy(train_y).float())
        opt.zero_grad()  # clear gradients for next train
        loss.backward()  # backpropagation, compute gradients
        opt.step()  # apply gradients
        experiment.log_metric("train_loss",loss.data.numpy(),step=epoch)

    predict_test_y = model(torch.from_numpy(test_x).float())
    test_loss = loss_func(predict_test_y.float(), torch.from_numpy(test_y).float())

    return loss.data.numpy(),test_loss

def run(h1, h2, h3, lr, epochs):
    hyper_params = {
        "hidden_size_1": h1,
        "hidden_size_2": h2,
        "hidden_size_3": h3,
        "learning_rate": lr,
        "num_epochs"   : epochs
    }
    train_loss, test_loss = train_new_model(hyper_params)
    return(train_loss, test_loss)

for experiment in comet_opt.get_experiments():
    experiment.add_tag("smaller_model_three_hyperparameters_opt")
    train_loss, test_loss = run(experiment.get_parameter("hidden_size_1"),
                             experiment.get_parameter("hidden_size_2"),
                             experiment.get_parameter("hidden_size_3"), 
                             experiment.get_parameter("learning_rate"),
                             experiment.get_parameter("num_epochs"))
    experiment.log_metric("train_loss", train_loss)
    experiment.log_metric("test_loss", test_loss)