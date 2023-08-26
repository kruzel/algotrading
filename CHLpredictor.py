import torch
import torch.nn as nn
from copy import deepcopy as dc
import numpy as np

class CHLpredictor(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_stacked_layers, out_features=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, out_features)

        self.device = device
        self.lstm.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class CHLLoss(nn.Module):
    def __init__(self):
        super(CHLLoss, self).__init__()

    def forward(self, inputs, targets):
        MSE = torch.pow(torch.mean(targets - inputs), 2)

        loss_LC = inputs[0] - inputs[1] # close - low
        loss_LC[loss_LC>0] = 0
        loss_LC = torch.pow(loss_LC.mean(), 2)

        loss_HC = inputs[2] - inputs[0]  # high - close
        loss_HC[loss_HC > 0] = 0
        loss_HC = torch.pow(loss_HC.mean(), 2)

        loss = MSE + loss_HC + loss_LC

        return loss

def prepare_dataframe_for_lstm_with_date(df, n_steps):
    df = dc(df)
    df.dropna(inplace=True)

    df1 = dc(df[['Date', 'Close']])
    df2 = dc(df[['Date', 'Low']])
    df3 = dc(df[['Date', 'High']])
    df4 = dc(df[['Date', 'Volume']])

    df1.set_index('Date', inplace=True)
    df2.set_index('Date', inplace=True)
    df3.set_index('Date', inplace=True)
    df4.set_index('Date', inplace=True)

    for i in range(1, n_steps + 1):
        df1[f'Close(t-{i})'] = df1['Close'].shift(i)
        df2[f'Low(t-{i})'] = df2['Low'].shift(i)
        df3[f'High(t-{i})'] = df3['High'].shift(i)
        df4[f'Volume(t-{i})'] = df4['Volume'].shift(i)

    df1.dropna(inplace=True)
    df2.dropna(inplace=True)
    df3.dropna(inplace=True)
    df4.dropna(inplace=True)

    return df1, df2, df3, df4

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df1 = dc(df[['Close']])
    df2 = dc(df[['Low']])
    df3 = dc(df[['High']])
    df4 = dc(df[['Volume']])

    for i in range(1, n_steps + 1):
        df1[f'Close(t-{i})'] = df1['Close'].shift(i)
        df2[f'Low(t-{i})'] = df2['Low'].shift(i)
        df3[f'High(t-{i})'] = df3['High'].shift(i)
        df4[f'Volume(t-{i})'] = df4['Volume'].shift(i)

    return df1, df2, df3, df4
