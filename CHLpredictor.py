import torch
import torch.nn as nn
from copy import deepcopy as dc
import numpy as np

class CHLpredictor(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_stacked_layers, out_features=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        # self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
        #                     batch_first=True)
        self.model = nn.GRU(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, out_features)

        self.device = device
        self.model.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        #out, _ = self.model(x, (h0, c0))
        out, _ = self.model(x, h0)
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

        #print(MSE.item(), loss_LC.item(), loss_HC.item())

        return loss

def prepare_dataframe_for_lstm_with_date(df, n_steps):
    df = dc(df)
    df.dropna(inplace=True)

    df1 = dc(df[['Date', 'Close']])
    df2 = dc(df[['Date', 'Close']])
    df3 = dc(df[['Date', 'Low']])
    df4 = dc(df[['Date', 'High']])
    df5 = dc(df[['Date', 'Volume']])

    df2['CloseChange'] = df1['Close'].pct_change(fill_method='ffill')
    df3['LowChange'] = df3['Low']/df1['Close'].shift(1)
    df4['HighChange'] = df4['High']/df1['Close'].shift(1)
    df1 = df1.shift(1)
    df5 = df5.shift(1)

    df2.drop(['Close'], axis='columns', inplace=True)
    df3.drop(['Low'], axis='columns', inplace=True)
    df4.drop(['High'], axis='columns', inplace=True)

    df1.set_index('Date', inplace=True)
    df2.set_index('Date', inplace=True)
    df3.set_index('Date', inplace=True)
    df4.set_index('Date', inplace=True)
    df5.set_index('Date', inplace=True)

    for i in range(1, n_steps + 1):
        df1[f'Close(t-{i})'] = df1['Close'].shift(i)
        df2[f'CloseChange(t-{i})'] = df2['CloseChange'].shift(i)
        df3[f'LowChange(t-{i})'] = df3['LowChange'].shift(i)
        df4[f'HighChange(t-{i})'] = df4['HighChange'].shift(i)
        df5[f'Volume(t-{i})'] = df5['Volume'].shift(i)

    df1.dropna(inplace=True)
    df2.dropna(inplace=True)
    df3.dropna(inplace=True)
    df4.dropna(inplace=True)
    df5.dropna(inplace=True)

    return df1, df2, df3, df4, df5

def prepare_dataframe_for_lstm(df, n_steps):
    df = dc(df)

    df1 = dc(df[['Close']])
    df2 = dc(df[['Close']])
    df3 = dc(df[['Low']])
    df4 = dc(df[['High']])
    df5 = dc(df[['Volume']])

    df2['CloseChange'] = df1['Close'].pct_change(fill_method='ffill')
    df3['LowChange'] = df3['Low'] / df1['Close'].shift(1)
    df4['HighChange'] = df3['High'] / df1['Close'].shift(1)
    df1 = df1.shift(1)
    df5 = df5.shift(1)

    df2.drop(['Close'], axis='columns', inplace=True)
    df3.drop(['Low'], axis='columns', inplace=True)
    df4.drop(['High'], axis='columns', inplace=True)

    for i in range(1, n_steps + 1):
        df1[f'Close(t-{i})'] = df1['Close'].shift(i)
        df2[f'CloseChange(t-{i})'] = df2['CloseChange'].shift(i)
        df3[f'LowChange(t-{i})'] = df3['LowChange'].shift(i)
        df4[f'HighChange(t-{i})'] = df4['HighChange'].shift(i)
        df5[f'Volume(t-{i})'] = df5['Volume'].shift(i)

    return df1, df2, df3, df4, df5
