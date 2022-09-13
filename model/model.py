import torch
from torch import nn


class WideDeep(nn.Module):
    """Multilayer Perceptron."""

    def __init__(self, input_size):
        super().__init__()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
        self.fc1 = nn.Linear(input_size['deep'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 32)
        self.fc_sql = nn.Linear(input_size['sql'], 16)
        self.fc_co = nn.Linear(input_size['co'], 32)
        self.fc_comb = nn.Linear(32 + 16 + 32, 32)
        self.fc_final = nn.Linear(32 + input_size['wide'], 1)

    def forward(self, wide_x, deep_x, sql_x, co_x):
        """Forward pass"""
        out1 = self.relu(self.dropout(self.fc1(deep_x)))
        out2 = self.relu(self.dropout(self.fc2(out1)))
        out_deep = self.relu(self.dropout(self.fc3(out2)))
        out_sql = self.relu(self.dropout(self.fc_sql(sql_x)))
        out_co = self.relu(self.dropout(self.fc_co(co_x)))
        in_comb = torch.cat([out_deep, out_sql, out_co], dim=1)
        in_comb.squeeze(-1)
        out_comb = self.relu(self.dropout(self.fc_comb(in_comb)))
        in_final = torch.cat([out_comb, wide_x], dim=1)
        in_final.squeeze(-1)
        out_final = self.sigmoid(self.fc_final(in_final))
        return out_final


class WideDeepNew(nn.Module):
    """Multilayer Perceptron."""

    def __init__(self, input_size):
        super().__init__()
        self.dropout = nn.Dropout(p=0.3)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fcco = nn.Linear(input_size['co'], 16)
        self.fcsql = nn.Linear(input_size['sql'], 8)
        self.fc1 = nn.Linear(input_size['deep'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 40)
        self.fc_final = nn.Linear(40 + input_size['wide'] + 16 + 8, 1)

    def forward(self, wide_x, deep_x, co_x, sql_x):
        """Forward pass"""
        out1 = self.relu(self.dropout(self.fc1(deep_x)))
        out2 = self.relu(self.dropout(self.fc2(out1)))
        out_deep = self.relu(self.dropout(self.fc3(out2)))
        out_co = self.relu(self.dropout(self.fcco(co_x)))
        out_sql = self.relu(self.dropout(self.fcsql(sql_x)))
        in_final = torch.cat([out_deep, out_co, out_sql, wide_x], dim=1)
        in_final.squeeze(-1)
        out_final = self.sigmoid(self.fc_final(in_final))
        return out_final