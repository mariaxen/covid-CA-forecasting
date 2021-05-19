from torch_geometric_temporal.nn.recurrent import DCRNN, GConvGRU, TGCN, EvolveGCNO, DyGrEncoder, GCLSTM, GConvLSTM, LRGCN
import torch.nn.functional as F
import torch

class GConvGRU_Model(torch.nn.Module):

    def __init__(self, node_features):
        super(GConvGRU_Model, self).__init__()
        self.recurrent_1 = GConvGRU(node_features, 50, 2)
        self.recurrent_2 = GConvGRU(50, 20, 2)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return x

class GConvLSTM_Model(torch.nn.Module):
    def __init__(self, node_features):
        super(GConvLSTM_Model, self).__init__()
        self.recurrent_1 = GConvLSTM(node_features, 50, 2)
        self.recurrent_2 = GConvLSTM(50, 20, 2)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, x, edge_index, edge_weight):
        x, _ = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x, _ = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return x

class DCRNN_Model(torch.nn.Module):

    def __init__(self, node_features):
        super(DCRNN_Model, self).__init__()
        self.recurrent_1 = DCRNN(node_features, 50, 2)
        self.recurrent_2 = DCRNN(50, 20, 2)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return x

class TGCN_Model(torch.nn.Module):

    def __init__(self, node_features):
        super(TGCN_Model, self).__init__()
        self.recurrent_1 = TGCN(node_features, 50, 2)
        self.recurrent_2 = TGCN(50, 20, 2)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return x

class GCLSTM_Model(torch.nn.Module):
    def __init__(self, node_features):
        super(GCLSTM_Model, self).__init__()
        self.recurrent_1 = GCLSTM(node_features, 50, 2)
        self.recurrent_2 = GCLSTM(50, 20, 2)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, x, edge_index, edge_weight):
        x, _ = self.recurrent_1(x, edge_index, edge_weight)
        x, _ = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return x

class LRGCN_Model(torch.nn.Module):
    def __init__(self, node_features, num_relations):
        super(LRGCN_Model, self).__init__()
        self.recurrent_1 = LRGCN(node_features, 50, num_relations, 3)
        self.recurrent_2 = LRGCN(50, 20,  num_relations, 3)
        self.linear = torch.nn.Linear(20, 1)

    def forward(self, x, edge_index, edge_weight):
        h, _,  = self.recurrent_1(x, edge_index, edge_weight)
        h, _,  = self.recurrent_2(h, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.linear(h)
        return h

class DyGrEncoder_Model(torch.nn.Module):
    def __init__(self, node_features):
        super(DyGrEncoder_Model, self).__init__()
        self.recurrent_1 = DyGrEncoder(node_features, 32, "mean", 32, 1)
        self.recurrent_2 = DyGrEncoder(32, 32, "mean", 32, 1)
        self.linear = torch.nn.Linear(32, 1)

    def forward(self, x, edge_index, edge_weight):
        h, _, _ = self.recurrent_1(x, edge_index, edge_weight)
        h, _, _ = self.recurrent_2(h, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, training=self.training)
        h = self.linear(h)
        return h

def train_error(model, loss_function, data):
    model.eval()
    cost = 0

    for time, snapshot in enumerate(data):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost += loss_function(torch.reshape(y_hat, (-1,)), snapshot.y)

    cost = cost / (time + 1)
    cost = cost.item()
    return cost

def test_error(model, loss_function, data):
    model.eval()
    cost = 0

    for time, snapshot in enumerate(data):
        y_hat = model(snapshot.x, snapshot.edge_index, snapshot.edge_attr)
        cost = loss_function(torch.reshape(y_hat, (-1,)), snapshot.y)

    cost = cost.item()
    return cost
