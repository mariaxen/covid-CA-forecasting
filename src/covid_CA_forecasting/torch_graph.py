import torch 
import numpy as np 
import pandas as pd 
from torch_geometric.data import Data, DataLoader

def to_torch_inputs(df, edge_raw_cols, edge_wt_cols=None, node_features=None, y_feature=None):
    # This function converts a dataframe format to PyTorch graph format with
    # df = dataframe that contains source and destination of edge, and if needed, edge properties
    # edge_raw_cols = a list of column names with [source, destination] format
    # node_feature = a dataframe with dimension [number of nodes, number of features] for node features
    # y_feature = a Series or list or numpy object for ground truth of each node
    # Return:
    # data = a torch graph object

    # handle edges
    edge_raw = df.loc[:, edge_raw_cols].to_numpy()
    # first row is starting and second row is destination
    edge_index = torch.tensor(np.transpose(edge_raw), dtype=torch.long)
    # Each column is the edge weight
    if edge_wt_cols is not None:
        edge_wt = df.loc[:, edge_wt_cols].to_numpy()
        edge_attr = torch.tensor(np.reshape(edge_wt, (edge_wt.shape[0], len(edge_wt_cols))), dtype=torch.float)
    else:
        # If there is no weight
        edge_attr = None

    # handle node features
    # This is the target
    # 1 column vector for pytorch
    if y_feature is not None:
        if ~isinstance(y_feature, pd.Series):
            y_feature = pd.Series(y_feature)
        y = y_feature.to_numpy()
        y = torch.tensor(np.reshape(y, (len(y), 1)), dtype=torch.float)
    else:
        y = None

    # Multiple columns for each node
    if node_features is not None:
        if ~isinstance(node_features, pd.DataFrame):
            node_features = pd.DataFrame(node_features)
        x = node_features.to_numpy()
        x = torch.tensor(x, dtype=torch.float)
    else:
        x = None

    # calculate number of nodes
    if node_features is not None:
        num_nodes = node_features.shape[0]
    elif y_feature is not None:
        num_nodes = y_feature.shape[0]
    else:
        num_nodes = len(pd.unique(edge_raw.flatten()))

    # merge to create torch graoh object
    # x is node features
    data = Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, num_nodes=num_nodes)

    return data
