
import sys
import os

from utils.data_func import *
import config.const as const
import config.params as params

import pandas as pd
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
import pickle
import sys
import warnings
import numpy as np
warnings.filterwarnings("ignore")

class FLIMGraphDataset(Dataset):
    """
    A PyTorch Dataset for loading patch-level graphs from disk and converting them
    into PyTorch Geometric `Data` objects.

    Each graph is stored as a pickled NetworkX object and includes:
    - Nodes: individual nuclei with extracted features
    - Edges: spatial connections (e.g., Delaunay neighbors within max distance)
    - Attributes: position, FLIM features, category label

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame with columns `graph_ID` and `category`, where each row maps to one graph.
    graphs_dict : dict
        A dictionary mapping `graph_ID` to the path of its corresponding pickled graph file.
    """

    def __init__(self, df, graphs_dict):
        super().__init__()
        self.df = df
        self.graphs_dict = graphs_dict


    def __len__(self):
        """
        Return the number of samples (graphs) in the dataset.
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Load a graph by index, extract node features and structure, and convert to PyTorch Geometric Data object.

        Returns
        -------
        Data
            A `torch_geometric.data.Data` object containing:
            - x: node features (Tensor of shape [num_nodes, num_features])
            - edge_index: edge list in COO format (Tensor of shape [2, num_edges])
            - pos: node coordinates (Tensor of shape [num_nodes, 2])
            - y: graph-level label
            - name: graph identifier
        """
        graph_id = self.df.iloc[idx]['graph_ID']
        group =  self.df.iloc[idx]['category']

        graph_path = self.graphs_dict[graph_id]
        with open(graph_path, 'rb') as file:
            G = pickle.load(file)

    # Extract node features dynamically, excluding 'pos'
        cells_features = []
        coords = []
        for n in G.nodes():
            node_features = []
            for k, v in G.nodes[n].items():
                if k not in ('pos', 'nucleus_label'): 
                    node_features.append(v)
                elif k == 'pos':
                    coords.append(v)
            cells_features.append(node_features)

        x = cells_features
        # coords = np.array([G.nodes[n]['pos'] for n in G.nodes()])
        coords = np.array(coords)
        edges = list(G.edges)

 
        ''' edge_index: is tensor shape [2, num of edges] - the first row contains the source nodes of the edges
                        and the second row contains the target nodes of the edges.'''
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        # x = [[cf] for cf in cells_features]
        x = torch.tensor(x, dtype=torch.float)

        y = torch.tensor(group, dtype=torch.long)
        pos = torch.tensor(coords, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, pos=pos, y=y, name=graph_id)
    

def create_subgraphs_df(graphs_data):
    """
    Generate a DataFrame linking each graph to its LEAP ID and binary responder category.

    Parameters
    ----------
    graphs_data : dict
        Dictionary of graph IDs and graph paths or objects.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ['graph_ID', 'leap_ID', 'category']
    """
    df_category_with_rcb, _, _ = extract_core_resection_from_tnbc(const.rcb_file)
    df_category_with_rcb.head()
    # Determine responder status (1 for responder, 0 for non-responder)
    df_category_with_rcb['responder'] = df_category_with_rcb['category'].apply(lambda x: 1 if x == "responder" else 0)

    # Create the new DataFrame
    subgraph_data = []

    for key in graphs_data.keys():
        # Extract leap_ID from the subgraph key
        # leap_id = key.split('_')[0]
        leap_id = key[:3]
        
        # Find the responder status for this leap_ID
        responder_status = df_category_with_rcb.loc[df_category_with_rcb['leap_ID'] == leap_id, 'responder'].values[0] if leap_id in df_category_with_rcb['leap_ID'].values else None
        
        # Append to the data list
        subgraph_data.append({
            "graph_ID": key,
            "leap_ID": leap_id,
            "category": responder_status
        })

    subgraph_df = pd.DataFrame(subgraph_data)
    return subgraph_df

def expand_subgraphs_df_with_pytorchgeo_idx(subgraph_df, pytorchgeo_data):
    """
    Add GNN-compatible index to graph DataFrame based on the PyTorch Geometric dataset.

    Parameters
    ----------
    subgraph_df : pd.DataFrame
        DataFrame with graph_IDs.
    pytorchgeo_data : dict
        Dictionary mapping dataset index to PyTorch Geometric Data objects.

    Returns
    -------
    pd.DataFrame
        Updated subgraph_df DataFrame with 'gnn_data_idx' column.
    """

    # Map graph_ID to data_dict index
    graph_id_to_idx = {data_obj.name: idx for idx, data_obj in pytorchgeo_data.items()}

    # Add gnn_data_idx column to DataFrame
    subgraph_df["gnn_data_idx"] = subgraph_df["graph_ID"].map(graph_id_to_idx)
    return subgraph_df

def convert_dict_files_to_graphs_object(dict_graph_files):
    """
    Load all pickled NetworkX graphs from a dictionary of file paths.

    Parameters
    ----------
    dict_graph_files : dict
        Dictionary mapping graph IDs to file paths.

    Returns
    -------
    dict
        Dictionary mapping graph IDs to loaded NetworkX graph objects.
    """
    graphs_data = {}
    for key, graph_path in dict_graph_files.items():
        with open(graph_path, 'rb') as file:
            graph = pickle.load(file)
        graphs_data[key] = graph

    return graphs_data


def preprocess_dataset(dataset):
    """
    Iterate through a dataset and index the results into a dictionary.

    Parameters
    ----------
    dataset : Dataset
        A PyTorch Dataset subclass (e.g., FLIMGraphDataset).

    Returns
    -------
    dict
        Dictionary where keys are indices and values are PyG Data objects.
    """
    preprocessed_data = {}
    for i, data in (enumerate(tqdm(dataset, total = len(dataset)))):
        preprocessed_data[i] = data
    return preprocessed_data


def create_gnn_data(save_dir, graphs_dict):
    """
    Create PyTorch Geometric data objects from a dictionary of graphs and save both
    the dataset and associated metadata.

    Parameters
    ----------
    save_dir : str
        Directory to store the processed data and metadata.
    graphs_dict : dict
        Dictionary of graph_ID → file_path for pickled NetworkX graphs.
    
    Saves
    -----
    - `data_pytorch_geo.pkl` : dictionary of indexed PyTorch Geometric Data objects
    - `graphs_df_pytorch_geo.pkl` : metadata DataFrame with the following columns:
        - `graph_ID`: unique identifier for each graph (e.g., "LEAP123_5")
        - `leap_ID`: sample identifier from which the graph originates
        - `category`: label or class (e.g., responder vs. non-responder)
        - `gnn_data_idx`: index of the graph in the PyTorch Geometric dataset
    """
    print("create_mapping_df")
    subgraph_df = create_subgraphs_df(graphs_dict)
    dataset = FLIMGraphDataset(subgraph_df, graphs_dict)
    print("process the data to pytorch geo data")
    preprocessed_data = preprocess_dataset(dataset)
    print("add pytorch geo data index")
    subgraph_df_pytorch_geo = expand_subgraphs_df_with_pytorchgeo_idx(subgraph_df, preprocessed_data)

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save preprocessed_data as a pickle file
    preprocessed_data_path = os.path.join(save_dir, "data_pytorch_geo.pkl")
    with open(preprocessed_data_path, "wb") as f:
        pickle.dump(preprocessed_data, f)
    print(f"Saved pytorch_geo_data to {preprocessed_data_path}")

    # Save subgraph_df_pytorch_geo as a pickle file
    subgraph_df_pytorch_geo_path = os.path.join(save_dir, "graphs_df_pytorch_geo.pkl")
    with open(subgraph_df_pytorch_geo_path, "wb") as f:
        pickle.dump(subgraph_df_pytorch_geo, f)
    print(f"Saved subgraph_df_pytorch_geo to {subgraph_df_pytorch_geo_path}")


def create_gnn_data_subgraphs(feature_type, max_dist, nodes_samples, k, subgraphs_num):
    """
    Load precomputed sample-subgraphs and convert them to PyTorch Geometric format for GNN training.

    Parameters
    ----------
    feature_type : str
        Type of cell features to use.
    max_dist : float
        Maximum distance used for building edges in the graphs.
    nodes_samples : int
        Number of nodes sampled per subgraph.
    k : int
        Number of hops in subgraph sampling.
    subgraphs_num : int
        Total number of subgraphs.
    """
    graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'full_tissue', f"max_distance_{max_dist}", 'graphs_data')
    subgraphs_params = f"subgraphs_{subgraphs_num}_nodes_samples_{nodes_samples}_k_hops_{k}"
    graphs_mapping_files = os.path.join(graph_file_dir, f"subgraphs_mapping_files_{subgraphs_params}.pkl")
    with open(graphs_mapping_files, 'rb') as file:
        graphs = pickle.load(file)
        
    save_dir = os.path.join(const.gnn_dir, feature_type, 'full_tissue', f"max_distance_{max_dist}", 'pytorch_geo', subgraphs_params, 'data')
    
    create_gnn_data(save_dir, graphs)


def create_gnn_data_patches(feature_type, max_dist, patch_size, overlap):
    """
    Load patch-based graphs and convert them into PyTorch Geometric format.

    Parameters
    ----------
    feature_type : str
        Type of node features to use.
    max_dist : float
        Max distance threshold used during graph construction.
    patch_size : int
        Size of the patches used to create graphs.
    overlap : float
        Patch overlap ratio.
    """
    patch_params = f'size_{patch_size}_overlap_{overlap}'
    graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'graphs_data')

    graphs_mapping_files = os.path.join(graph_file_dir, f"graphs_mapping_files_{patch_params}.pkl")

    with open(graphs_mapping_files, 'rb') as file:
        graphs = pickle.load(file)

    save_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'pytorch_geo', patch_params, 'data')

    create_gnn_data(save_dir, graphs)

    

def create_gnn_data_patches_shuffling(feature_type, max_dist, patch_size, overlap, seed):
    """
    Load patch-based graphs and convert them into PyTorch Geometric format.

    Parameters
    ----------
    feature_type : str
        Type of node features to use.
    max_dist : float
        Max distance threshold used during graph construction.
    patch_size : int
        Size of the patches used to create graphs.
    overlap : float
        Patch overlap ratio.
    seed : int
        Random value for shuffling.
    """
    patch_params = f'size_{patch_size}_overlap_{overlap}'
    graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'graphs_data', 'shuffling_lifetime', f'{seed}')

    graphs_mapping_files = os.path.join(graph_file_dir, f"graphs_mapping_files_{patch_params}.pkl")

    with open(graphs_mapping_files, 'rb') as file:
        graphs = pickle.load(file)

    save_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'pytorch_geo', 'shuffling_lifetime', patch_params, 'data', f'{seed}')

    create_gnn_data(save_dir, graphs)


def create_gnn_data_patches_structure(feature_type, max_dist, patch_size, overlap):
    """
    Load patch-based graphs and convert them into PyTorch Geometric format.

    Parameters
    ----------
    feature_type : str
        Type of node features to use.
    max_dist : float
        Max distance threshold used during graph construction.
    patch_size : int
        Size of the patches used to create graphs.
    overlap : float
        Patch overlap ratio.
    """
    patch_params = f'size_{patch_size}_overlap_{overlap}'
    graph_file_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'graphs_data', 'structure_only')

    graphs_mapping_files = os.path.join(graph_file_dir, f"graphs_mapping_files_{patch_params}.pkl")

    with open(graphs_mapping_files, 'rb') as file:
        graphs = pickle.load(file)

    save_dir = os.path.join(const.gnn_dir, feature_type, 'patch_tissue', f"max_distance_{max_dist}", 'pytorch_geo', 'structure_only', patch_params, 'data')

    create_gnn_data(save_dir, graphs)