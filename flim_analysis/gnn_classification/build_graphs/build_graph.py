
import os

import config.const as const
import config.params as params
import numpy as np
import networkx as nx
from scipy.spatial import Delaunay
from joblib import Parallel, delayed
import pickle
import random
import os

import pandas as pd


####################################
#####     Create full graph    #####
####################################

def create_graph_for_leap_id(graph_id, group, feature_type, max_distance, save_dir=False):
    """
    Create a spatial graph for a single LEAP sample or patch using Delaunay triangulation.

    Nodes represent individual nuclei, with features and spatial coordinates as attributes.
    Edges are formed between neighboring nuclei that fall within a maximum distance threshold.

    Parameters
    ----------
    graph_id : str or int
        Unique identifier for the graph (e.g., LEAP ID or patch ID).
    group : pd.DataFrame
        DataFrame containing the features of a single LEAP sample or patch.
        Must include 'nucleus_label', 'X coordinate', 'Y coordinate', and feature columns.
    feature_type : str
        Key to extract the relevant feature list from `params.features_params`.
    max_distance : float
        Maximum Euclidean distance allowed between connected nodes.
    save_dir : str or bool, optional
        Directory to save the graph as a `.pkl` file. If False, the graph is returned in memory.

    Returns
    -------
    tuple
        A tuple containing:
        - graph_id: The identifier of the graph.
        - g_info: Either the NetworkX graph object or the file path to the saved graph.
    """
    print(f"start with {graph_id}")
    # Create a new graph
    G = nx.Graph()

    # Add nodes with attributes
    for idx, (_,row) in enumerate(group.iterrows()):
        attributes = {feature: row[feature] for feature in params.features_params[feature_type]}

        G.add_node(
            idx,
            nucleus_label=row['nucleus_label'],
            pos=(row['Y coordinate'], row['X coordinate']),
            **attributes
        )

    # Debug: Check if all nodes have the required attributes
    for node in G.nodes():
        if 'pos' not in G.nodes[node]:
            print(f"Node {node} is missing the 'pos' attribute in graph {graph_id}.")
        if 'lifetime_mean' not in G.nodes[node]:
            print(f"Node {node} is missing the 'lifetime_mean' attribute in graph {graph_id}.")

    # Prepare coordinates for Delaunay triangulation
    coords = np.array([G.nodes[node]['pos'] for node in G.nodes()])

    if max_distance == "fully_connected":
        # Connect all pairs of nodes
        n_nodes = len(coords)
        edges = [[i, j] for i in range(n_nodes) for j in range(i+1, n_nodes)]
    else:
        tri = Delaunay(coords)

        # Create edges based on neighbors within max_distance
        indptr_neigh, neighbours = tri.vertex_neighbor_vertices

        edges = []
        for i in range(len(coords)):
            i_neigh = neighbours[indptr_neigh[i]:indptr_neigh[i+1]]
            for j in i_neigh:
                if np.linalg.norm(coords[i] - coords[j]) <= max_distance:
                    edges.append([i, j])
    
    # Add edges to the graph
    G.add_edges_from(edges)

    g_info = G
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        graph_file_path = os.path.join(save_dir, f"graph_{graph_id}.pkl")
        with open(graph_file_path, 'wb') as f:
            pickle.dump(G, f)
        g_info = graph_file_path

    print(f"finish with {graph_id}")
    return graph_id, g_info


######################################
##### Local graphs - tissue wise #####
######################################
def create_graphs_parallel(df, group_id, cell_feature_name, max_distance, save_dir=False, n_jobs=-1):
    """
    Create spatial graphs in parallel for multiple LEAP samples or groups.

    This function groups a DataFrame by the specified `group_id` (e.g., 'leap_ID'), and
    for each group, constructs a spatial graph using Delaunay triangulation and a distance threshold.
    Graph creation is parallelized using joblib for faster processing.

    Parameters
    ----------
    df : pd.DataFrame
        Full DataFrame containing per-nucleus features and coordinates across all samples.
    group_id : str
        Column name by which to group the DataFrame (e.g., 'leap_ID').
    cell_feature_name : str
        Key for selecting feature columns from `params.features_params`.
    max_distance : float
        Maximum Euclidean distance allowed for connecting two nodes in the graph.
    save_dir : str or bool, optional
        If provided, each graph is saved as a `.pkl` file in this directory.
        A dictionary mapping of all graph IDs to file paths is also saved.
    n_jobs : int, optional
        Number of parallel jobs to run. Use -1 to utilize all available cores.

    Returns
    -------
    dict
        Dictionary mapping each `graph_id` to either the corresponding NetworkX graph object
        or its saved file path (if `save_dir` is specified).
    """
    def get_save_path(graph_id):
        if save_dir:
            return os.path.join(save_dir, str(graph_id))
        return False

    grouped = df.groupby(group_id)
    
    # Parallel execution for each group
    results = Parallel(n_jobs=n_jobs)(
        delayed(create_graph_for_leap_id)(
            graph_id,
            group,
            cell_feature_name,
            max_distance,
            get_save_path(graph_id),
        ) for graph_id, group in grouped
    )
    
    graphs = {id: g_info for id, g_info in results}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        graphs_mapping_files = os.path.join(save_dir, f"graphs_mapping_files.pkl")
        with open(graphs_mapping_files, 'wb') as f:
            pickle.dump(graphs, f)
            
    return graphs


#####################################
##### Local graphs - patch wise #####
#####################################
            
def create_patches_graphs_parallel(df, group_id, cell_feature_name, max_distance, save_dir=False, local_params=None, n_jobs=-1):
    """
    Create spatial graphs in parallel for patch-level FLIM data.

    This function groups the input DataFrame by multiple keys (typically LEAP ID and patch ID),
    and creates a graph for each patch using Delaunay triangulation and a max distance threshold.
    Graphs are optionally saved to disk in a structured folder hierarchy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing patch-level single-cell features and coordinates.
    group_id : list or tuple of str
        Column names to group by (e.g., ['leap_ID', 'patch_ID']) for patch-level graphs.
    cell_feature_name : str
        Key used to select the feature list from `params.features_params`.
    max_distance : float
        Maximum Euclidean distance for connecting nodes in the graph.
    save_dir : str or bool, optional
        Directory to save individual graph `.pkl` files and the mapping dictionary.
        If False, graphs are returned in memory only.
    local_params : str, optional
        A string identifier (e.g., 'size_1500_overlap_0.75') used for naming and organizing outputs.
    n_jobs : int, optional
        Number of parallel jobs (default is -1 for all available cores).

    Returns
    -------
    dict
        Dictionary mapping each patch graph ID (formatted as 'LEAP_patch') to either:
        - the NetworkX graph object (if not saving), or
        - the file path of the saved `.pkl` file (if `save_dir` is specified).
    """
    def get_save_path(graph_id):
        if save_dir:
            return os.path.join(save_dir, graph_id[0], local_params)
        return False
    
    def get_graph_id(graph_id):
        return f"{graph_id[0]}_{graph_id[1]}"
    
    grouped = df.groupby(group_id)
    
    # Parallel execution for each group
    results = Parallel(n_jobs=n_jobs)(
        delayed(create_graph_for_leap_id)(
            get_graph_id(graph_id),
            group,
            cell_feature_name,
            max_distance,
            get_save_path(graph_id),
        ) for graph_id, group in grouped
    )
    
    graphs = {id: g_info for id, g_info in results}
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        graphs_mapping_files = os.path.join(save_dir, f"graphs_mapping_files_{local_params}.pkl")
        with open(graphs_mapping_files, 'wb') as f:
            pickle.dump(graphs, f)
            
    return graphs


####################################
#######   Global sampling    #######
####################################
def sample_and_expand_subgraph(graph, num_samples, k):
    """
    Create a subgraph by sampling nodes and adding edges up to k hops, 
    then reindex the nodes to ensure contiguous indices.

    Parameters:
    graph (networkx.Graph): The input graph.
    num_samples (int): The number of nodes to sample initially.
    k (int): The number of hops to expand.

    Returns:
    networkx.Graph: The resulting subgraph with reindexed nodes.
    dict: A mapping from old node indices to new indices.
    """

    # nodes = graph.nodes()
    nodes = list(graph.nodes())
        
    sampled_nodes = random.sample(nodes, min(num_samples, len(nodes)))
    
    subgraph = nx.Graph()
    
    current_layer = set(sampled_nodes)
    visited_nodes = set()

    for step in range(k):
        next_layer = set()
        for node in current_layer:
            if node not in visited_nodes:
                subgraph.add_node(node, **graph.nodes[node])
                for neighbor in graph.neighbors(node):
                    if neighbor not in visited_nodes:
                        subgraph.add_node(neighbor, **graph.nodes[neighbor])
                        subgraph.add_edge(node, neighbor)
                        next_layer.add(neighbor)
        
        visited_nodes.update(current_layer)
        current_layer = next_layer - visited_nodes

    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(subgraph.nodes())}
    subgraph = nx.relabel_nodes(subgraph, mapping)
    
    return subgraph


def create_subgraphs_for_graph(leap_id, graph, num_samples, k, num_subsamples, save_dir=False):
    """
    Create multiple subgraphs for a single graph.

    Parameters:
    leap_id (str): The ID of the graph.
    graph (networkx.Graph): The input graph.
    num_samples (int): The number of nodes to sample initially.
    k (int): The number of hops to expand.
    num_subsamples (int): Number of subgraphs to generate.

    Returns:
    dict: A dictionary of subgraphs with keys in the format {leap_id_idx}.
    """
    print(f"start with {leap_id}")

    subgraphs = {}
    for idx in range(num_subsamples):
        graph_id = f"{leap_id}_{idx}"
        subgraph = sample_and_expand_subgraph(graph, num_samples, k)
        check_existence_nodes_attributes(subgraph, graph_id)
        g_info = subgraph

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
            graph_file_path = os.path.join(save_dir, f"graph_{graph_id}.pkl")
            with open(graph_file_path, 'wb') as f:
                pickle.dump(subgraph, f)
            g_info = graph_file_path

        subgraphs[graph_id] = g_info
    
    print(f"finish with {leap_id}")
    return subgraphs

def create_all_subgraphs(graphs, nodes_samples, k, subgraphs_num, save_dir=False):
    """
    Generate subgraphs for all graphs in the dictionary.

    Parameters:
    graphs (dict): Dictionary of graphs with leap_id as keys.
    num_samples (int): The number of nodes to sample initially.
    k (int): The number of hops to expand.
    num_subsamples (int): Number of subgraphs to generate for each graph.

    Returns:
    dict: A dictionary of all subgraphs with keys in the format {leap_id_idx}.
    """

    def get_save_path(graph_id, subgraphs_params):
        if save_dir:
            return os.path.join(save_dir, graph_id, subgraphs_params)
        return False
    
    def get_graph(graph):
        if save_dir:
            with open(graph, 'rb') as file:
                graph = pickle.load(file)
        return graph
        
    subgraphs_params = f"subgraphs_{subgraphs_num}_nodes_samples_{nodes_samples}_k_hops_{k}"

    results = Parallel(n_jobs=-1)(
        delayed(create_subgraphs_for_graph)(leap_id, get_graph(graph),
                                            nodes_samples, k, subgraphs_num,
                                            get_save_path(leap_id, subgraphs_params))
        for leap_id, graph in graphs.items()
    )

    all_subgraphs = {}
    for subgraph_dict in results:
        all_subgraphs.update(subgraph_dict)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)  # Create directory if it doesn't exist
        graphs_mapping_files = os.path.join(save_dir, f"subgraphs_mapping_files_{subgraphs_params}.pkl")
        with open(graphs_mapping_files, 'wb') as f:
            pickle.dump(all_subgraphs, f)

    return all_subgraphs

def check_existence_nodes_attributes(graph, graph_id):
    for n in graph.nodes():
        if 'pos' not in graph.nodes[n]:
            print(f"Node {n} in graph {graph_id} is missing the 'pos' attribute.")
        if 'lifetime_mean' not in graph.nodes[n]:
            print(f"Node {n} in graph {graph_id} is missing the 'lifetime' attribute.")