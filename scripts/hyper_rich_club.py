import argparse
from pyflagsercount import flagser_count
import numpy as np
import pickle as pk
from collections import defaultdict
from pathlib import Path
from functools import partial
#import connalysis
import pandas as pd
import matplotlib.pyplot as plt
import scipy.sparse as ss
from scipy.special import binom
import networkx as nx
import math
import warnings
import tqdm
from more_itertools import unique_everseen
from collections import Counter
import time
from pypdf import PdfWriter

##########################################################################################
# Helper Functions

def concatenate_hyperedges_dicts (dict1, dict2):
    """
    Concatenate two dictionaries of hyperedges along dimension/order.

    Parameters
    ----------
    dict1 : dict
        First dictionary mapping dimension/order to its corresponding numpy arrays of hyperedges.
    dict2 : dict
        Second dictionary mapping dimension/order to its corresponding numpy arrays of hyperedges.

    Returns
    -------
    dict
        Combined dictionary representing the union of all hyperedges split by dimension/order.
    """
    agg = {
    key: np.vstack([
        dict1.get(key, np.empty((0, key))),
        dict2.get(key, np.empty((0, key)))
        ])
        for key in set(dict1) | set(dict2)
        }
    return agg

def concatenate_filtration_dicts (dict1, dict2):
    """
    Concatenate two dictionaries of filtrations along dimension/order.

    Parameters
    ----------
    dict1 : dict
        First dictionary mapping dimension/order to its corresponding list of filtrations.
    dict2 : dict
        Second dictionary mapping dimension/order to its corresponding list of filtrations.

    Returns
    -------
    dict
        Combined dictionary representing the union of all filtrations split by dimension/order.
    """
    agg = {
    key: dict1.get(key, []) + dict2.get(key, [])
         for key in set(dict1) | set(dict2)
        }
    return agg


def remove_minimal_vertices(hyperedges, hyperedges_weights, min_weights): 
    # Remove nodes from hyperedges with the min_weight and split by their order

    """
    Remove from each hyperedge, the vertices with minimum weight. Returns the 
    hyperedges with removed vertices (grouped by dimension/order) together with the 
    filtration value of the removed vertices. 

    Parameters
    ----------
    hyperedges : numpy.ndarray
        2D array of hyperedges with shape (n_edges, order).
    hyperedges_weights : pandas.DataFrame
        DataFrame containing weight values corresponding to vertices in hyperedges.
    min_weights : array-like
        Array of minimum weights for each hyperedge.

    Returns
    -------
    agg : dict
        Dictionary mapping hyperedge orders (int) to numpy arrays containing
        the hyperedges of that order after removing the vertices of minimal weight.
    filt : dict
        Dictionary mapping hyperedge orders (int) to lists containing
        the corresponding minimum weights of the vertices removed.

    Raises
    ------
    AssertionError
        If hyperedges has fewer than 2 columns.
    """

    l = hyperedges.shape[1]
    assert hyperedges.shape[1] >= 2, "Hyperedges must between at least 2 nodes"
    # removes nodes of min weight across each row 
    # splits the list by number of nodes 
    # better to do unique here already?
    # Get mask of nodes to keep 
    mask = (hyperedges_weights != min_weights[:, None]).to_numpy()
    hyperedges_min_removed = [(tuple(h[mask_row]),min_weight) for h, mask_row, min_weight in zip(hyperedges, mask, min_weights)]
    # Split by order
    agg = defaultdict(list)
    filt = defaultdict(list)
    for row in hyperedges_min_removed:
        agg[len(row[0])].append(row[0])
        filt[len(row[0])].append(row[1])
    for k in agg:
        agg[k] = np.vstack(agg[k])
    #return dict(agg)
    return dict(agg), dict(filt)


######################################################################################################
# Shuffle Functions 
#TODO put comments in proper docs format



def aschenputtel(jumbled_facets):
    #INPUT: list shape (num_faces, k) contain the jumbled k facets.
    #OUTPUT: two lists: one of valid k-facets, the other with duplicates and degenerates
    jumbled_facets = Counter([tuple(row) for row in jumbled_facets])
    valid, dnd = [], [] 

    for x in jumbled_facets:
        if len(set(x)) == len(x):
            valid.append(x)
            dnd += [x]*(jumbled_facets[x]-1)
        else:
            dnd += [x]*(jumbled_facets[x])

    return valid, dnd


def shuffle_k_facets(facets, rng, rerolls=10, progress=False):
    #INPUT: list of length num_faces with numpy arrays of length k, each representing the k-facets to be shuffled
    #OUTPUT: list of k-facets of approximately equal length

    if len(facets) == 0:
        return facets
    if progress:
        print(f"shuffling {len(facets[0])}-facets")

    shuffled_facets = []
    reroll = facets.copy()

    for i in range(rerolls):
        # shuffle each column independently:
        # np.random.shuffle does not support axis right now, thus following workaround
        reroll = np.transpose(reroll)
        for i in range(len(reroll)):
            rng.shuffle(reroll[i])
        reroll = np.transpose(reroll)

        # sort:
        shuffled_facets, reroll = aschenputtel(shuffled_facets+list(reroll))

        if progress:
            print(f"after round {i} there are {len(reroll)} duplicates and degenerates remaining")
        if len(reroll) <= 1:
            break

    return shuffled_facets


def shuffle_all_facets_seperately(facets, rerolls=10, seed=None):
    rng=np.random.default_rng(seed=seed)
    return {i:shuffle_k_facets(facets[i], rng, rerolls=rerolls) for i in facets.keys()}





# Calculates vertex participation for given simplex list, per dimension and per position
# Input:
#       simp_comp a list of simplices in the format returned by flagser_count
#       n_nodes number of vertices
# Output:
#       a list vfp of length equal to dimension of the complex
#       vfp[dim] has length dim+1, where vfp[dim][i] is an np.array of length
#       number of vertices, and vfp[dim][i][j] is the number of dim-simplices that vertex
#       j is occurs in position i.
def vfp_calc(simp_comp, n_nodes):
    # convert entry to an array for each vertex
    simp_comp = {dim:np.asarray(list(simp_comp[dim])) for dim in simp_comp.keys()}
    vfp = []
    for dim in simp_comp.keys():
        if len(simp_comp[dim]) == 0:
            vfp.append([np.zeros(shape=(n_nodes)) for pos in range(dim)])
        else:
            vfp.append([np.histogram(simp_comp[dim][:,pos], bins=n_nodes, range=(0,n_nodes))[0] for pos in range(dim)])
    return vfp


# Shuffles the maximal simplices
# Input:
#     g is the adjacency matrix in coo format
#     method is the shuffle type, values can be "reroll", "normal", "inflate"
#     reps is the number of repetitions
#     rerollreps is the number of times to reroll degenerates, only used with reroll
#     blow_up_factor is used in the inflate, method
#     seed is the random seed
# Output:
#     first output is a list of length reps where each entry is the maximal simplices of the shuffled complex
#     second output is the vertex participation of the shuffled complex
def shuffle(simplices, number_nodes, method="reroll", reps=10, seed=None, rerollreps=10, blow_up_factor=25):
    new_complex = []
    vfp0 = []

    for i in range(reps):
        if seed is not None:
            newseed = seed+i*234
        else: 
            newseed = None
        new_complex.append(shuffle_all_facets_seperately(simplices, seed = newseed, rerolls=rerollreps))
        vfp0.append(vfp_calc(new_complex[-1], number_nodes)) #computes vertex participation of the shuffle

    return new_complex, vfp0



######################################################################################################
# Main Rich Club Functions

def get_bins_and_node_counts(node_metric, sparse_bin_set = False, bins = None):
    """
    Generate bins and count nodes across filtration levels.

    Parameters
    ----------
    node_metric : pandas.Series or array-like
        Contains the weights of each node.
    sparse_bin_set : bool, default False
        If True, creates bins only for unique values in node_metric plus 
        boundary values. If False, creates consecutive integer bins from 
        0 to max(node_metric) + 1.
    bins : np.array or None
        The bins to use if precomputed, if not None, then spare_bin_set is ignored

    Returns
    -------
    counts : pandas.DataFrame
        DataFrame with cumulative node counts indexed by filtration levels.
    degree_bins : numpy.ndarray
        Array of bin edges used for histogram calculation.
    degree_bins_rv : numpy.ndarray
        Reversed degree bins (excluding the last element), used as index
        for the counts DataFrame.

    Notes
    -----
    The function computes cumulative sums in reverse order to ensure
    F_0 = V (total vertices) and F_{n+1} = empty set for filtration analysis.
    """
    if bins is None:
        if not sparse_bin_set:
            degree_bins = np.arange(node_metric.max() + 2)
        else:
            degree_bins = np.unique(np.append(node_metric, [node_metric.max() + 1]))
            #print(degree_bins)
    else:
        degree_bins = np.append(bins,[bins[-1] + 1])

    degree_bins_rv = degree_bins[-2::-1]
    counts = pd.DataFrame(index= pd.Index(degree_bins_rv, name="filtration"))

    # Count nodes across filtration
    nrn_degree_distribution = np.histogram(node_metric.values, bins=degree_bins)[0]
    nrn_cum_degrees = np.cumsum(nrn_degree_distribution[-1::-1]) # Invert before taking the cumulative, to have F_0 = V and F_{n+1} = empty_set
    counts["nodes"]=nrn_cum_degrees

    return counts, degree_bins, degree_bins_rv

def hyperedge_filtration(hyperedges, node_metric):
    """
    Given a list of hyperedges and a node_metric it returns an arrays 
    of the same shape with the corresponding weights as well as the minimal weight
    of each hyperedge. 

    Parameters
    ----------
    hyperedges : numpy.ndarray
        2D array of shape (n_edges, edge_size) containing node indices
        for each hyperedge. Must have at least 2 columns.
    node_metric : pandas.Series
        Series mapping node indices to their metric values. Index should
        correspond to node indices in hyperedges.

    Returns
    -------
    hyperedges_weights : pandas.DataFrame
        DataFrame where node indices are replaced with their corresponding
        metric values from node_metric.
    min_weights : numpy.ndarray
        1D array containing the minimum weight of each hyperedge.

    Raises
    ------
    AssertionError
        If hyperedges has fewer than 2 columns or if any hyperedge contains
        node indices outside the valid range [0, n-1], where n is the length 
        of node_metric.
    """


    
    # Basic input checks 
    assert hyperedges.shape[1] >= 2, "Hyperedges must between at least 2 nodes"
    n = len(node_metric) # Number of nodes
    assert np.all((hyperedges >= 0) & (hyperedges < n)), "All entries in hyperedges, the list of hyperedges, represent nodes and thus must be between 0 and n-1 (inclusive)"

    # Create dataframe of hyperedges, and replace vertex index with node metric used for the filtration
    metric_dict = node_metric.to_dict() # TODO CHECK THAT node_metric has the right format! 
    hyperedges_weights = pd.DataFrame(hyperedges,dtype=int).copy() 
    for i in hyperedges_weights.keys():
        hyperedges_weights.update(hyperedges_weights[i].map(metric_dict))
    # Sort rows in incresing order, so the first column is the lowest weight of the hyperedge
    sorted_weights = pd.DataFrame(hyperedges_weights.apply(np.sort,axis=1).tolist())

    return hyperedges_weights, sorted_weights.iloc[:, 0].to_numpy() 


def count_edges_on_filtration(hyperedges, node_metric, sparse_bin_set=False, bins=None):
    """
    Count hyperedges across filtration levels. 
    It count remaining hypereges by progressively removing vertices with minimum filtration values 
    When part of the nodes of a hyperedge are removed 
    it counts the lower order hyperedges on the remaining vertices.

    Parameters
    ----------
    hyperedges : numpy.ndarray
        2D array of shape (n_edges, l) containing hyperedges where l is the
        order of hyperedges i.e., hypereges on l nodes. Must be non-empty with consistent shape.
    node_metric : pandas.Series or array-like
        Node metric values used for filtration. Length must equal the number
        of nodes referenced in hyperedges.
    sparse_bin_set : bool, default False
        If True, uses sparse binning based on unique metric values.
        If False, uses dense consecutive integer binning.
    bins : numpy.array or None
        precomputed bins to use

    Returns
    -------
    pandas.DataFrame
        DataFrame with filtration levels as index and hyperedge size as columns, 
        entries are number of hyperedges at that filtration level or that size,
        and additional column 'nodes' with number of nodes at that filtration level.

    Raises
    ------
    AssertionError
        If hyperedges is not a non-empty 2D numpy array with the expected shape,
        or if node indices in hyperedges are outside valid range [0, n-1].
    """

    assert isinstance(hyperedges, np.ndarray) and hyperedges.ndim == 2 and hyperedges.shape[0] > 0, "hyperedges must be a non-empty 2D numpy array with shape (n_edges, l)"
    l = hyperedges.shape[1] # order of hyperedges
    n = len(node_metric) # Number of nodes

    assert np.all((hyperedges >= 0) & (hyperedges < n)), "All entries in hyperedges, the list of hyperedges, represent nodes and thus must be between 0 and n-1 (inclusive)"
    edge_count = [] # filtered edge counts across dimensions

    # Get bins and node_counts 
    nodes, bins, degree_bins_rv = get_bins_and_node_counts(node_metric, sparse_bin_set=sparse_bin_set, bins=bins)
    edge_count.append(nodes)

    # Count hyperedges on top dimension
    hyperedges_weights, min_weights = hyperedge_filtration(hyperedges, node_metric)
    filtered_count = np.cumsum(np.histogram(min_weights, bins=bins)[0][-1::-1])
    edge_count.append(pd.DataFrame(filtered_count ,columns = [l], 
                                    index= pd.Index(degree_bins_rv, name="filtration")))
    carry_over, carry_over_filtrations = remove_minimal_vertices(hyperedges, hyperedges_weights, min_weights) # Remaining hyperedges after removing vertices with lowest filtration value
    # Loop process until the carry over is empty or contains only hyperedges on 2 nodes 
    while carry_over:
        # Get edge counts at the next dimension
        alpha = max(list(carry_over.keys())) # Max dimension on the carry over 
        if alpha <= 1:
            break
        #hyperedges_lower = np.unique(carry_over[alpha], axis = 0)
        hyperedges_lower = pd.DataFrame({'edge':map(tuple, carry_over[alpha]), 'filtration':carry_over_filtrations[alpha]})
        hyperedges_lower = hyperedges_lower.groupby(['edge']).max()
        carry_over.pop(alpha) # Remove them from carry_over
        carry_over_filtrations.pop(alpha)

        hyperedges_weights, min_weights = hyperedge_filtration(np.vstack(hyperedges_lower.index), node_metric)
        filtered_count = np.cumsum(np.histogram(min_weights, bins=bins)[0][-1::-1])
        filtered_count_mins = np.cumsum(np.histogram(hyperedges_lower['filtration'].to_numpy(), bins=bins)[0][-1::-1])
        count_across_filtration = filtered_count-filtered_count_mins
        edge_count.append(pd.DataFrame(count_across_filtration, columns = [alpha], 
                                       index= pd.Index(degree_bins_rv, name="filtration")))
        if carry_over and len(set(carry_over.keys()) - {0, 1, 2}) > 0: #carry over is not empty and contains at least one higher order hyperedge            
            new_carry, new_carry_filt = remove_minimal_vertices(np.vstack(hyperedges_lower.index), hyperedges_weights, min_weights)
            carry_over = concatenate_hyperedges_dicts(carry_over, new_carry)
            carry_over_filtrations = concatenate_filtration_dicts(carry_over_filtrations, new_carry_filt)
        elif carry_over and len(set(carry_over.keys()) - {0, 1}) == 0: #carry over is not empty and contains at least one higher order hyperedge            
            carry_over = {}
    edge_count = pd.concat(edge_count, axis=1).sort_index()
    for i in range(l,1,-1):
        if i not in edge_count.columns:
            edge_count[i] = [0]*edge_count.shape[0] # add in any missing columns of zeroes
    return edge_count
    #return pd.DataFrame.from_dict(edge_count).set_index(pd.Index(bins[:-1], name ="filtration"))




def count_edge_original_and_control(number_vertices, hypergraph, node_metric, reps=10, seed=None, sparse_bin_set=False, using_vertex_degree=True):
    """
    Computes the edge counts of the original circuit and the controls across all dimensions.

    Parameters
    ----------
    number_vertices : int
        number of vertices in the hypergraph
    hypergraph : list of numpy.ndarray
        The i'th entry a 2D array of shape (i_edges,i)
    node_metric : pandas.Series or array-like
        Node metric values used for filtration. Length must equal the number
        of nodes referenced in hyperedges.
    reps : int, default 10
        Number of repetitions for the normalised shuffle
    seed : int, default 0
        Random seed to use for the shuffles
    sparse_bin_set : bool, default False
        If True, uses sparse binning based on unique metric values.
        If False, uses dense consecutive integer binning.
    using_vertex_degree : bool, default True
        If True the shuffle we recompute node metric as vertex degree, if false the inputted node metric will be used for the shuffled as well
    
    Returns
    -------
    original_edge_counts : pandas.DataFrame
        dict of DataFrames. original_edge_counts[i] is Dataframe
        of edge counts for the original hypergraph
        with filtration levels as index and hyperedge size as columns, 
        entries are number of hyperedges at that filtration level of that size,
        and additional column 'nodes' with number of nodes at that filtration level.
    control_edge_counts_mean : pandas.DataFrame
        dict of DataFrames. control_edge_counts_mean[i] is DataFrame 
        of the average edge count of the controls, same format as original_edge_counts
    control_edge_counts_sem : pandas.DataFrame
        dict of DataFrames. control_edge_counts_sem[i] is DataFrame 
        of the standard error of the mean of the  edge count of the controls,
        same format as original_edge_counts
    control_edge_counts_all : pandas.DataFrame
        dict of dict of DataFrames. First key of dict is repetition number
        second key of dict is dimension, the DataFrame is the same form as original_edge_counts
    """


    # edge_counts per dimension of the original graph
    edge_counts={}
    dims = [dim for dim in node_metric.keys() if dim > 1]
    for dim in dims:
        if dim in hypergraph and len(hypergraph[dim]) > 0:
            edge_counts[dim] = count_edges_on_filtration(hypergraph[dim], node_metric[dim], sparse_bin_set=sparse_bin_set)
        else:
            edge_counts[dim] = pd.DataFrame({'nodes':[0],2:[np.nan]})

    # edge_counts per dimension of the shuffles 
    shuffled = shuffle(hypergraph,number_vertices,reps=reps,seed=seed)     # shuffle the complex
    edge_counts_ctr ={}

    for r in range(reps):
        edge_counts_ctr[r]={}
        # shuffled gives vertex participation separated by position, this combines them
        if using_vertex_degree:
            combine_pos = [sum(j) for j in shuffled[1][r]] 
            df = pd.DataFrame(combine_pos).replace(np.nan,0).transpose()
            df.columns=range(2,len(combine_pos)+2)
        else:
            df = node_metric
        ctrl_dims = [dim for dim in df.keys() if dim > 1]
        for dim in ctrl_dims:
            if dim in shuffled[0][r] and len(shuffled[0][r][dim]) > 0 and dim in edge_counts:
                #edge_counts_ctr[r][dim] = count_edges_on_filtration(np.array(shuffled[0][r][dim]), df[dim], bins=np.array(edge_counts[dim].index)) 
                edge_counts_ctr[r][dim] = count_edges_on_filtration(np.array(shuffled[0][r][dim]), df[dim])
            else:
                edge_counts_ctr[r][dim] = pd.DataFrame({'nodes':[0],2:[np.nan]}) 

    # Average across shuffles and normalize the original 
    edge_counts_ctr_mean = {}
    edge_counts_ctr_sem = {}
    for dim in dims:
        dfs = [edge_counts_ctr[r][dim] for r in edge_counts_ctr if dim in edge_counts_ctr[r]]
        # Mean and SEM of controls 
        if len(dfs) > 0:
            edge_counts_ctr_mean[dim] = sum(dfs) / len(dfs)
            squared_diffs = [(df - edge_counts_ctr_mean[dim]) ** 2 for df in dfs]
            variance = sum(squared_diffs) / (len(dfs) - 1)
            edge_counts_ctr_sem[dim] = (variance ** 0.5) / (len(dfs) ** 0.5)
        else:
            edge_counts_ctr_mean[dim] = np.nan #should thus be nan or inf?
            edge_counts_ctr_sem[dim] = np.nan

    return (edge_counts, edge_counts_ctr_mean, edge_counts_ctr_sem, edge_counts_ctr)
        
def normalised_hyper_rich_club_curve(number_vertices, hypergraph, node_metric, weights='scaled',
                                    reps=10, seed=None, sparse_bin_set=False, ignore_infs=True,
                                    separated=False, using_vertex_degree=True):
    """
    Computes the normalised rich club curve across all dimensions

    Parameters
    ----------
    number_vertices : int
        number of vertices in the hypergraph
    hypergraph : dictionary of numpy.ndarray
        The i'th entry a 2D array of shape (i_edges,i)
    node_metric : pandas.Series or array-like
        Node metric values used for filtration. Length must equal the number
        of nodes referenced in hyperedges.
    weights : str
        Specifies how to weight the different size hyperedges within a dimension
        options are: 'scaled', 'unweighted'
    reps : int, default 10
        Number of repetitions for the normalised shuffle
    seed : int, default None
        Random seed to use for the shuffles, when None a random seed is chosen
    sparse_bin_set : bool, default False
        If True, uses sparse binning based on unique metric values.
        If False, uses dense consecutive integer binning.
    ignore_infs : bool, default True
        Ignores any infs created when dividing by the control, replacing them with np.nan
    separated : bool, default False
        If True returns normalised_rich_club, edge_counts_normalised, edge_counts, edge_count_ctr_mean, edge_count_ctr_sem, edge_count_ctr_all
    using_vertex_degree : bool, default True
        If True the shuffle we recompute node metric as vertex degree, if false the inputted node metric will be used for the shuffled as well
    
    Returns
    -------
    normalised_rich_club : pandas.DataFrame
        Where columns are hyperedge size and rows are node_metric filtration values
        entries are rich club value at that filtration level of that size.
    """
    if (not (int == node_metric.dtypes).all() or node_metric.min(axis=None) < 0) and (not sparse_bin_set or using_vertex_degree):
        print("WARNING: If node metric values are not non-negative integers, then sparse_bin_set should be True and using_vertex_degree should be False")

    edge_counts, edge_count_ctr_mean, edge_count_ctr_sem, edge_count_ctr_all= count_edge_original_and_control(number_vertices, hypergraph, node_metric, 
                                                                                                              reps=reps, seed=seed, sparse_bin_set=sparse_bin_set, 
                                                                                                              using_vertex_degree=using_vertex_degree)
    #return edge_counts, edge_count_ctr_mean, edge_count_ctr_sem, edge_count_ctr_all
    edge_counts_normalised_with_nodes = {}
    edge_counts_normalised = {}
    dims = [dim for dim in node_metric.keys() if dim > 1]
    for dim in dims:
        edge_counts_normalised_with_nodes[dim] = edge_counts[dim].divide(edge_count_ctr_mean[dim])
        edge_counts_normalised[dim] = edge_counts_normalised_with_nodes[dim].drop(columns=['nodes'])

    normalised_rich_club = []
    if weights == 'unweighted': # normalised weights unchanged
        normalised_rich_club = [edge_counts_normalised[dim] for dim in edge_counts_normalised.keys()]
    elif weights == 'scaled':   # normalised weights giving higher dimension bigger factor
        normalised_rich_club = [edge_counts_normalised[dim].multiply(edge_counts_normalised[dim].columns.astype(float), axis = 1)
                    /np.sum(edge_counts_normalised[dim].columns.astype(float)) for dim in edge_counts_normalised.keys()]
    elif weights == 'relative': # normalised weights weighted by the number of edges in each dimension
        normalised_rich_club = [edge_counts_normalised[dim].multiply(edge_counts[dim].divide(edge_counts[dim].sum(axis=1),axis=0)) for dim in edge_counts_normalised.keys()]
    else:
        raise ValueError("Invalid weights string")
    
    if ignore_infs:
        normalised_rich_club = [x.replace([np.inf, -np.inf], np.nan) for x in normalised_rich_club]

    normalised_rich_club = pd.concat([x.sum(axis=1) for x in normalised_rich_club], axis = 1, keys = dims)
    normalised_rich_club.sort_index(inplace=True) #If different dimensions have non overlapping filtrations, the index can be out of order
    if separated:
        return normalised_rich_club, edge_counts_normalised_with_nodes, edge_counts, edge_count_ctr_mean, edge_count_ctr_sem, edge_count_ctr_all
    else: 
        return normalised_rich_club



def normalised_hyper_rich_club_curve_from_graph(graph, maximal_simplices=True, weights='scaled',
                                    reps=10, seed=None, sparse_bin_set=False, ignore_infs=True, separated=False):
    """
    Computes the hypergraph whose has the same vertices as graph and the hyperedges are the
    (maximal) directed cliques of the graph

    Parameters
    ----------
    graph : adjacency matrix in sparse format
        The adjacency matrix of a directed graph
    maximal_simplices : bool, default True
        Considers only the maximal directed cliques of the graph as the hyperedges
    weights : str
        Specifies how to weight the different size hyperedges within a dimension
        options are: 'scaled', 'unweighted'
    reps : int, default 10
        Number of repetitions for the normalised shuffle
    seed : int, default None
        Random seed to use for the shuffles, when None a random seed is chosen
    sparse_bin_set : bool, default False
        If True, uses sparse binning based on unique metric values.
        If False, uses dense consecutive integer binning.
    ignore_infs : bool, default True
        Ignores any infs created when dividing by the control, replacing them with np.nan
    separated : bool, default False
        If True returns normalised_rich_club, edge_counts_normalised, edge_counts, edge_count_ctr_mean, edge_count_ctr_sem

    Returns
    -------
    normalised_rich_club : pandas.DataFrame
        Where columns are hyperedge size and rows are node_metric filtration values
        entries are rich club value at that filtration level of that size.
    """
    
    flagser_output = flagser_count(graph, return_simplices=True, max_simplices=maximal_simplices, containment=True)
    node_metric = pd.DataFrame(flagser_output['contain_counts']).replace(np.nan,0).astype(int)
    node_metric.columns+=1
    hyperedges = flagser_output['simplices']

    #If using nonmaximal simplices add the 0 and 1-dim simplices, as flagser does not return these.
    if not maximal_simplices:
        hyperedges[1] = [range(graph.shape[0])]
        if isinstance(graph,(ss.csr_matrix,ss.csc_matrix)):
            graph = graph.tocoo()
        elif isinstance(graph,np.ndarray):
            graph = ss.coo_matrix(graph)
        else:
            raise TypeError("Adjacency matrix is invalid type, must be numpy array or scipy sparse matrix")
        
        hyperedges[2] = [[graph.row[i],graph.col[i]] for i in range(len(graph.row))]
        

    hyperedges = {i+1:np.array(flagser_output['simplices'][i]) for i in range(len(flagser_output['simplices'])) if i > 0}
    return normalised_hyper_rich_club_curve(graph.shape[0], hyperedges, node_metric, weights=weights, reps=reps, seed=seed, 
                                              sparse_bin_set=sparse_bin_set, ignore_infs=ignore_infs, separated=separated)

##############################################################################################################################
# Loader Functions


#np.seterr(divide='ignore', invalid='ignore') # ignore divide by zero warnings

# Loads connectomes adjacency matrices
# Input:
#      contype can be any of "c_elegans", "bbp", "dros_larva", "microns", "ER"(Erdos-Renyi) "CM" (configuration model)
#      root is address of folder containing this file
#      config_model if True takes returns a configuration model shuffle of that connectome
#      seed is the random seed to use for configuration model
#      n is number of vertices, only needed for ER
#      p is density, only needed for ER
# Output: the adjacency matrix in coo format
def loader(contype,root='',config_model=False,seed=None,n=0,p=0):
    if contype == 'c_elegans':
        from data_loader import load_celegans_chem
        g = load_celegans_chem(8,root)
        #label = f'c_elegans'
    elif contype == 'bbp':
        from data_loader import load_bbp
        g = load_bbp(root)
        #g = bbp(i=args.run, allowed_neuron_types=args.bbp_allowed_neuron_types)
        #g = g.toarray()
        #label = f'bbp'
    elif contype == 'dros_larva':
        from data_loader import load_dros_larva
        g = load_dros_larva(connections = "axo-dendritic",root=root)
        #label = f'dros_larva'
    elif contype == 'microns':
        from data_loader import load_microns
        g = load_microns(root)
        #label = f'microns'
    elif contype == 'ER':
        G = nx.erdos_renyi_graph(n,p,seed=seed,directed=True)
        return nx.to_scipy_sparse_array(G,format='coo')
    else:
        print("ERROR: Invalid input type")
        return None

    g = g.matrix
    g.eliminate_zeros() # c_elegans has entries with zero value, these are removed (and other connectomes to be safe)

    if config_model:
        g.data = np.array([1]*len(g.data)) # set all nonzero entries in adjacency matrix to 1
        dout = np.array(np.sum(g,axis=1).transpose())[0]
        din = np.array(np.sum(g,axis=0))[0]
        G = nx.directed_configuration_model(din,dout,seed=seed)
        G = nx.DiGraph(G)
        G.remove_edges_from(nx.selfloop_edges(G))
        g = nx.to_scipy_sparse_array(G,format='coo')
        #label = f'microns'
    #if args.contype == 'mouse_v1':
    #    from connectome_loader.sparse_reconstructions import mouse_v1
    #    g = mouse_v1(run=args.run, fraction=args.fraction, allowed_neuron_types=args.mouse_v1_allowed_neuron_types)
    #    g = g.toarray()
    #    label = f'mouse_v1_{args.mouse_v1_allowed_neuron_types}_{args.fraction:.2f}'

    #if args.contype == 'er':
    #    from connectome_loader.null_models import random_with_p
    #    np.random.seed(args.run)
    #    g = random_with_p(N=args.N, p=args.p)
    #    label = f'random_ER_N{args.N}_p{args.p:.02}_{args.run:03}'

    #if args.contype == 'q_rewiring':
    #    g = q_rewiring()[1]
    #    label = f'q_rewiring'
    
    return g#, label


##########################################################################################################
#Run Specific Circuits
#TODO set this up to use config files


def hyper_rich_club_celegans(root='..', separated=True, weights='relative', seed=None,reps=10):
    return normalised_hyper_rich_club_curve_from_graph(loader('c_elegans',root=root),separated=separated,weights=weights,seed=seed,reps=reps)

def hyper_rich_club_dros_larva(root='..', separated=True, weights='relative', seed=None,reps=10):
    return normalised_hyper_rich_club_curve_from_graph(loader('dros_larva',root=root),separated=separated,weights=weights,seed=seed,reps=10)    

def hyper_rich_club_bbp(root='..', separated=True, weights='relative', seed=None, reps=10, address=None):
    if address == None:
        return normalised_hyper_rich_club_curve_from_graph(loader('bbp',root=root),separated=separated,weights=weights,seed=seed,reps=10)
    else:
        #simplices saved in ./output/max_simplices/bbp_v5.pkl
        with open(address,'rb') as f:
            flagser_output = pk.load(f)
        node_metric = pd.DataFrame(flagser_output['contain_counts']).replace(np.nan,0).astype(int)
        node_metric.columns+=1
        #hyperedges = flagser_output['simplices']
        hyperedges = {i+1:np.array(flagser_output['simplices'][i]) for i in range(len(flagser_output['simplices'])) if i > 0}
        n = 31346
        print('Loaded simplices')

        #hyperedges = [np.array(edges) for edges in hyperedges]
        return normalised_hyper_rich_club_curve(n, hyperedges, node_metric, weights=weights, reps=reps, seed=seed, separated=separated)

def hyper_rich_club_microns(root='..', separated=True, weights='relative', seed=None,reps=10):
    return normalised_hyper_rich_club_curve_from_graph(loader('microns',root=root),separated=separated,weights=weights,seed=seed,reps=10)    

def hyper_rich_club_ER(circuit, reps=10, root='..', weights='relative', ERseed=0, shuffleseed=0):
    if circuit == 'c_elegans':
        n=324
        p=0.02
    elif circuit == 'dros_larva':
        n=2956
        p=0.007
    elif circuit == 'bbp':
        n=31346
        p=0.008
    else:
        raise ValueError("Invalid circuit type")
    rich_club_all_reps = [normalised_hyper_rich_club_curve_from_graph(loader('ER',n=n,p=p,config_model=False,root=root,seed=ERseed+i*97),
                                      separated=False,weights=weights,seed=shuffleseed+i*79)
                                               for i in range(reps)]
    
    return pd.concat(rich_club_all_reps).groupby(level=0).mean()

def hyper_rich_club_config_model(circuit, reps=10, root='..', weights='relative', CMseed=0, shuffleseed=0):
    rich_club_all_reps = [normalised_hyper_rich_club_curve_from_graph(loader(circuit,config_model=True,root=root,seed=CMseed+i*97),
                                      separated=False,weights=weights,seed=shuffleseed+i*79)
                                               for i in range(reps)]
    
    return pd.concat(rich_club_all_reps).groupby(level=0).mean()

###########################################################################################################
# Plotting functions

def plot_hyper_rich_club_by_dimension_with_separated(rich_club_curve, save_name, plot_zero_tail=False):
    """
    Plots the output of normalised_hyper_rich_club_curve, with separated=True

    Parameters
    ----------
    rich_club_curve : tuple
        The output from normalised_hyper_rich_club_curve, with separated=True
    save_name : str
        Where to save the figure, without the .pdf
    plot_zero_tail : bool
        If this is False then the tail of zeroes is removed for the normalised rich club plot (rightmost)
    """
    normalised_rich_club = rich_club_curve[0]
    edge_counts_normalised_with_nodes = rich_club_curve[1]
    edge_counts = rich_club_curve[2]
    edge_counts_ctr_mean = rich_club_curve[3]
    edge_counts_ctr_sem = rich_club_curve[4]
    #normalised_rich_club, edge_counts_normalised_with_nodes, edge_counts, edge_count_ctr_mean, edge_count_ctr_sem, edge_count_ctr_all

    combine_str = []
    for dim in edge_counts.keys():
        fig, axs = plt.subplots(1,3, figsize=(15,5))
            
        # Plot node counts 
        axs[0].plot(edge_counts[dim]['nodes'], label = f"original", c = f"black")
        axs[0].plot(edge_counts_ctr_mean[dim]['nodes'], label = f"controls", c = f"C0")
        '''axs[0].fill_between(range(len(nodes_ctr[dim])), 
                           nodes_ctr[dim].mean(axis=1)-nodes_ctr[dim].sem(axis=1), 
                           nodes_ctr[dim].mean(axis=1)+nodes_ctr[dim].sem(axis=1), 
                           alpha=0.2, color = f"C0")'''

        for i, col in enumerate(edge_counts[dim].columns[::-1]): 
            #Normalized per dimension
            axs[1].plot(edge_counts_normalised_with_nodes[dim][col], label = f"# v = {col}", c = f"C{i}")

        #remove tail of zeroes if required
        normalised_rich_club_dim = normalised_rich_club[dim]
        if not plot_zero_tail:
            normalised_rich_club_dim = normalised_rich_club_dim[np.flip(np.cumsum(np.flip(normalised_rich_club_dim)))>0]


        # Aggregated
        axs[2].plot(normalised_rich_club_dim, color = "rebeccapurple")

        # Format
        for ax in axs: 
            ax.set_xlabel(f"Node participation dimension {dim}")
            ax.set_ylabel("Normalized density")
            ax.spines[["top", "right"]].set_visible(False)
        axs[0].set_ylabel("Node count")
        axs[0].legend()
        axs[0].set_yscale('log')
        axs[1].legend()
        fig.suptitle(f"Simplicial rich club dimension {dim} ({dim+1} nodes)", y=1)
        axs[0].set_title("Nodes across filtration")
        axs[1].set_title("Normalized per dimension")
        axs[2].set_title("Normalized rich club")
        #axs[0].set_yscale("log")

        figname = save_name+'_dim'+str(dim)+'.pdf'
        fig.savefig(figname, transparent=True, bbox_inches='tight')
        combine_str.append(figname)
        plt.close()

    merger = PdfWriter()
    for pdf in combine_str:
        merger.append(pdf)
    merger.write(save_name + '.pdf')
    merger.close()


def plot_hyper_rich_club_by_dimension(rich_club_curve, save_address='', row_length=5, plot_zero_tail=False, figsize=(18,6)):
    """
    Plots the output of normalised_hyper_rich_club_curve, with separated=True

    Parameters
    ----------
    rich_club_curve : pandas.DataFrame
        The output from normalised_hyper_rich_club_curve, with separated=False
    save_address : str
        Where to save the figure
    plot_zero_tail : bool
        If this is False then the tail of zeroes is removed for the normalised rich club plot (rightmost)
    """
    fig, axs = plt.subplots(int(rich_club_curve.shape[1]/(row_length+1))+1,min(rich_club_curve.shape[1],row_length), figsize=figsize, squeeze=False)

    plt.subplots_adjust(hspace=0.4)
    combine_str = []
    for i in range(len(rich_club_curve.columns)):

        dim = rich_club_curve.columns[i]
        normalised_rich_club_dim = rich_club_curve[dim]
        if not plot_zero_tail:
            normalised_rich_club_dim = normalised_rich_club_dim[np.flip(np.cumsum(np.flip(normalised_rich_club_dim)))>0]
        axs[int(i/row_length),i%row_length].plot(normalised_rich_club_dim, color = "rebeccapurple")
        #axs[int(i/5),i%5].set_title(f"Normalized rich club {dim}-edges")
        axs[int(i/row_length),i%row_length].set_xlabel(f"{dim}-degree")
        axs[int(i/row_length),i%row_length].set_title(f"{dim}-hyper rich club")
        axs[int(i/row_length),i%row_length].axhline(1, linestyle='--')
        
    if len(rich_club_curve.columns)%row_length != 0:
        for i in range(len(rich_club_curve.columns)%row_length,row_length):
            axs[-1,i].remove()

    #axs[,0].set_ylabel("Normalized density")
    if(save_address != ''):
        fig.savefig(save_address, transparent=False, bbox_inches='tight')
        plt.close()
    else:
        plt.show()