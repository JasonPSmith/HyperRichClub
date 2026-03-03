import requests
import pandas
import numpy
import conntility
import os
import tarfile

local_root = "../network_data/"

############################################################################################
#C Elegans
#Data comes from https://wormwiring.org/pages/witvliet.html 
#            and https://www.wormatlas.org/images/NeuronType.xls
#h5 file taken from conntility examples

#stage can take values 1,2,3,4,5,6,7,8 corresponding to the C. Elegans life stages
def load_celegans_chem(stage=8,root=''):
    return conntility.ConnectivityMatrix.from_h5(root+local_root+"C_elegans_witvliet.h5", prefix="chemical").default(stage)

def load_celegans_elec(stage=8,root=''):
    return conntility.ConnectivityMatrix.from_h5(root+local_root+"C_elegans_witvliet.h5", prefix="electrical").default(stage)

def load_celegans_comb(stage=8,root=''):
    return conntility.ConnectivityMatrix.from_h5(root+local_root+"C_elegans_witvliet.h5", prefix="combined").default(stage)

############################################################################################
#Dros Larva
#Data comes from https://www.science.org/doi/suppl/10.1126/science.add9330/suppl_file/science.add9330_data_s1_to_s4.zip
#Associated paper: Winding et al., 2023 - The connectome of an insect brain. https://www.science.org/doi/10.1126/science.add9330
#h5 file created using example in conntility
#connections can be: "all", "axo-dendritic", "axo-axonic", "dendro-dendritic", "dendro-axonic"
def load_dros_larva(connections = "axo-dendritic",root=''):
    M = conntility.ConnectivityMatrix.from_h5(root+local_root+"dros_larva.h5")
    if connections != "all":
        M = M.filter("type").eq(connections)
    return M


############################################################################################
#Blue Brain v5
#Data comes from https://zenodo.org/records/10812497
#           file BlobStimReliability_O1v5-SONATA_Baseline/working_dir/connectome.h5
def load_bbp(root=''):
    fn = root+local_root+"bbp_v5.h5"
    if not os.path.isfile(fn):
        with tarfile.open(root+local_root+"bbp_v5.tar.xz") as f:
            f.extract("bbp_v5.h5",path=root+local_root)
    return conntility.ConnectivityMatrix.from_h5(root+local_root+"bbp_v5.h5")

############################################################################################
#MICRoNS


#displays progress bar when downloading, taken from https://gist.github.com/yanqd0/c13ed29e29432e3cf3e7c38467f42f51
def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
        desc=fname,
        total=total,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

def load_microns(root='', restrict_to_interior=True,interval_z = [700000, 1000000], interval_x = [650000, 950000]):
    fn = root+local_root+"microns_mm3_connectome.h5"
    if not os.path.isfile(fn):
        download('https://zenodo.org/records/8364070/files/microns_mm3_connectome.h5',fn)
    # Create ConnectivityMatrix object for the analysis
    M = conntility.ConnectivityMatrix.from_h5(fn, "condensed")
    M.add_vertex_property("gid", M.gids)
    if restrict_to_interior:
        # Excluding boundary and restricting to exc types 
        M = M.index("x_nm").gt(interval_x[0]).index("x_nm").lt(interval_x[1]).index("z_nm").gt(interval_z[0]).index("z_nm").lt(interval_z[1])
        exc_types=['23P', '4P', '5P_IT', '5P_NP', '5P_PT', '6CT', '6IT', 'BPC']
        M = M.index("cell_type").isin(exc_types)
    return M
