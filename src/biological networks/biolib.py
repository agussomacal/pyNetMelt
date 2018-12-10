import rpy2.robjects as robjects
import pandas as pd
import numpy as np
from time import time
import os

from config import *


class Load:

    @staticmethod
    def gene_sequence_bed_data(path=GTEx_dir):
        data = pd.read_csv("{}/genes_hsa_sort.bed".format(path), sep="\t", header=None)
        data.columns = ["cromosoma", "pos_inicial", "pos_final", "ensembl", "score", "where_to_read"]
        return data

    @staticmethod
    def gene_sequence_entrez_data(path=GTEx_dir):
        data = pd.read_csv("{}/genes_hsa_sort_entrez.csv".format(path), sep=",")
        return data

    @staticmethod
    def ontology_similarity_data(ontology="BP", path=GTEx_dir):
        data = robjects.r['readRDS']("{}/MultiplexNetworks/{}adjacency_matrix_entrez.rds".format(path, ontology))
        return pd.DataFrame(np.array(data), columns=np.array(data.colnames, dtype=int),
                            index=np.array(data.rownames, dtype=int))

    @staticmethod
    def ppi(path=GTEx_dir, filename="Human-PPI-HIPPIE.txt"):
        ppi_df = pd.read_csv("{}/{}".format(path, filename), sep="\t",
                             names=["protein1", "entrezid1", "protein2", "entrezid2", "score", "meassure_type"])
        ppi_df = ppi_df[["entrezid1", "entrezid2", "score"]]
        ppi_df["entrezid1"] = ppi_df["entrezid1"].apply(int)
        ppi_df["entrezid2"] = ppi_df["entrezid2"].apply(int)
        return ppi_df

    @staticmethod
    def coexp(tissue, dataset="reads.dgelist", path=byTissue_data_dir):
        data = robjects.r['readRDS']("{}/{}/coexp_{}.RDS".format(path, dataset, tissue))
        return pd.DataFrame(np.array(data), columns=np.array(data.colnames, dtype=int),
                            index=np.array(data.rownames, dtype=int))

    @staticmethod
    def gen_meansd_data(filename):
        data = robjects.r['readRDS']("{}/{}.RDS".format(byTissue_data_dir, filename))
        return pd.DataFrame(np.array(data), columns=["global"], index=np.array(data.names, dtype=int))

    @staticmethod
    def gen_meansd_bytissue_data(filename):
        data = robjects.r['readRDS']("{}/{}.RDS".format(byTissue_data_dir, filename))
        return pd.DataFrame(np.array(data), columns=np.array(data.colnames), index=np.array(data.rownames, dtype=int))

    @staticmethod
    def diseases(path=GTEx_dir):
        data = robjects.r("""
                            load_dataGV <- function(filepath){
                                load(paste0(filepath,"/data.table.genAlt.Rdata"))
                                as.data.frame(dataGV[dataGV$associationType %in% genAlt,])
                            }
                        """)(path)
        return pd.DataFrame(np.transpose(np.array(data)), columns=np.array(data.colnames))

    @staticmethod
    def W(path, filename):
        t0 = time()
        fname = "{}/{}".format(path, filename)
        if fname[-3:] != ".gz":
            fname += ".gz"

        if not os.path.isfile(fname):
            print("W not in folder")
            return None
        else:
            print("loading W...")
            w = pd.read_csv(fname, compression='gzip', index_col=0)
            w.columns = pd.to_numeric(w.columns)  # de otro modo las deja como strings
            w.index = pd.to_numeric(w.index)  # de otro modo las deja como strings
        print("Duracion: {}".format(time() - t0))
        return w


class Save:

    @staticmethod
    def W(w, path, filename):
        if type(w) == pd.core.frame.DataFrame:
            t0 = time()
            fname = "{}/{}".format(path, filename)
            if fname[-3:] != ".gz":
                fname += ".gz"

            if not os.path.isfile(fname):
                print("Saving W network...")
                w.to_csv(fname, compression='gzip')
            else:
                print("Already calculated...")
            print("Duracion: {}".format(time() - t0))


########################################################################################################################

###############################################################################
# -----------------PPI data-----------------------
class PPI:
    def __init__(self, ppi_df):
        self.data = ppi_df

        def change_order(row):
            if row[0] > row[1]:
                temp = row[0]
                row[0] = int(row[1])
                row[1] = int(temp)
            return row

        self.data = self.data.apply(change_order, axis=1)
        self.data = self.data.sort_values(["entrezid1", "entrezid2"])
        self.data["entrezid1"] = self.data["entrezid1"].apply(int)
        self.data["entrezid2"] = self.data["entrezid2"].apply(int)

        self.gene_names = list(set(self.data["entrezid1"]).union(set(self.data["entrezid2"])))

        print("First 5 genes: {}".format(self.gene_names[0:5]))
        print("ppi data:")
        print(self.data.head())

    def filter_gene_names(self, gene_names):
        self.gene_names = [gene for gene in gene_names if gene in self.gene_names]
        self.data = self.data[
            self.data["entrezid1"].isin(self.gene_names) & self.data["entrezid2"].isin(self.gene_names)]

        print("Number of genes at end: {}".format(len(self.gene_names)))
        print("Shape of ppi after filtering: {}".format(self.data.shape))

    def apply_threshold(self, umbral):
        self.data = self.data.groupby(["entrezid1", "entrezid2"]).sum()
        self.data = self.data[self.data["score"] >= umbral]
        self.data = self.data.reset_index()  # para sacarle el index herarquico y quede solo 2 columnas mas legibles

        print("ppi shape after thresholding: {}".format(self.data.shape))

    def binarize(self):
        self.data["score"] = 1
        # self.data = self.data[["entrezid1","entrezid2"]]

    def to_matrix(self, sparse=True):
        matrix = pd.DataFrame(0, columns=self.gene_names, index=self.gene_names)

        for index, row in self.data.iterrows():
            matrix.loc[row["entrezid1"], row["entrezid2"]] = row["score"]
            matrix.loc[row["entrezid2"], row["entrezid1"]] = row["score"]

        if sparse:
            return csc_matrix(matrix.as_matrix())
        else:
            return matrix

    def to_np(self, node_names):
        data = nx.from_pandas_edgelist(self.data, source="entrezid1", target="entrezid2",
                                       edge_attr="score").to_undirected()
        data.add_nodes_from([gen for gen in node_names if
                             gen not in self.gene_names])  # agrega nodos si es que en la lista que viene hay mas nodos.
        self.gene_names = node_names
        self.data = nx.to_numpy_matrix(data, nodelist=node_names, weight="score")

    def isolate_chaperones(self, deg_threshold):
        deg = np.squeeze(np.array(np.sum(self.data > 0, axis=0)))  # para sumar degree y no strength se pone el >0
        print("Isolating {} chaperon genes".format(np.sum(deg >= deg_threshold)))
        self.data[deg >= deg_threshold, :][:,
        deg >= deg_threshold] = 0  # pone a 0 las interacciones de los genes chaperones
        # self.gene_names = list(np.array(self.gene_names)[deg<deg_threshold])

        h = np.histogram(deg, bins=int(np.sqrt(len(deg))))
        plt.figure()
        plt.loglog((h[1][0:-1] + h[1][1:]) / 2, h[0], c="blue")
        plt.title("ppi degree distribution")
        plt.xlabel("degree")
        plt.ylabel("counts")
        plt.axvline(deg_threshold, c="red")
        plt.savefig("ppi_chaperon_filtering.svg")


###############################################################################
# ------------------Coexpression data---------------------
class COEXP:
    def __init__(self, coexp_df):
        self.gene_names = list(set(coexp_df.columns).union(set(coexp_df.index)))
        self.data = coexp_df
        # -------put autocorrelatin to 0-------------
        for i in range(self.data.shape[0]):
            self.data.iloc[i, i] = 0

    def filter_gene_names(self, gene_names):
        self.gene_names = [gene for gene in gene_names if gene in self.gene_names]
        self.data = self.data.loc[self.data.index.isin(self.gene_names), self.data.columns.isin(self.gene_names)]

        print("Number of genes at end: {}".format(len(self.gene_names)))
        print("Shape of COEXP after filtering: {}".format(self.data.shape))

    def blend_by_signal(self, signal):

        ass = np.logspace(1, 1.7, 12)
        coexp_assort_by_exponent = np.zeros(len(ass))
        hw = []
        hs = []
        titulo = "potencia"
        for i, a in enumerate(ass):
            print(a, end=" ")
            coexp_assort_by_exponent[i] = assortativity(np.power(coexp.data, a), signal.values)
            hw.append(np.histogram(np.power(coexp.data, a).ravel(), bins=coexp.data.shape[0]))
            hs.append(
                np.histogram(np.sum(np.power(coexp.data, a), axis=0).ravel(), bins=int(np.sqrt(coexp.data.shape[0]))))

        assortativity(self.data, signal)

    def apply_threshold(self, umbral):
        self.data[self.data <= umbral] = 0

    def binarize(self):
        self.data[self.data != 0] = 1

    def to_sparse(self):
        self.data = csc_matrix(self.data.as_matrix())

    def convert_nan_to_zero(self):
        self.data = self.data.fillna(0)

    def to_np(self, node_names):
        #        data = nx.from_pandas_adjacency(self.data).to_undirected()
        #        data.add_nodes_from([gen for gen in node_names if gen not in self.gene_names])#agrega nodos si es que en la lista que viene hay mas nodos.
        #        self.gene_names = node_names
        #        self.data = nx.to_numpy_matrix(data, nodelist=node_names, weight="weight")
        self.data = self.data.loc[node_names, node_names].values


###############################################################################
# ------------------NN data---------------------
class NN:
    def __init__(self, gen_mean):
        self.gene_names = gen_mean.index
        self.data = gen_mean

    def filter_gene_names(self, gene_names):
        self.gene_names = [gene for gene in gene_names if gene in self.gene_names]
        self.data = self.data[self.data.index.isin(self.gene_names)]

        print("Number of genes at end: {}".format(len(self.gene_names)))
        print("Shape of NN after filtering: {}".format(self.data.shape))

    def apply_threshold(self, umbral, r):
        self.data[self.data <= umbral] = r
        self.data[self.data > umbral] = 1

    def make_network_np(self, node_names):
        self.data = np.outer(self.data.loc[node_names].values, self.data.loc[node_names].values)


###############################################################################
# ------------------Ontology data---------------------
class Ontology:
    def __init__(self, ontology_df):
        self.data = ontology_df
        self.gene_names = self.data.columns

    def filter_gene_names(self, gene_names):
        self.gene_names = [gene for gene in gene_names if gene in self.gene_names]
        self.data = self.data.loc[self.data.index.isin(self.gene_names), self.data.columns.isin(self.gene_names)]

        print("Number of genes at end: {}".format(len(self.gene_names)))
        print("Shape of ontology after filtering: {}".format(self.data.shape))

    def apply_threshold(self, umbral):
        self.data[self.data <= umbral] = 0

    def binarize(self):
        self.data[self.data != 0] = 1

    def convert_nan_to_zero(self):
        self.data = self.data.fillna(0)

    def to_np(self, node_names):
        self.data = self.data.loc[node_names, node_names].values


###############################################################################
# -------------------Expression Data----------------------
class EXP:
    def __init__(self):
        self.gen_mean = Load.gen_meansd_data(filename="gene_mean_activation")
        self.gen_sd = Load.gen_meansd_data(filename="gene_sd_activation")
        self.gen_mean_by_tissue = Load.gen_meansd_bytissue_data(filename="gene_mean_activation_by_tissue")
        self.gen_sd_by_tissue = Load.gen_meansd_bytissue_data(filename="gene_sd_activation_by_tissue")
        self.gene_names = self.gen_mean.index
        self.ngenes = len(self.gene_names)



