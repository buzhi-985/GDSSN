
import string
import pandas as pd
import os


def TransMatrix(sample, outpath):
    sample.set_index(sample['GeneSymbol'], drop=True, inplace=True)
    sample.drop('GeneSymbol', axis=1, inplace=True)
    sample = sample.T
    sample.to_csv(outpath)
    return outpath


def getGrpah():
    gene = pd.read_csv("gene_relationship_ke.csv", names=['a', 'b'])
    gene1 = gene.copy()
    gene['a'] = gene['a'].apply(lambda x: x + "_geneExp")
    gene['b'] = gene['b'].apply(lambda x: x + "_geneExp")

    gene1['a'] = gene1['a'].apply(lambda x: x + "_methylation")
    gene1['b'] = gene1['b'].apply(lambda x: x + "_methylation")
    gene = pd.concat([gene, gene1], axis=0)
    return gene


def csvToH5ad(inpath, outpath):
    import scanpy as sc
    mydata = sc.read_csv(inpath)
    mydata.write(outpath)


def firstMatch(df, gene):
    for i in range(df.shape[0]):
        gene.replace(df['GeneSymbol'][i], i, inplace=True)
    gene.to_csv("temp.csv", index=False)
    p_data = pd.read_csv("temp.csv")
    for i in range(26):
        p_data.drop(p_data[p_data['a'].str.contains(string.ascii_uppercase[i])].index, axis=0, inplace=True)
        p_data.drop(p_data[p_data['b'].str.contains(string.ascii_uppercase[i])].index, axis=0, inplace=True)
    p_data.reset_index(drop=True, inplace=True)
    os.remove("temp.csv")
    return p_data


def deletSample(df, p_data):
    newdata = pd.DataFrame()
    for i in range(p_data.shape[0]):
        pos1 = int(p_data['a'][i])
        pos2 = int(p_data['b'][i])
        if pos1 != -1:
            newdata = pd.concat([newdata, df.loc[[pos1]]], copy=False)
            p_data.replace(p_data['a'][i], -1, inplace=True)
        if pos2 != -1:
            newdata = pd.concat([newdata, df.loc[[pos2]]], copy=False)
            p_data.replace(p_data['b'][i], -1, inplace=True)

    newdata = pd.concat([newdata, df[df['GeneSymbol'].str.contains("miRNAExp")]], copy=False)
    newdata.reset_index(drop=True, inplace=True)
    return newdata


# file_list = ["brca.csv"]
file_list = os.listdir("data/expression")
for item in file_list:
    print(item)
    b = item.index(".csv")
    cancer = item[0:b]
    df = pd.read_csv(f"data/expression/{cancer}.csv")
    df.rename(columns={'GeneSymbol_Platform': 'GeneSymbol'}, inplace=True)
    gene_relation = getGrpah()
    temp = gene_relation.copy()
    first_edge = firstMatch(df, temp)
    mid_sample = deletSample(df, first_edge)
    last_edge = firstMatch(mid_sample, gene_relation)
    last_edge.to_csv("data/preprocessed/{}_matrix.csv".format(cancer), index=False, header=None)
    datapath = TransMatrix(mid_sample, "data/preprocessed/{}.csv".format(cancer))
    csvToH5ad(datapath, "data/preprocessed/{}.h5ad".format(cancer))
