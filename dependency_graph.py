# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle
import networkx as nx

nlp = spacy.load('en_core_web_sm')


# def dependency_adj_matrix(text):
#     # https://spacy.io/docs/usage/processing-text
#     document = nlp(text)
#     seq_len = len(text.split())
#     matrix = np.zeros((seq_len, seq_len)).astype('float32')
#
#     for token in document:
#         if token.i < seq_len:
#             matrix[token.i][token.i] = 1
#             # https://spacy.io/docs/api/token
#             for child in token.children:
#                 if child.i < seq_len:
#                     matrix[token.i][child.i] = 1
#                     matrix[child.i][token.i] = 1
#
#     return matrix
def dependency_adj_matrix(text, aspect_term):
    # https://spacy.io/docs/usage/processing-text
    document = nlp(text)
    edges = []
    for token in document:
        # FYI https://spacy.io/docs/api/token
        for child in token.children:
            edges.append((token.i, child.i))
    graph = nx.Graph(edges)
    seq_len = len(text.split())
    matrix = np.zeros((seq_len, seq_len)).astype('float32')
    text_left, _, _ = text.partition(aspect_term)
    start = len(text_left.split())
    end = start + len(aspect_term.split())
    asp_idx = [i for i in range(start, end)]
    dist_matrix = seq_len * np.ones((seq_len, len(asp_idx))).astype('float32')
    dist_matrix_2 = seq_len * np.ones((seq_len, len(asp_idx))).astype('float32')
    dist_matrix_3 = seq_len * np.ones((seq_len, len(asp_idx))).astype('float32')
    asp_matrix = np.zeros((seq_len,seq_len)).astype('float32')
    asp_matrix_2 = np.zeros((seq_len,seq_len)).astype('float32')
    asp_matrix_3 = np.zeros((seq_len,seq_len)).astype('float32')
    for i, asp in enumerate(asp_idx):
        for j in range(seq_len):
            try:
                dis = nx.shortest_path_length(graph,source=asp, target=j)
                if dis <= 1:
                    dist_matrix[j][i] = 1
                else:
                    dist_matrix[j][i] = 0
                if dis == 2:
                    dist_matrix_2[j][i] = 1
                else:
                    dist_matrix_2[j][i] = 0
                if dis == 3:
                    dist_matrix_3[j][i] = 1
                else:
                    dist_matrix_3[j][i] = 0
            except:
                dis = seq_len / 2
                if dis <= 1:
                    dist_matrix[j][i] = 1
                else:
                    dist_matrix[j][i] = 0
                if dis == 2:
                    dist_matrix_2[j][i] = 1
                else:
                    dist_matrix_2[j][i] = 0
                if dis == 3:
                    dist_matrix_3[j][i] = 1
                else:
                    dist_matrix_3[j][i] = 0
            # for j in range(seq_len):
            #     try:
            #         dist_matrix[j][i] = nx.shortest_path_length(graph, source=asp, target=j)
            #     except:
            #         dist_matrix[j][i] = seq_len / 2
    dist_matrix = np.min(dist_matrix, axis=1)
    dist_matrix_2 = np.min(dist_matrix_2, axis=1)
    dist_matrix_3 = np.min(dist_matrix_3, axis=1)

    for i, asp in enumerate(asp_idx):
        for j in range(seq_len):
                asp_matrix[j][asp] = 1
                asp_matrix[asp][j] = 1
                asp_matrix_2[j][asp] = 1
                asp_matrix_2[asp][j] = 1
                asp_matrix_3[j][asp] = 1
                asp_matrix_3[asp][j] = 1
        asp_matrix[asp][asp] = 1
        asp_matrix_2[asp][asp] = 1
        asp_matrix_3[asp][asp] = 1
    asp_matrix1 = dist_matrix * asp_matrix
    asp_matrix2 = dist_matrix_2 * asp_matrix_2
    asp_matrix3 = dist_matrix_3 * asp_matrix_3
    asp_matrix = asp_matrix1 + asp_matrix1.T
    asp_matrix_2 = asp_matrix2 + asp_matrix2.T
    asp_matrix_3 = asp_matrix3 + asp_matrix3.T
    for i in range(seq_len):
        for j in range(seq_len):
            if(asp_matrix[i][j] > 1):
                asp_matrix[i][[j]] = 1
            if(asp_matrix_2[i][j] > 1):
                asp_matrix_2[i][[j]] = 1
            if(asp_matrix_3[i][j] > 1):
                asp_matrix_3[i][[j]] = 1
   # print(asp_matrix)



    for token in document:
        if token.i < seq_len:
            matrix[token.i][token.i] = 1
            # https://spacy.io/docs/api/token
            for child in token.children:
                if child.i < seq_len:
                    matrix[token.i][child.i] = 1
                    matrix[child.i][token.i] = 1
    #print(matrix)
    return matrix, asp_matrix, asp_matrix_2, asp_matrix_3

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    idx2asp_graph = {}
    idx2asp_graph_2 = {}
    idx2asp_graph_3 = {}
    fout = open(filename+'.graph', 'wb')
    fout_asp = open(filename+'.asp_graph', 'wb')
    fout_asp2 = open(filename+'.asp2_graph', 'wb')
    fout_asp3 = open(filename+'.asp3_graph', 'wb')
    for i in range(0, len(lines), 3):
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        adj_matrix, aspect_matrix, aspect_matrix_2, aspect_matrix_3 = dependency_adj_matrix(text_left+' '+aspect+' '+text_right, aspect)
        idx2graph[i] = adj_matrix
        idx2asp_graph[i] = aspect_matrix
        idx2asp_graph_2[i] = aspect_matrix_2
        idx2asp_graph_3[i] = aspect_matrix_3
    pickle.dump(idx2graph, fout)
    pickle.dump(idx2asp_graph, fout_asp)
    pickle.dump(idx2asp_graph_2, fout_asp2)
    pickle.dump(idx2asp_graph_3, fout_asp3)
    fout.close()
    fout_asp.close()
    fout_asp2.close()
    fout_asp3.close()

if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')