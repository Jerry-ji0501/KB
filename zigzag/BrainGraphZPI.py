import scipy.io as sio
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import zigzagtools as zzt
import dionysus as d
from data_loader import data_loader


def BrainZPI(BrainGraphs,Timesnap,NVertices=100,scaleParameter=1.0,maxDimHoles=2):
    BrainGraphsNet = []
    plt.figure(num=None, figsize=(16, 1.5), dpi=80, facecolor='w', edgecolor='k')
    for i in range(0, Timesnap):
        print('Graph' + str(i))
        g = nx.Graph()
        g.add_nodes_from(list(range(1, NVertices + 1)))
        if (BrainGraphs[i].ndim == 1 and len(BrainGraphs[i]) > 0):
            g.add_edge(BrainGraphs[i][0], BrainGraphs[i][1], weight=BrainGraphs[i][2])
        elif (BrainGraphs[i].ndim == 2 and len(BrainGraphs[i]) > 0):
            for k in range(0, BrainGraphs[i].shape[0]):
                g.add_edge(BrainGraphs[i][k, 0], BrainGraphs[i][k, 1], weight=BrainGraphs[i][k, 2])
        BrainGraphsNet.append(g)
        # plt.subplot(1,Timesnap,i+1)
        # plt.title(str(i))
        # plt.show()
        pos = nx.circular_layout(BrainGraphsNet[i])
        nx.draw(BrainGraphsNet[i], pos, node_size=15, edge_color='r')
        labels = nx.get_edge_attributes(BrainGraphsNet[i], 'weight')
        for lab in labels:
            labels[lab] = round(labels[lab], 2)
        # nx.draw_networkx_edge_labels(BrainGraphsNet[i],pos,edge_labels=labels,font_size=5)

        # Building union and computing distance matrix
        BrainGUion = []  # GraphUion List
        MDisBrainGUion = []  # GraphUion distance Matrix List
        for i in range(0, Timesnap - 1):
            UionAux = []
            MDisAux = np.zeros((2 * NVertices, 2 * NVertices))
            A = nx.adjacency_matrix(BrainGraphsNet[i]).todense()
            B = nx.adjacency_matrix(BrainGraphsNet[i + 1]).todense()

            C = (A + B) / 2
            A[A == 0] = 1.1
            A[range(NVertices), range(NVertices)] = 0
            B[B == 0] = 1.1
            B[range(NVertices), range(NVertices)] = 0
            MDisAux[0:NVertices, 0:NVertices] = A
            C[C == 0] = 1.1
            C[range(NVertices), range(NVertices)] = 0
            MDisAux[NVertices:(2 * NVertices), NVertices:(2 * NVertices)] = B
            MDisAux[0:NVertices, NVertices:(2 * NVertices)] = C
            MDisAux[NVertices:(2 * NVertices), 0:NVertices] = C.transpose()

            # distance in condensed form
            pDisAux = squareform(MDisAux)
            # To save the uion and distance
            BrainGUion.append(UionAux)
            MDisBrainGUion.append(pDisAux)

        # To perform the Risper Coputation
        print("Computing Vietoris-Rips complexes...")
        BrainGVRips = []
        for i in range(0, Timesnap - 1):
            print(i)
            ripsAux = d.fill_rips(MDisBrainGUion[i], maxDimHoles, scaleParameter)
            BrainGVRips.append(ripsAux)
        print('  ---Ending the Computation of Vetoris Rips')

        # shifting filtration
        print("Shifting filtrations...")  # Beginning
        BrainGVRips_shift = []
        BrainGVRips_shift.append(BrainGVRips[0])
        for i in range(1, Timesnap - 1):
            shiftAux = zzt.shift_filtration(BrainGVRips[i], NVertices * i)
            BrainGVRips_shift.append(shiftAux)
        print("  --- End shifting...")  # Ending

        # To Combine complexes
        print("Combining complexes...")  # Beginning
        completeBrainGVRips = zzt.complex_union(BrainGVRips[0], BrainGVRips_shift[1])
        for i in range(2, Timesnap - 1):
            completeBrainGVRips = zzt.complex_union(completeBrainGVRips, BrainGVRips_shift[i])
        print("  --- End combining")  # Ending

        # To compute the time intervals of simplices
        print("Determining time intervals...")  # Beginning
        time_intervals = zzt.build_zigzag_times(completeBrainGVRips, NVertices, Timesnap)
        print("  --- End time")  # Beginning

        # to build zigzag persistence
        print("Computing Zigzag homology...")  # Beginning
        G_zz, G_dgms, G_cells = d.zigzag_homology_persistence(completeBrainGVRips, time_intervals)
        print("  --- End Zigzag")  # Beginning

        # persistence interval
        windows_ZPD = []
        birth = []
        death = []
        for v, dgm in enumerate(G_dgms):
            print('Dimension', v)
            if (v < 2):
                matBarcode = np.zeros((len(dgm), 2))
                k = 0
                for p in dgm:
                    matBarcode[k, 0] = p.birth
                    matBarcode[k, 1] = p.death
                    birth.append(matBarcode[k, 0])
                    death.append(matBarcode[k, 1])
                    k = k + 1
                    matBarcode = matBarcode / 2
                    windows_ZPD.append(matBarcode)
        print(birth)
        print(death)
        plt.scatter(birth, death)
        plt.xlabel('birth')
        plt.ylabel('death')
        plt.title('ZPD')
        # plt.show()
        plt.savefig('ZPD.png', bbox_inches="tight")
        plt.close()

        # zigzag persistence image
        resolution = (50, 50)
        bandwith = 1
        power = 1
        dimensional = 1
        bandwidth = 1

        PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in windows_ZPD]), np.vstack([dgm[:, 1:2] for dgm in windows_ZPD])
        xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
        x = np.linspace(xm, xM, resolution[0])
        y = np.linspace(ym, yM, resolution[1])
        X, Y = np.meshgrid(x, y)
        Zfinal = np.zeros(X.shape)
        print(X.shape)
        X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

        # computing the zigzag persistence image
        P0, P1 = np.reshape(windows_ZPD[int(dimensional)][:, 0], [1, 1, -1]), np.reshape(
            windows_ZPD[int(dimensional)][:, 1], [1, 1, -1])
        weight = np.abs(P1 - P0)
        distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

        weight = weight ** power
        Zfinal = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)
        zpi = (Zfinal - np.min(Zfinal)) / (np.max(Zfinal) - np.min(Zfinal))
        return zpi



