import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
import zigzagtools as zzt
import dionysus as d
import time


def ZPI_generator(BrainGraphs, Window, NVertices, scaleParameter, maxDimHoles,  index,j):
    BrainGraphsNet = []
    j+=1
    #plt.figure(num=None, figsize=(16, 1.5), dpi=80, facecolor='w', edgecolor='k')
    BrainGraphs_Num = len(BrainGraphs)
    for i in range(0, BrainGraphs_Num):
        print('Graph' + str(i))
        g = nx.Graph()
        g.add_nodes_from(list(range(1, NVertices + 1)))
        if BrainGraphs[i].ndim == 1 and len(BrainGraphs[i]) > 0:
            g.add_edge(BrainGraphs[i][0], BrainGraphs[i][1], weight=BrainGraphs[i][2])
        elif BrainGraphs[i].ndim == 2 and len(BrainGraphs[i]) > 0:
            for k in range(0, BrainGraphs[i].shape[0]):
                g.add_edge(BrainGraphs[i][k, 0], BrainGraphs[i][k, 1], weight=BrainGraphs[i][k, 2])
        BrainGraphsNet.append(g)

        pos = nx.circular_layout(BrainGraphsNet[i])
        nx.draw(BrainGraphsNet[i], pos, node_size=15, edge_color='r')
        labels = nx.get_edge_attributes(BrainGraphsNet[i], 'weight')
        for lab in labels:
            labels[lab] = round(labels[lab], 2)

    # Building union and computing distance matrix
    BrainGUnion = []  # GraphUnion List
    MDisBrainGUnion = []  # GraphUnion distance Matrix List
    for i in range(0, BrainGraphs_Num - 1):
        UnionAux = []
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
        # To save the Union and distance
        BrainGUnion.append(UnionAux)
        MDisBrainGUnion.append(pDisAux)

    # To perform the Riser Computation
    print("Computing Vietoris-Rips complexes...")
    BrainGVRips = []
    for i in range(0, BrainGraphs_Num - 1):
        print(i)
        ripsAux = d.fill_rips(MDisBrainGUnion[i], maxDimHoles, scaleParameter)
        BrainGVRips.append(ripsAux)
    print('  ---Ending the Computation of Vietoris Rips')

    # shifting filtration
    print("Shifting filtration...")  # Beginning
    BrainGVRips_shift = [BrainGVRips[0]]
    for i in range(1, BrainGraphs_Num - 1):
        shiftAux = zzt.shift_filtration(BrainGVRips[i], NVertices * i)
        BrainGVRips_shift.append(shiftAux)
    print("  --- End shifting...")  # Ending

    # To Combine complexes
    print("Combining complexes...")  # Beginning
    completeBrainGVRips = zzt.complex_union(BrainGVRips[0], BrainGVRips_shift[1])
    for i in range(2, BrainGraphs_Num - 1):
        completeBrainGVRips = zzt.complex_union(completeBrainGVRips, BrainGVRips_shift[i])
    print("  --- End combining")  # Ending

    # To compute the time intervals of simplices
    print("Determining time intervals...")  # Beginning
    time_intervals = zzt.build_zigzag_times(completeBrainGVRips, NVertices, Window)
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
        if v < 2:
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

    # zigzag persistence image
    resolution = (100, 100)
    power = 1
    dimensional = 1
    bandwidth = 1

    PXs, PYs = np.vstack([dgm[:, 0:1] for dgm in windows_ZPD]), np.vstack([dgm[:, 1:2] for dgm in windows_ZPD])
    xm, xM, ym, yM = PXs.min(), PXs.max(), PYs.min(), PYs.max()
    x = np.linspace(xm, xM, resolution[0])
    y = np.linspace(ym, yM, resolution[1])
    X, Y = np.meshgrid(x, y)
    Z_final = np.zeros(X.shape)
    print(X.shape)
    X, Y = X[:, :, np.newaxis], Y[:, :, np.newaxis]

    # computing the zigzag persistence image
    P0 = np.reshape(windows_ZPD[int(dimensional)][:, 0], [1, 1, -1])
    P1 = np.reshape(windows_ZPD[int(dimensional)][:, 1], [1, 1, -1])
    weight = np.abs(P1 - P0)
    distpts = np.sqrt((X - P0) ** 2 + (Y - P1) ** 2)

    weight = weight ** power
    Z_final = (np.multiply(weight, np.exp(-distpts ** 2 / bandwidth))).sum(axis=2)
    zpi = (Z_final - np.min(Z_final)) / (np.max(Z_final) - np.min(Z_final))
    #np.save(ZPI_location + "{}.npy".format(index), zpi)
    fig = plt.figure()
    plt.imshow(zpi, interpolation='nearest', origin='lower', cmap='jet')
    plt.colorbar()
    plt.savefig('ZPI-{}.png'.format(str(j)))
    plt.clf()

    return zpi



if __name__ == '__main__':

    NVertices = 62
    scaleParameter = 1
    maxDimHoles = 2
    index = 0
    Window = 3
    stride = 3
    path = os.getcwd()


    paths = os.getcwd()#  F:\ResearchData\Demo\adjacent_matrix(2)\pm0.3
    # Load data
    print('Loading data...')
    first_list = os.listdir(paths)
    j=0
    ZPI = []

    for first_dir in first_list:
        second_list = os.listdir(paths + '/' + first_dir)
        for second_dir in second_list:
            print(second_dir)

            third_list = os.listdir(paths + '/' + first_dir + '/' + second_dir)
            print(third_list)
            BrainGraphs = []
            for third_dir in third_list:
                path = os.path.join(paths + '/' + first_dir + '/' + second_dir + '/' + third_dir)
                edge_list = np.loadtxt(path, delimiter=',')
                BrainGraphs.append(edge_list)
                print('BrainGraphs:',len(BrainGraphs))

            print('----Complete the BrainGraph  Into the zpi of Window BrainGraph------')

            for i in range(0, len(BrainGraphs), stride):

                BrainGraphs_Window = BrainGraphs[i:i + Window]
                print(len(BrainGraphs_Window))

                if len(BrainGraphs) < Window:
                    BrainGraphs_Window += [BrainGraphs[0]] * (Window - len(BrainGraphs_Window))
                print('BrainGraphs_Window:', len(BrainGraphs_Window))
                start = time.time()
                zpi = ZPI_generator(BrainGraphs_Window, Window, NVertices, scaleParameter, maxDimHoles, index, j)
                print('time',time.time() - start)
                ZPI.append(zpi)
















            #if os.path.isdir(path+'/'+first_dir):
            #    pass
            #else:
            #    os.makedirs(path+'/'+first_dir)
            #if os.path.isdir(path+'/' + first_dir+'/'+second_dir):
            #    pass
            #else:
            #    os.makedirs(path+'/'+first_dir+'/'+second_dir)
            #np.save(path+'/'+first_dir+'/'+second_dir+'/'+'ZPI_npy.npy',ZPI_np)









