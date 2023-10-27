import random
import pandas as pd
import numpy as np
from numpy.linalg import eig
import gurobipy as gp
from gurobipy import GRB
import math
import csv
from scipy.stats import norm
from optspace import OptSpace

# ALGORITHMS BEING TESTED:
# Linear Programming: lp
# Spectral Method/Rank Centrality: spec
# Spectral MLE: mle
# Low Rank Pairwise Ranking: lrpr


def logit(p):
    return math.log2(p / (1 - p))

def logit_inv(p):
    return math.pow(2, p) / (1 + math.pow(2, p))

# creating synthetic data for P matrix in BTL and Thurstone models, approximating ideal matrix P*
def make_P(n, e, L, w):

    P_thu = np.zeros((n, n))
    # fill in probability matrix
    for i in range(0, n):
        for j in range(i, n):
            # diagonals (i beats i) should have probability of 1/2
            if (i==j):
                P_thu[i][j] = 1/2
            # if i and j get compared:
            elif (random.random() < e):
                # generate a number of wins according to each model's respective probability function
                wins = np.random.binomial(L,  norm.cdf(w[i] - w[j]))
                # record the probability of wins and the inverse in each matrix ([i][j] and [j][i])
                P_thu[i][j] = wins / L
                P_thu[j][i] = 1 - P_thu[i][j]
            # else: P_ij = 0
            else:
                P_thu[i][j] = 0

    df = pd.DataFrame(data=P_thu)
    df.to_excel('P_thu.xlsx', sheet_name='P_thu')
    return P_thu

# creating 1D  w vector: value/weight/score of each element
def make_w(n, delta_k):
    w = [0] * n
    w[0] = 0.5
    for i in range(1, n):
        w[i] = random.random() * (0.5 - delta_k) + (0.5 + delta_k)


    #df = pd.DataFrame(data=w)
    #df.to_excel('w_vector.xlsx', sheet_name='w_vector')
    return w

# approximate w with linear program
def lp_algorithm(P):

    m_thu = gp.Model("Thurstone")
    # x: 1D vector (used to approximate s)
    x_thu = m_thu.addMVar((n, 1), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    # z: nxn matrix, z = Y - P
    z_thu = m_thu.addMVar((n, n), lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    zeroMat = np.zeros((n, n))
    eVec = np.full((n, 1), 1)
    mask = np.zeros((n, n))

    # P with linking function applied
    link_thu = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            # we assume no probabilities are zero, if btl is zero, so is thu
            if (P[i][j] == 0):
                link_thu[i][j] = 0
            else:
                # 1 / (1 - x) undefined
                if (P[i][j] == 1):
                    link_thu[i][j] = 0
                    continue
                link_thu[i][j] = norm.ppf(P[i][j])
                if (i != j):
                    mask[i][j] = 1
    
    # constraints
    m_thu.addConstr((x_thu @ eVec.T) - (eVec @ x_thu.T) + z_thu - link_thu >= zeroMat)
    m_thu.addConstr((x_thu @ eVec.T) - (eVec @ x_thu.T) - z_thu - link_thu <= zeroMat)

    # constraints include entire matrix, but we only care when link[i][j] != 0 and i != j; so apply a mask: np.multiply(z, mask)
    m_thu.setObjective((z_thu * mask).sum(), GRB.MINIMIZE)

    m_thu.optimize()

    # initialized approximation vector of ideal w
    w_thu = []

    # for thurstone: read in values directly
    for i in range(0, n):
        w_thu.append(x_thu.X[i][0])

    w_thu = w_thu / sum(w_thu)
    return w_thu


def lrpr_algorithm(P):

    # P with linking function applied
    link = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (P[i][j] == 0):
                link[i][j] = 0
            else:
                # 1 / (1 - x) undefined
                if (P[i][j] == 1):
                    link[i][j] = 0
                    continue
                link[i][j] = norm.ppf(P[i][j])          

    os = OptSpace(2, 5, 0.0001)
    U, S, V = os.solve(link)

    opt_mat = np.matmul(np.matmul(U, S), V.T)

    df = pd.DataFrame(data = opt_mat)
    df.to_excel('opt_mat.xlsx', sheet_name='opt_mat')

    inv_link = np.zeros((n, n))
    
    for i in range(0, n):
        for j in range(0, n):
                inv_ij = norm.cdf(opt_mat[i][j])   
                inv_ji = norm.cdf(opt_mat[j][i]) 
                if (i == j):
                    inv_link[i][j] = 1/2
                elif (opt_mat[i][j] > opt_mat[j][i]):                                     
                    inv_link[i][j] = 1/2 + min(abs(inv_ij - 1/2), abs(inv_ji - 1/2))
                else:
                    inv_link[i][j] = 1/2 - min(abs(inv_ij - 1/2), abs(inv_ji - 1/2))

    sigma = [0] * n

    for i in range(0, n):
        for j in range(0, n):
            if (inv_link[i][j] > 0.5):
                sigma[i] += 1

    df = pd.DataFrame(data = sigma)
    df.to_excel('sigma.xlsx', sheet_name='sigma')

    s_norm = np.asarray(sigma) / sum(sigma)

    return s_norm


# run a trial with n items, 
# where e is the probability that any pair of elements will be compared, 
# and L is the number of comparisons per pair
def simulate(n, L, e, gap):
    w = make_w(n, gap)
    w_norm = np.asarray(w) / sum(w)
    P = make_P(n, e, L, w)

    #==================Linear Program==================
    # models: btl, thurstone
    # approximating w vector with: w_btl, w_thu
    print("before lp")
    w_lp = lp_algorithm(P)
    #================Low Rank Pairwise Ranking=================
    # models: btl, thu
    # approximating w vector with: w_lrpr
    print("before lrpr")
    w_lrpr = lrpr_algorithm(P) 
    print("done")

    #==================Comparing Algorithms==================

    w_comp = np.zeros((n, 4))

    # comparing normalized w (ideal), our algorithm (btl), spectral mle (btl), our algorithm (thurstone), spectral MLE (btl)
    for i in range(0, n):
        w_comp[i][0] = i
        w_comp[i][1] = w_norm[i]
        w_comp[i][2] = w_lp[i]
        w_comp[i][3] = w_lrpr[i]

    w_sort = np.argsort(w)
    lp_sort = np.argsort(w_lp)
    lrpr_sort = np.argsort(w_lrpr)

    rankings_comp = np.zeros((n, 3))

    for i in range(0, n):
        rankings_comp[i][0] = w_sort[i]
        rankings_comp[i][1] = lp_sort[i]
        rankings_comp[i][2] = lrpr_sort[i]

    df = pd.DataFrame(data = rankings_comp)
    df.to_excel('rankings_comp.xlsx', sheet_name='rankings_comp')

    l_inf_err = [0]*2
    D_w_err = [0]*2

    # entrywise error
    l_inf_err[0] = max(abs(np.subtract(w_lp, w_norm)))/max(w_norm)
    l_inf_err[1] = max(abs(np.subtract(w_lrpr, w_norm)))/max(w_norm)

    # D_w error
    for i in range(0, n):
        for j in range(i, n):
            if ((w_norm[i] - w_norm[j]) * (w_lp[i] - w_lp[j]) > 0):
                D_w_err[0] += math.pow((w_norm[i] - w_norm[j]), 2)
            if ((w_norm[i] - w_norm[j]) * (w_lrpr[i] - w_lrpr[j]) > 0):
                D_w_err[1] += math.pow((w_norm[i] - w_norm[j]), 2)

    D_w_err[0] = math.sqrt(D_w_err[0] / (2*n*math.pow(np.linalg.norm(w), 2)))
    D_w_err[1] = math.sqrt(D_w_err[1] / (2*n*math.pow(np.linalg.norm(w), 2)))

    df = pd.DataFrame(data = l_inf_err)
    df.to_excel('l_inf_err.xlsx', sheet_name='l_inf_err')

    return rankings_comp, l_inf_err, D_w_err
    


# running an instance of a simulation, and recording the results in csv files
def run_trial(n, L, e, gap, rank, err):

    rankings, l_inf_err, D_w_err = simulate(n, L, e, gap)
    # print("rankings: ", rankings)
    # print("errors: ", errors)

    str_lp = "\"thu\",\"lp\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","
    str_lrpr = "\"thu\",\"lrpr\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","

    err.write("\"thu\",\"lp\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(l_inf_err[0]) + "," + str(D_w_err[0]) + "\n")
    err.write("\"thu\",\"lrpr\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(l_inf_err[1]) + "," + str(D_w_err[1]) + "\n")

    # objective ranking (w), vs each of our algorithms
    rank_true = set()
    rank_lp = set()
    rank_lrpr = set()

    for k in range(0, 20):
        rank_true.add(rankings[k][0])
        rank_lp.add(rankings[k][1])
        rank_lrpr.add(rankings[k][2])

        lp_correct = len(rank_true.intersection(rank_lp))
        lrpr_correct = len(rank_true.intersection(rank_lrpr))

        lp_correct = lp_correct == k + 1
        lrpr_correct = lrpr_correct == k + 1

        str_lp += (str(round(lp_correct, 3)))
        str_lrpr += (str(round(lrpr_correct, 3)))
    
        if (k==19):
            str_lp += ("\n")
            str_lrpr += ("\n")
        else:
            str_lp += (",")
            str_lrpr += (",")
    
    rank.write(str_lp)
    rank.write(str_lrpr)
  

def init_csv(main, error) :
    main.write("\"model\",")
    error.write("\"model\",")
    main.write("\"algorithm\",")
    error.write("\"algorithm\",")
    main.write("\"items\",")
    error.write("\"items\",")
    main.write("\"epsilon\",")
    error.write("\"epsilon\",")
    main.write("\"comparisons\",")
    error.write("\"comparisons\",")
    main.write("\"gap\",")
    error.write("\"gap\",")

    error.write("\"l_inf_error\",")
    error.write("\"D_w_error\"\n")

    for i in range(0, 20):
        main.write("\"top_" + str(i + 1) + "\"")
        if (i==19):
            main.write("\n")
        else:
            main.write(",")

model = 'thu'

ranks = open(("rank_" + model + "2.csv"), "a")
err = open(("error_" + model + "2.csv"), "a")
init_csv(ranks, err)


for n in [500]:
    #L = n*n*50
    for L in [30]:  
        e = 10 * math.log(n) / n
        gap = 0.0
        for j in range(0, 20):
                print(n, L, j)
                run_trial(n, L, e, gap, ranks, err)


ranks.close()
err.close()

