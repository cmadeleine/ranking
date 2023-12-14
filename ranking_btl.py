import random
import pandas as pd
import numpy as np
from numpy.linalg import eig
import gurobipy as gp
from gurobipy import GRB
import math
from optspace import OptSpace

# ALGORITHMS BEING TESTED:
# Linear Programming: lp
# Spectral Method/Rank Centrality: spec
# Spectral MLE: mle
# Low Rank Pairwise Ranking: lrpr


# creating synthetic data for P matrix, approximating ideal matrix P*
def make_P(n, e, L, w):

    P_btl = np.zeros((n, n))

    # keep track of nonzero entries for i != j, to speed up algorithm
    P_nonzero = [[] for i in range(n)]

    # fill in probability matrix
    for i in range(0, n):
        for j in range(i, n):
            # diagonals (i beats i) should have probability of 1/2
            if (i==j):
                P_btl[i][j] = 1/2
            # if i and j get compared:
            elif (random.random() < e):
                # generate a number of wins according to each model's respective probability function
                wins = np.random.binomial(L, (w[i] / (w[i] + w[j])))
                # record the probability of wins and the inverse in each matrix ([i][j] and [j][i])
                P_btl[i][j] = wins / L
                P_btl[j][i] = 1 - P_btl[i][j]

                P_nonzero[i].append(j)
                P_nonzero[j].append(i)
            # else: P_ij = 0
            else:
                P_btl[i][j] = 0

    df = pd.DataFrame(data=P_btl)
    df.to_excel('P_btl.xlsx', sheet_name='P_btl')
    return P_btl, P_nonzero         

# creating 1D  w vector: value/weight/score of each element; ranges from 0.5-1
def make_w(n, delta_k):
    w = [0] * n
    if (delta_k > 0):
        w[0] = 0.5
    else:
        w[0] = random.random() * (0.5 - delta_k) + (0.5 + delta_k)
    for i in range(1, n):
        w[i] = random.random() * (0.5 - delta_k) + (0.5 + delta_k)

    # for i in range(0, n):
    #     w[i] = random.random()
    
    df = pd.DataFrame(data=w)
    df.to_excel('w.xlsx', sheet_name='w')

    return w

# approximate w with linear program
def lp_algorithm(P, w_min, w_max):
    # vector approximating w: w_btl

    m_btl = gp.Model("BTL")
    # x: 1D vector (used to approximate s)
    x_btl = m_btl.addMVar((n, 1), lb=math.log(w_min), ub=math.log(w_max), vtype=GRB.CONTINUOUS, name="x")
    # z: nxn matrix, z = Y - P
    z_btl = m_btl.addMVar((n, n), lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    zeroMat = np.zeros((n, n))
    eVec = np.full((n, 1), 1)
    mask = np.zeros((n, n))

    # P with linking function applied
    link_btl = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            # we assume no probabilities are zero unless they were not compared
            if (P[i][j] == 0):
                link_btl[i][j] = 0
            else:
                # 1 / (1 - x) undefined
                if (P[i][j] == 1):
                    link_btl[i][j] = 0
                    continue
                # re run with natural log
                link_btl[i][j] = math.log(P[i][j] / (1 - P[i][j]))
                if (i != j):
                    mask[i][j] = 1

    # constraints
    m_btl.addConstr((x_btl @ eVec.T) - (eVec @ x_btl.T) + z_btl - link_btl >= zeroMat)
    m_btl.addConstr((x_btl @ eVec.T) - (eVec @ x_btl.T) - z_btl - link_btl <= zeroMat)

    # constraints include entire matrix, but we only care when link[i][j] != 0 and i != j; so apply a mask: np.multiply(z, mask)
    # this way, the optimization problem can ignore zero entries
    m_btl.setObjective((z_btl * mask).sum(), GRB.MINIMIZE)

    m_btl.optimize()

    # initialized approximation vector of ideal w
    w_btl = []

    # for btl:  undo log ->   x = log(w) => w = 2^x
    for i in range(0, n):
        w_btl.append(pow(math.e, x_btl.X[i])[0])

    w_btl = w_btl / sum(w_btl)
    return w_btl

# from a base matrix P_btl, find the pi vector estimating w
def spec_algorithm(P_btl):
    # vector approximating w: pi

    P_T = P_btl.T

    # d = maximum row sum
    d = 0
    for i in range(0, n):
        if (sum(P_T[i]) > d):
                d = sum(P_T[i])
        
    # P transition matrix 
    P_transit = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (P_T[i][j] == 0):
                P_transit[i][j] = 0
            else:
                # apply link function
                if (i==j):
                    P_transit[i][j] = 1 - (1/d) * (sum(P_T[i]) - P_T[i][i])
                else:
                    P_transit[i][j] = (1/d) * P_T[i][j]


    for i in range(0, n):
        P_transit[i] = P_transit[i] / sum(P_transit[i])

    # find eigenvector
    # need to transpose since stationary distrib. is pi*P = pi, not P*pi = pi
    evals, evecs = eig(P_transit.T)

    pi = []
    for i in range(0, len(evecs)):
        if (np.isclose(evals[i], 1)):
            pi = evecs[:, i]

    # flip the sign if negative (should this be happening?)
    if (pi[0] < 0):
        for i in range(0, n):
            pi[i] = -1 * pi[i]

    pi = pi / sum(pi)
    return pi

def mle_algorithm(P_btl, P_nonzero, w_min, w_max):
    # vector approximating w: w_t

    w_spec = spec_algorithm(P_btl)

    w_mle = [0] * n

    # checking 100 values of tau between min and max, could increase/decrease for more/less precision
    tau_step = (w_max - w_min) / 100

    w_t = w_spec.copy()

    # number of iterations
    #T = math.floor(5 * math.log(n))
    T = 20

    
    for t in range(0, T):
        # coordinate wise MLE part 1
        for i in range(0, n): 
            tau_best = float('-inf')
            max_P = float('-inf')
            tau = w_min
            # compute probability for each possible value for tau
            while (tau <= w_max):
                curr_P = 0
                for j in P_nonzero[i]:
                    # sum across j : (i, j) in E
                    #if (P_btl[i][j] != 0 and j != i):
                        curr_P += P_btl[i][j] * math.log(tau / (tau + w_t[j])) + (1 - P_btl[i][j]) * math.log(w_t[j]/ (tau + w_t[j]))
                # new best value for tau
                if (curr_P > max_P):
                    max_P = curr_P
                    tau_best = tau
                tau += tau_step
            w_mle[i] = tau_best

        # threshold value for using mle-generated values
        E_min = math.sqrt(math.log(n) / (n * e * L))
        E_max = math.sqrt(math.log(n) / (e * L))
        E_t = (E_min + 1/(math.pow(2, t)) * (E_max - E_min))
        E_t = 0

        # MLE part 2
        # NOTE: make separate version for all updates
        #quit = True
        for i in range(0, n):
            # if difference is large enough, pick new (mle-generated) value
            if (abs(w_mle[i] - w_t[i]) > E_t):
                w_t[i] = w_mle[i]
                #modified = True
                #print("\tmodified: ", i, ", ", w_mle[i])
            #elif (abs(w_mle[i] - w_t[i]) > E_min):
                #quit = False
            else:
                # redundant, but here for clarity
                w_t[i] = w_t[i]
            # if (quit):
            #     pass

        # stop iterating if no values are modified to avoid redundant iterations
        #if (not modified):
        #    break
    return w_t

def lrpr_algorithm(P):
    # vector approximating w: sigma

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
                link[i][j] = math.log(P[i][j] / (1 - P[i][j]))      

    os = OptSpace(2, 5, 0.0001)
    U, S, V = os.solve(link)

    # construct filled-in matrix
    opt_mat = np.matmul(np.matmul(U, S), V.T)

    inv_link = np.zeros((n, n))
    
    # construct matrix according to algorithm
    for i in range(0, n):
        for j in range(0, n):
                inv_ij = math.pow(math.e, opt_mat[i][j]) / (1 + math.pow(math.e, opt_mat[i][j]))
                inv_ji = math.pow(math.e, opt_mat[j][i]) / (1 + math.pow(math.e, opt_mat[j][i]))
                if (i == j):
                    inv_link[i][j] = 1/2
                elif (opt_mat[i][j] > opt_mat[j][i]):                                     
                    inv_link[i][j] = 1/2 + min(abs(inv_ij - 1/2), abs(inv_ji - 1/2))
                else:
                    inv_link[i][j] = 1/2 - min(abs(inv_ij - 1/2), abs(inv_ji - 1/2))

    # copeland ranking: w[i] = number of entries with probability > 0.5
    sigma = [0] * n
    for i in range(0, n):
        for j in range(0, n):
            if (inv_link[i][j] > 0.5):
                sigma[i] += 1

    s_norm = np.asarray(sigma) / sum(sigma)

    return s_norm

# run a trial with n items, 
# where e is the probability that any pair of elements will be compared, 
# and L is the number of comparisons per pair
def simulate(n, L, e, gap):
    w = make_w(n, gap)
    w_norm = np.asarray(w) / sum(w)
    P, P_nonzero = make_P(n, e, L, w)

    #==================Linear Program==================
    # models: btl, thurstone
    print("before lp")
    w_lp = lp_algorithm(P, min(w_norm), max(w_norm))
    # w_lp = [0]*n
    #==================Spectral Method==================
    # models: btl
    print("before spec")
    w_spec = spec_algorithm(P)

    #======================MLE==========================
    # models: btl
    print("before mle")
    w_mle = mle_algorithm(P, P_nonzero, min(w_norm), max(w_norm))
    # w_mle = [0]*n

    #================Low Rank Pairwise Ranking=================
    # models: btl, thu
    print("before lrpr")
    w_lrpr = lrpr_algorithm(P) 
    # w_lrpr = [0]*n
    print("done")

    #==================Comparing Algorithms==================

    w_comp = np.zeros((n, 6))

    # comparing normalized w (ideal), our algorithm (btl), spectral mle (btl), our algorithm (thurstone), spectral MLE (btl)
    for i in range(0, n):
        w_comp[i][0] = i
        w_comp[i][1] = w_norm[i]
        w_comp[i][2] = w_lp[i]
        w_comp[i][3] = w_spec[i]
        w_comp[i][4] = w_mle[i]
        w_comp[i][5] = w_lrpr[i]

    # sort each ranking
    w_sort = np.argsort(w)
    lp_sort = np.argsort(w_lp)
    spec_sort = np.argsort(w_spec)
    mle_sort = np.argsort(w_mle)
    lrpr_sort = np.argsort(w_lrpr)

    rankings_comp = np.zeros((n, 5))

    for i in range(0, n):
        rankings_comp[i][0] = w_sort[i]
        rankings_comp[i][1] = lp_sort[i]
        rankings_comp[i][2] = spec_sort[i]
        rankings_comp[i][3] = mle_sort[i]
        rankings_comp[i][4] = lrpr_sort[i]

    df = pd.DataFrame(data = rankings_comp)
    df.to_excel('rankings_comp.xlsx', sheet_name='rankings_comp')

    l_inf_err = [0]*4
    D_w_err = [0]*4

    # l infinity error
    
    l_inf_err[0] = max(abs(np.subtract(w_lp, w_norm)))/max(w_norm)
    l_inf_err[1] = max(abs(np.subtract(w_spec, w_norm)))/max(w_norm)
    l_inf_err[2] = max(abs(np.subtract(w_mle, w_norm)))/max(w_norm)
    l_inf_err[3] = max(abs(np.subtract(w_lrpr, w_norm)))/max(w_norm)

    # D_w error
    for i in range(0, n):
        for j in range(i, n):
            if ((w_norm[i] - w_norm[j]) * (w_lp[i] - w_lp[j]) < 0):
                D_w_err[0] += math.pow((w_norm[i] - w_norm[j]), 2)
            if ((w_norm[i] - w_norm[j]) * (w_spec[i] - w_spec[j]) < 0):
                D_w_err[1] += math.pow((w_norm[i] - w_norm[j]), 2)
            if ((w_norm[i] - w_norm[j]) * (w_mle[i] - w_mle[j]) < 0):
                D_w_err[2] += math.pow((w_norm[i] - w_norm[j]), 2)
            if ((w_norm[i] - w_norm[j]) * (w_lrpr[i] - w_lrpr[j]) < 0):
                D_w_err[3] += math.pow((w_norm[i] - w_norm[j]), 2)

    D_w_err[0] = math.sqrt(D_w_err[0] / (2*n*math.pow(np.linalg.norm(w_norm), 2)))
    D_w_err[1] = math.sqrt(D_w_err[1] / (2*n*math.pow(np.linalg.norm(w_norm), 2)))
    D_w_err[2] = math.sqrt(D_w_err[2] / (2*n*math.pow(np.linalg.norm(w_norm), 2)))
    D_w_err[3] = math.sqrt(D_w_err[3] / (2*n*math.pow(np.linalg.norm(w_norm), 2)))

    # alternative D_w error according to LRPR paper
    # for i in range(0, n):
    #     for j in range(i, n):
    #         if ((w_norm[i] - w_norm[j]) * (w_lp[i] - w_lp[j]) < 0):
    #             D_w_err[0] += 1
    #         if ((w_norm[i] - w_norm[j]) * (w_spec[i] - w_spec[j]) < 0):
    #             D_w_err[1] += 1
    #         if ((w_norm[i] - w_norm[j]) * (w_mle[i] - w_mle[j]) < 0):
    #             D_w_err[2] += 1
    #         if ((w_norm[i] - w_norm[j]) * (w_lrpr[i] - w_lrpr[j]) < 0):
    #             D_w_err[3] += 1

    # D_w_err[0] = D_w_err[0] / (n * (n-1) / 2)
    # D_w_err[1] = D_w_err[1] / (n * (n-1) / 2)
    # D_w_err[2] = D_w_err[2] / (n * (n-1) / 2)
    # D_w_err[3] = D_w_err[3] / (n * (n-1) / 2)

    return rankings_comp, l_inf_err, D_w_err

# running an instance of a simulation, and recording the results in csv files
def run_trial(n, L, e, gap, rank, err):

    rankings, l_inf_err, D_w_err = simulate(n, L, e, gap)

    str_lp = "\"btl\",\"lp\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","
    str_spec = "\"btl\",\"spec\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","
    str_mle = "\"btl\",\"mle\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","
    str_lrpr = "\"btl\",\"lrpr\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","

    err.write("\"btl\",\"lp\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(l_inf_err[0]) + "," + str(D_w_err[0]) + "\n")
    err.write("\"btl\",\"spec\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(l_inf_err[1]) + "," + str(D_w_err[1]) + "\n")
    err.write("\"btl\",\"mle\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(l_inf_err[2]) + "," + str(D_w_err[2]) + "\n")
    err.write("\"btl\",\"lrpr\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(l_inf_err[3]) + "," + str(D_w_err[3]) + "\n")

    # objective ranking (w), vs each of our algorithms
    rank_true = set()
    rank_lp = set()
    rank_spec = set()
    rank_mle = set()
    rank_lrpr = set()

    # record up to top 20 ranks
    for k in range(0, 20):
        rank_true.add(rankings[k][0])
        rank_lp.add(rankings[k][1])
        rank_spec.add(rankings[k][2])
        rank_mle.add(rankings[k][3])
        rank_lrpr.add(rankings[k][4])

        lp_correct = len(rank_true.intersection(rank_lp))
        spec_correct = len(rank_true.intersection(rank_spec))
        mle_correct = len(rank_true.intersection(rank_mle))
        lrpr_correct = len(rank_true.intersection(rank_lrpr))

        # record when it guesses the set of top k items correctly
        lp_correct = lp_correct == k + 1
        spec_correct = spec_correct == k + 1
        mle_correct = mle_correct == k + 1
        lrpr_correct = lrpr_correct == k + 1

        str_lp += (str(round(lp_correct, 3)))
        str_spec += (str(round(spec_correct, 3)))
        str_mle += (str(round(mle_correct, 3)))
        str_lrpr += (str(round(lrpr_correct, 3)))
    
        if (k==19):
            str_lp += ("\n")
            str_spec += ("\n")
            str_mle += ("\n")
            str_lrpr += ("\n")
        else:
            str_lp += (",")
            str_spec += (",")
            str_mle += (",")
            str_lrpr += (",")
    
    rank.write(str_lp)
    rank.write(str_spec)
    rank.write(str_mle)
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

model = 'btl'

# if appending trials to an existing file, comment out init_csv and change "w"s to "a"
ranks = open(("rank_" + model + "fig2_2.csv"), "w")
err = open(("error_" + model + "fig2_2.csv"), "w")
init_csv(ranks, err)

# n: number of items
# L: number of comparisons
# e probability of 2 items being compared
# gap: imposed gap in w score between first element and all others
# j: trials per data point

for n in [500]:
    for L in [30]:   
        for e in [5*math.log(n)/n, 10*math.log(n)/n, 15*math.log(n)/n, 20*math.log(n)/n]: 
            gap = 0.0
            for j in range(0, 20):
                    print(n, L, e, gap, j)
                    run_trial(n, L, e, gap, ranks, err)


ranks.close()
err.close()