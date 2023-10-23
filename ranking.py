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

    P_btl = np.zeros((n, n))
    P_thu = np.zeros((n, n))

    E_set = set()

    # fill in probability matrix
    for i in range(0, n):
        for j in range(i, n):
            # diagonals (i beats i) should have probability of 1/2
            if (i==j):
                P_btl[i][j] = 1/2
                P_thu[i][j] = 1/2
            # if i and j get compared:
            elif (random.random() < e):
                # generate a number of wins according to each model's respective probability function
                wins_btl = np.random.binomial(L, (w[i] / (w[i] + w[j])))
                wins_thu = np.random.binomial(L,  norm.cdf(w[i] - w[j]))
                # record the probability of wins and the inverse in each matrix ([i][j] and [j][i])
                P_btl[i][j] = wins_btl / L
                P_btl[j][i] = 1 - P_btl[i][j]
                P_thu[i][j] = wins_thu / L
                P_thu[j][i] = 1 - P_thu[i][j]
                E_set.add((i, j))
            # else: P_ij = 0
            else:
                P_btl[i][j] = 0
                P_thu[i][j] = 0

    df = pd.DataFrame(data=P_btl)
    df.to_excel('P_btl.xlsx', sheet_name='P_btl')

    # #======================MLE==========================
    # E_size = len(E_set)
    # E_half = random.sample(E_set, (E_size // 2))

    # # partition of edge set into init and iter
    E_init = np.zeros((n, n))
    E_iter = np.zeros((n, n))

    # # fill in E_init and E_iter matrices for MLE
    # for i in range(0, n):
    #     for j in range(i, n):
    #         if (i==j):
    #             E_init[i][j] = 1/2
    #             E_iter[i][j] = 1/2
    #         elif (P_btl[i][j] != 0):
    #             if ((i, j) in E_half):
    #                 E_init[i][j] = P_btl[i][j]
    #                 E_init[j][i] = P_btl[j][i]
    #             else:
    #                 E_iter[i][j] = P_btl[i][j]
    #                 E_iter[j][i] = P_btl[j][i]

    # df = pd.DataFrame(data=E_init)
    # df.to_excel('E_init.xlsx', sheet_name='E_init')
    # df = pd.DataFrame(data=E_iter)
    # df.to_excel('E_iter.xlsx', sheet_name='E_iter')

    return P_btl, P_thu, E_init, E_iter

# creating 1D  w vector: value/weight/score of each element
def make_w(n, delta_k):
    w = [0] * n
    w[0] = 0.5
    for i in range(1, n):
        # FIX THIS!!!!!
        w[i] = random.random() * (0.5 - delta_k) + (0.5 + delta_k)


    #df = pd.DataFrame(data=w)
    #df.to_excel('w_vector.xlsx', sheet_name='w_vector')
    return w

# approximate w with linear program
def lp_algorithm(P_btl, P_thu):

    m_btl = gp.Model("BTL")
    m_thu = gp.Model("Thurstone")
    # x: 1D vector (used to approximate s)
    x_btl = m_btl.addMVar((n, 1), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    x_thu = m_thu.addMVar((n, 1), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
    # z: nxn matrix, z = Y - P
    z_btl = m_btl.addMVar((n, n), lb=0.0, vtype=GRB.CONTINUOUS, name="z")
    z_thu = m_thu.addMVar((n, n), lb=0.0, vtype=GRB.CONTINUOUS, name="z")

    # minimize the size of z => || Y - P ||
    # m_btl.setObjective(z_btl.sum(), GRB.MINIMIZE)
    # m_thu.setObjective(z_thu.sum(), GRB.MINIMIZE)

    zeroMat = np.zeros((n, n))
    eVec = np.full((n, 1), 1)
    mask = np.zeros((n, n))

    # P with linking function applied
    link_btl = np.zeros((n, n))
    link_thu = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            # we assume no probabilities are zero, if btl is zero, so is thu
            if (P_btl[i][j] == 0):
                link_btl[i][j] = 0
                link_thu[i][j] = 0
            else:
                # 1 / (1 - x) undefined
                if (P_btl[i][j] == 1):
                    link_btl[i][j] = 0
                    continue
                link_btl[i][j] = math.log2(P_btl[i][j] / (1 - P_btl[i][j]))
                link_thu[i][j] = norm.ppf(P_thu[i][j])
                if (i != j):
                    mask[i][j] = 1

    # constraints
    m_btl.addConstr((x_btl @ eVec.T) - (eVec @ x_btl.T) + z_btl - link_btl >= zeroMat)
    m_btl.addConstr((x_btl @ eVec.T) - (eVec @ x_btl.T) - z_btl - link_btl <= zeroMat)
    m_thu.addConstr((x_thu @ eVec.T) - (eVec @ x_thu.T) + z_thu - link_thu >= zeroMat)
    m_thu.addConstr((x_thu @ eVec.T) - (eVec @ x_thu.T) - z_thu - link_thu <= zeroMat)

    # constraints include entire matrix, but we only care when link[i][j] != 0 and i != j; so apply a mask: np.multiply(z, mask)
    m_btl.setObjective((z_btl * mask).sum(), GRB.MINIMIZE)
    m_thu.setObjective((z_thu * mask).sum(), GRB.MINIMIZE)

    m_btl.optimize()
    m_thu.optimize()

    # initialized approximation vector of ideal w
    w_btl = []
    w_thu = []

    # for thurstone: read in values directly
    # for btl:  undo log ->   x = log(w) => w = 2^x
    for i in range(0, n):
        w_btl.append(pow(2, x_btl.X[i])[0])
        w_thu.append(x_thu.X[i][0])

    w_btl = w_btl / sum(w_btl)
    w_thu = w_thu / sum(w_thu)
    return w_btl, w_thu

# from a base matrix P_btl, find the pi vector estimating w
def spec_algorithm(P_btl):
    # without MLE
    P_T = P_btl.T

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
                if (i==j):
                    P_transit[i][j] = 1 - (1/d) * sum(P_T[i])
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

    if (pi[0] < 0):
        for i in range(0, n):
            pi[i] = -1 * pi[i]

    # QUESTION: transition matrix gets normalized, so all vectors going forth get normalized. should mle use normalized w or raw values?
    pi = pi / sum(pi)
    # print("pi2: ", pi)
    return pi

def mle_algorithm(E_init, E_iter, w_min, w_max):

    w_spec = spec_algorithm(E_init)
    # print('before part 1: ', w_spec)
    w_mle = [0] * n
    tau_step = (w_max - w_min) / 100

    # print("tau_step: ", tau_step)

    # print("w_min, w_max: ", w_min, w_max)
    w_t = w_spec.copy()

    T = math.floor(5 * math.log2(n))
    # print("T: ", T)

    for t in range(0, T):
        # coordinate wise MLE part 1
        for i in range(0, n): 
            tau_best = float('-inf')
            max_P = float('-inf')
            tau = w_min
            while (tau <= w_max):
                curr_P = 0
                for j in range(0, n):
                    # sum across j : (i, j) in E
                    # QUESTION: should we exclude j == i, should we use E_iter
                    if (E_iter[i][j] != 0 and j != i):
                        curr_P += E_iter[i][j] * math.log2(tau / (tau + w_spec[j])) + (1 - E_iter[i][j]) * math.log2(w_spec[j]/ (tau + w_spec[j]))

                if (curr_P > max_P):
                    max_P = curr_P
                    tau_best = tau
                tau += tau_step
            w_mle[i] = tau_best
            # print("tau_best: ", tau_best)
            # break

        # print('before part 2: ', w_mle)

        
        E_min = math.sqrt(math.log2(n) / (n * e * L))
        E_max = math.sqrt(math.log2(n) / (e * L))
        E_t = (E_min + 1/(math.pow(2, t)) * (E_max - E_min)) / 10

        # print("\tE_t: ", E_t)

        # print("mle before part 2:")
        # print(w_mle)
        # MLE part 2
        modified = False
        for i in range(0, n):
            # print("diff: ", abs(w_mle[i] - w_t[i]))
            if (abs(w_mle[i] - w_t[i]) > E_t):
                w_t[i] = w_mle[i]
                modified = True
                print("\tmodified: ", i, ", ", w_mle[i])
            else:
                # redundant, but here for clarity
                w_t[i] = w_t[i]
        if (not modified):
            break
        # print('before part 3: ', w_t)
    return w_t

def lrpr_algorithm(P_btl):

    # P with linking function applied
    link_btl = np.zeros((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if (P_btl[i][j] == 0):
                link_btl[i][j] = 0
            else:
                # 1 / (1 - x) undefined
                if (P_btl[i][j] == 1):
                    link_btl[i][j] = 0
                    continue
                link_btl[i][j] = math.log2(P_btl[i][j] / (1 - P_btl[i][j]))

    df = pd.DataFrame(data = link_btl)
    df.to_excel('link_btl.xlsx', sheet_name='link_btl')

    os = OptSpace(2, 5, 0.0001)
    U, S, V = os.solve(link_btl)

    opt_mat = np.matmul(np.matmul(U, S), V.T)

    df = pd.DataFrame(data = U)
    df.to_excel('U.xlsx', sheet_name='U')
    df = pd.DataFrame(data = S)
    df.to_excel('S.xlsx', sheet_name='S')
    df = pd.DataFrame(data = V)
    df.to_excel('V.xlsx', sheet_name='V')

    df = pd.DataFrame(data = opt_mat)
    df.to_excel('opt_mat.xlsx', sheet_name='opt_mat')

    # print("U: ", U, "\nS: ", S, "\nV: ", V)

    inv_link_btl = np.zeros((n, n))
    
    for i in range (0, n):
        for j in range(0, n):
            # added this because as is, optspace changes all matrix values (not just fills in missing)
            # if (P_btl[i][j] == 0):
                inv_ij = math.pow(2, opt_mat[i][j]) / (1 + math.pow(2, opt_mat[i][j]))
                inv_ji = math.pow(2, opt_mat[j][i]) / (1 + math.pow(2, opt_mat[j][i]))  
                if (i == j):
                    inv_link_btl[i][j] = 1/2
                elif (opt_mat[i][j] > opt_mat[j][i]):                                     
                    inv_link_btl[i][j] = 1/2 + min(abs(inv_ij - 1/2), abs(inv_ji - 1/2))
                else:
                    inv_link_btl[i][j] = 1/2 - min(abs(inv_ij - 1/2), abs(inv_ji - 1/2))
            # else:
            #     inv_link_btl[i][j] = P_btl[i][j]

    df = pd.DataFrame(data = inv_link_btl)
    df.to_excel('inv_link_btl.xlsx', sheet_name='inv_link_btl')

    # sigma = spec_algorithm(inv_link_btl)

    sigma = [0] * n

    for i in range (0, n):
        for j in range (0, n):
            if (inv_link_btl[i][j] > 0.5):
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
    P_btl, P_thu, E_init, E_iter = make_P(n, e, L, w)

    df = pd.DataFrame(data = P_btl)
    df.to_excel('P_btl.xlsx', sheet_name='P_btl')

    # print("w_norm: ", w_norm)
#==================Linear Program==================
# models: btl, thurstone
# approximating w vector with: w_btl, w_thu

    print("before lp")
    w_btl, w_thu = lp_algorithm(P_btl, P_thu)

    # w_btl = [0]*n
    # w_thu = [0]*n

#==================Spectral Method==================
# models: btl
# approximating w vector with: pi
    print("before spec")
    pi = spec_algorithm(P_btl)

    df = pd.DataFrame(data = pi)
    df.to_excel('pi.xlsx', sheet_name='pi')

    # print("pi: ", pi)

#======================MLE==========================
# models: btl
# approximating w vector with: w_mle
    print("before mle")
    w_mle = mle_algorithm(P_btl, P_btl, min(w_norm), max(w_norm))
    # w_mle = [0]*n

    df = pd.DataFrame(data = w_mle)
    df.to_excel('w_mle.xlsx', sheet_name='w_mle')
    # print("w_mle: ", w_mle)

#================Low Rank Pairwise Ranking=================
# models: btl, thu
# approximating w vector with: w_lrpr
    # apply a link function to matrix, probit and logit same as link functions applied above
    # run a matrix completion algorithm -
    print("before lrpr")
    w_lrpr = lrpr_algorithm(P_btl) 
    print("done")
    
    # w_lrpr = [0] * n
    

#==================Comparing Algorithms==================

    w_comp = np.zeros((n, 7))

    # comparing normalized w (ideal), our algorithm (btl), spectral mle (btl), our algorithm (thurstone), spectral MLE (btl)
    for i in range(0, n):
        w_comp[i][0] = i
        w_comp[i][1] = w_norm[i]
        w_comp[i][2] = w_btl[i]
        w_comp[i][3] = w_thu[i]
        w_comp[i][4] = pi[i]
        w_comp[i][5] = w_mle[i]
        w_comp[i][6] = w_lrpr[i]

    w_sort = np.argsort(w)
    w_btl_sort = np.argsort(w_btl)
    w_thu_sort = np.argsort(w_thu)
    pi_sort = np.argsort(pi)
    mle_sort = np.argsort(w_mle)
    lrpr_sort = np.argsort(w_lrpr)

    rankings_comp = np.zeros((n, 6))

    for i in range(0, n):
        rankings_comp[i][0] = w_sort[i]
        rankings_comp[i][1] = w_btl_sort[i]
        rankings_comp[i][2] = w_thu_sort[i]
        rankings_comp[i][3] = pi_sort[i]
        rankings_comp[i][4] = mle_sort[i]
        rankings_comp[i][5] = lrpr_sort[i]

    df = pd.DataFrame(data = rankings_comp)
    df.to_excel('rankings_comp.xlsx', sheet_name='rankings_comp')

    # entrywise error
    btl_error = max(abs(np.subtract(w_btl, w_norm)))/max(w_norm)
    thu_error = max(abs(np.subtract(w_thu, w_norm)))/max(w_norm)
    spec_error = max(abs(np.subtract(pi, w_norm)))/max(w_norm)
    mle_error = max(abs(np.subtract(w_mle, w_norm)))/max(w_norm)
    lrpr_error = max(abs(np.subtract(w_lrpr, w_norm)))/max(w_norm)

    df = pd.DataFrame(data = [btl_error, thu_error, spec_error, mle_error, lrpr_error])
    df.to_excel('errors.xlsx', sheet_name='errors')


    return rankings_comp, [btl_error, thu_error, spec_error, mle_error, lrpr_error]

# running an instance of a simulation, and recording the results in csv files
def run_trial(n, L, e, gap, rank, err):

    rankings, errors = simulate(n, L, e, gap)
    # print("rankings: ", rankings)
    # print("errors: ", errors)

    str_lp_btl = "\"btl\",\"lp\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","
    str_lp_thu = "\"thu\",\"lp\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","
    str_spec_btl = "\"btl\",\"spec\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","
    str_mle_btl = "\"btl\",\"mle\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","
    str_lrpr_btl = "\"btl\",\"lrpr\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + ","

    err.write("\"btl\",\"lp\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(errors[0]) + "\n")
    err.write("\"thu\",\"lp\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(errors[1]) + "\n")
    err.write("\"btl\",\"spec\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(errors[2]) + "\n")
    err.write("\"btl\",\"mle\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(errors[3]) + "\n")
    err.write("\"btl\",\"lrpr\"," + str(n) + "," + str(e) + "," + str(L) + "," + str(gap) + "," + str(errors[4]) + "\n")

    # objective ranking (w), vs each of our algorithms
    rank_true = set()
    rank_lp_btl = set()
    rank_lp_thu = set()
    rank_spec_btl = set()
    rank_mle_btl = set()
    rank_lrpr_btl = set()

    for k in range(0, 20):
        rank_true.add(rankings[k][0])
        rank_lp_btl.add(rankings[k][1])
        rank_lp_thu.add(rankings[k][2])
        rank_spec_btl.add(rankings[k][3])
        rank_mle_btl.add(rankings[k][4])
        rank_lrpr_btl.add(rankings[k][5])

        lp_btl_correct = len(rank_true.intersection(rank_lp_btl))
        lp_thu_correct = len(rank_true.intersection(rank_lp_thu))
        spec_btl_correct = len(rank_true.intersection(rank_spec_btl))
        mle_btl_correct = len(rank_true.intersection(rank_mle_btl))
        lrpr_btl_correct = len(rank_true.intersection(rank_lrpr_btl))

        lp_btl_correct = lp_btl_correct == k + 1
        lp_thu_correct = lp_thu_correct == k + 1
        spec_btl_correct = spec_btl_correct == k + 1
        mle_btl_correct = mle_btl_correct == k + 1
        lrpr_btl_correct = lrpr_btl_correct == k + 1

        str_lp_btl += (str(round(lp_btl_correct, 3)))
        str_lp_thu += (str(round(lp_thu_correct, 3)))
        str_spec_btl += (str(round(spec_btl_correct, 3)))
        str_mle_btl += (str(round(mle_btl_correct, 3)))
        str_lrpr_btl += (str(round(lrpr_btl_correct, 3)))
    
        if (k==19):
            str_lp_btl += ("\n")
            str_lp_thu += ("\n")
            str_spec_btl += ("\n")
            str_mle_btl += ("\n")
            str_lrpr_btl += ("\n")
        else:
            str_lp_btl += (",")
            str_lp_thu += (",")
            str_spec_btl += (",")
            str_mle_btl += (",")
            str_lrpr_btl += (",")
    
    rank.write(str_lp_btl)
    rank.write(str_lp_thu)
    rank.write(str_spec_btl)
    rank.write(str_mle_btl)
    rank.write(str_lrpr_btl)

        

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

    error.write("\"error\"\n")
    for i in range(0, 20):
        main.write("\"top_" + str(i + 1) + "\"")
        if (i==19):
            main.write("\n")
        else:
            main.write(",")

ranks = open("ranks.csv", "w")
err = open("errors.csv", "w")
init_csv(ranks, err)


for n in [500]:
    #L = n*n*50
    for L in [50]:
        e = 10 * math.log(n) / n
        # e = 0.5
        gap = 0.1
        for j in range(0, 25):
                print(n, L, j)
                run_trial(n, L, e, gap, ranks, err)


ranks.close()
err.close()

