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
def make_P(n, e, L, w, model):

    if (model == 'btl'):
        P_btl = np.zeros((n, n))
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
                # else: P_ij = 0
                else:
                    P_btl[i][j] = 0

        df = pd.DataFrame(data=P_btl)
        df.to_excel('P_btl.xlsx', sheet_name='P_btl')
        return P_btl
                
    elif (model == 'thu'):
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
            
    else:
       raise Exception("Invalid model")


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
def lp_algorithm(P, model):

    if (model == 'btl'):

        m_btl = gp.Model("BTL")
        # x: 1D vector (used to approximate s)
        x_btl = m_btl.addMVar((n, 1), lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="x")
        # z: nxn matrix, z = Y - P
        z_btl = m_btl.addMVar((n, n), lb=0.0, vtype=GRB.CONTINUOUS, name="z")

        zeroMat = np.zeros((n, n))
        eVec = np.full((n, 1), 1)
        mask = np.zeros((n, n))

        # P with linking function applied
        link_btl = np.zeros((n, n))
        for i in range(0, n):
            for j in range(0, n):
                # we assume no probabilities are zero, if btl is zero, so is thu
                if (P[i][j] == 0):
                    link_btl[i][j] = 0
                else:
                    # 1 / (1 - x) undefined
                    if (P[i][j] == 1):
                        link_btl[i][j] = 0
                        continue
                    link_btl[i][j] = math.log2(P[i][j] / (1 - P[i][j]))
                    if (i != j):
                        mask[i][j] = 1

        # constraints
        m_btl.addConstr((x_btl @ eVec.T) - (eVec @ x_btl.T) + z_btl - link_btl >= zeroMat)
        m_btl.addConstr((x_btl @ eVec.T) - (eVec @ x_btl.T) - z_btl - link_btl <= zeroMat)

        # constraints include entire matrix, but we only care when link[i][j] != 0 and i != j; so apply a mask: np.multiply(z, mask)
        m_btl.setObjective((z_btl * mask).sum(), GRB.MINIMIZE)

        m_btl.optimize()

        # initialized approximation vector of ideal w
        w_btl = []

        # for btl:  undo log ->   x = log(w) => w = 2^x
        for i in range(0, n):
            w_btl.append(pow(2, x_btl.X[i])[0])

        w_btl = w_btl / sum(w_btl)
        return w_btl

    elif (model == 'thu'):

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

    else:
        raise Exception("Invalid model")    


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

def mle_algorithm(P_btl, w_min, w_max):

    w_spec = spec_algorithm(P_btl)
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
                    if (P_btl[i][j] != 0 and j != i):
                        curr_P += P_btl[i][j] * math.log2(tau / (tau + w_spec[j])) + (1 - P_btl[i][j]) * math.log2(w_spec[j]/ (tau + w_spec[j]))

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

def lrpr_algorithm(P, model):

    if (model == 'btl'):
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
                    link[i][j] = math.log2(P[i][j] / (1 - P[i][j]))      

        os = OptSpace(2, 5, 0.0001)
        U, S, V = os.solve(link)

        opt_mat = np.matmul(np.matmul(U, S), V.T)

        df = pd.DataFrame(data = opt_mat)
        df.to_excel('opt_mat.xlsx', sheet_name='opt_mat')

        inv_link = np.zeros((n, n))
        
        for i in range(0, n):
            for j in range(0, n):
                    inv_ij = math.pow(2, opt_mat[i][j]) / (1 + math.pow(2, opt_mat[i][j]))
                    inv_ji = math.pow(2, opt_mat[j][i]) / (1 + math.pow(2, opt_mat[j][i]))
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

    elif (model == 'thu'):
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

    else:
        raise Exception("Invalid model")

# run a trial with n items, 
# where e is the probability that any pair of elements will be compared, 
# and L is the number of comparisons per pair
def simulate(n, L, e, gap, model):
    w = make_w(n, gap)
    w_norm = np.asarray(w) / sum(w)
    P = make_P(n, e, L, w, model)

    if (model == 'btl'):
        #==================Linear Program==================
        # models: btl, thurstone
        # approximating w vector with: w_btl, w_thu
        print("before lp")
        w_lp = lp_algorithm(P, model)
        #==================Spectral Method==================
        # models: btl
        # approximating w vector with: pi
        print("before spec")
        pi = spec_algorithm(P)

        df = pd.DataFrame(data = pi)
        df.to_excel('pi.xlsx', sheet_name='pi')
        #======================MLE==========================
        # models: btl
        # approximating w vector with: w_mle
        print("before mle")
        w_mle = mle_algorithm(P, min(w_norm), max(w_norm))

        df = pd.DataFrame(data = w_mle)
        df.to_excel('w_mle.xlsx', sheet_name='w_mle')

        #================Low Rank Pairwise Ranking=================
        # models: btl, thu
        # approximating w vector with: w_lrpr
        print("before lrpr")
        w_lrpr = lrpr_algorithm(P, model) 
        print("done")

        #==================Comparing Algorithms==================

        w_comp = np.zeros((n, 6))

        # comparing normalized w (ideal), our algorithm (btl), spectral mle (btl), our algorithm (thurstone), spectral MLE (btl)
        for i in range(0, n):
            w_comp[i][0] = i
            w_comp[i][1] = w_norm[i]
            w_comp[i][2] = w_lp[i]
            w_comp[i][3] = pi[i]
            w_comp[i][4] = w_mle[i]
            w_comp[i][5] = w_lrpr[i]

        w_sort = np.argsort(w)
        lp_sort = np.argsort(w_lp)
        pi_sort = np.argsort(pi)
        mle_sort = np.argsort(w_mle)
        lrpr_sort = np.argsort(w_lrpr)

        rankings_comp = np.zeros((n, 5))

        for i in range(0, n):
            rankings_comp[i][0] = w_sort[i]
            rankings_comp[i][1] = lp_sort[i]
            rankings_comp[i][2] = pi_sort[i]
            rankings_comp[i][3] = mle_sort[i]
            rankings_comp[i][4] = lrpr_sort[i]

        df = pd.DataFrame(data = rankings_comp)
        df.to_excel('rankings_comp.xlsx', sheet_name='rankings_comp')

        l_inf_err = [0]*4
        D_w_err = [0]*4

        # l infinity error
        
        l_inf_err[0] = max(abs(np.subtract(w_lp, w_norm)))/max(w_norm)
        l_inf_err[1] = max(abs(np.subtract(pi, w_norm)))/max(w_norm)
        l_inf_err[2] = max(abs(np.subtract(w_mle, w_norm)))/max(w_norm)
        l_inf_err[3] = max(abs(np.subtract(w_lrpr, w_norm)))/max(w_norm)

        # D_w error
        for i in range(0, n):
            for j in range(i, n):
                if ((w_norm[i] - w_norm[j]) * (w_lp[i] - w_lp[j]) > 0):
                    D_w_err[0] += math.pow((w_norm[i] - w_norm[j]), 2)
                if ((w_norm[i] - w_norm[j]) * (pi[i] - pi[j]) > 0):
                    D_w_err[1] += math.pow((w_norm[i] - w_norm[j]), 2)
                if ((w_norm[i] - w_norm[j]) * (w_mle[i] - w_mle[j]) > 0):
                    D_w_err[2] += math.pow((w_norm[i] - w_norm[j]), 2)
                if ((w_norm[i] - w_norm[j]) * (w_lrpr[i] - w_lrpr[j]) > 0):
                    D_w_err[3] += math.pow((w_norm[i] - w_norm[j]), 2)

        D_w_err[0] = math.sqrt(D_w_err[0] / (2*n*math.pow(np.linalg.norm(w), 2)))
        D_w_err[1] = math.sqrt(D_w_err[1] / (2*n*math.pow(np.linalg.norm(w), 2)))
        D_w_err[2] = math.sqrt(D_w_err[2] / (2*n*math.pow(np.linalg.norm(w), 2)))
        D_w_err[3] = math.sqrt(D_w_err[3] / (2*n*math.pow(np.linalg.norm(w), 2)))

     

        df = pd.DataFrame(data = l_inf_err)
        df.to_excel('l_inf_err.xlsx', sheet_name='l_inf_err')

        return rankings_comp, l_inf_err, D_w_err
        
    elif (model == 'thu'):
        #==================Linear Program==================
        # models: btl, thurstone
        # approximating w vector with: w_btl, w_thu
        print("before lp")
        w_lp = lp_algorithm(P, model)
        #================Low Rank Pairwise Ranking=================
        # models: btl, thu
        # approximating w vector with: w_lrpr
        print("before lrpr")
        w_lrpr = lrpr_algorithm(P, model) 
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

        l_inf_err = [1]*2
        D_w_err = [1]*2

        # entrywise error
        l_inf_err[0] = max(abs(np.subtract(w_lp, w_norm)))/max(w_norm)
        l_inf_err[1] = max(abs(np.subtract(w_lrpr, w_norm)))/max(w_norm)

        df = pd.DataFrame(data = l_inf_err)
        df.to_excel('l_inf_err.xlsx', sheet_name='l_inf_err')

        return rankings_comp, l_inf_err, D_w_err
    
    else:
        raise Exception("Invalid model")


# running an instance of a simulation, and recording the results in csv files
def run_trial(n, L, e, gap, model, rank, err):

    rankings, l_inf_err, D_w_err = simulate(n, L, e, gap, model)
    # print("rankings: ", rankings)
    # print("errors: ", errors)

    if (model == 'btl'):

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

    elif (model == 'thu'):

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

    else:
       raise Exception("Invalid model")

        

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

ranks = open(("rank_" + model + ".csv"), "w")
err = open(("error_" + model + ".csv"), "w")
init_csv(ranks, err)


for n in [500]:
    # for e in [(math.log(n) / n), 5*(math.log(n) / n), 10*(math.log(n) / n), 15*(math.log(n) / n), 20*(math.log(n) / n)]:
    for e in [(math.log(n) / n)]:
        L = 30
        gap = 0.0
        for j in range(0, 20):
                print(n, L, j)
                run_trial(n, L, e, gap, model, ranks, err)


ranks.close()
err.close()

