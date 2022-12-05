# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 19:58:24 2022

@author: Will Fossett
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random
random.seed(0)
plt.rcParams['axes.grid'] = True  # Applies a grid to every plot
plt.rcParams['figure.dpi'] = 150  # increases clarity of all plots, default 100


# ---------------------------- Multi-Use Functions ----------------------------
# Use pandas to read files into arrays, separating data from labels
def read_data(pct_train):
    data = np.array(pd.read_csv(r'./data.csv'))[:, 1:]  # Remove team name
    el = len(data)
    tr = random.sample(range(0, el), int(el*pct_train))  # rand train inst
    train_data = data[tr].astype(float)
    te = []  # find which elements in the data have not been chosen for train
    for i in range(0, el):
        if range(0, el)[i] not in tr:
            te.append(range(0, el)[i])
    test_data = data[te].astype(float)
    train_reg = data[tr, 0].astype(float)
    test_reg = data[te, 0].astype(float)
    return train_data[:, 1:], test_data[:, 1:], train_reg, test_reg


# Calculate mean vector of data
def mean(data):
    # data: nx64 array, 64 features
    d = np.shape(data)[1]
    r = np.shape(data)[0]
    m = np.zeros([d, 1])
    for i in range(0, d):
        m[i] = np.sum(data[:, i])/r
    return m


# Compute eigenvalues and eigenvectors, not sorted
def eig(sigma):
    eigs = np.linalg.eig(sigma)
    eig_vals = eigs[0]
    eig_vec = eigs[1]
    return eig_vals, eig_vec


# Calculate number of eigenvectors k to reach proportion of variance
def proportion_variance(eigs, prop_limit):
    d = np.shape(eigs)[0]  # num of eigenvalues
    den = np.sum(eigs)  # denominator of proportion of variance
    prop_var = np.zeros([d, 2])  # ordered pairs
    for i in range(0, d):
        prop_var[i, 0] = i
        num = np.sum(eigs[0:i])  # calculate sum up to the ith eigval
        prop_var[i, 1] = num/den
    return np.where(prop_var[:, 1] >= prop_limit)[0][0]


# Calculate scree graph to determine number of relevant dimensions
def scree(eigs):
    # ------------------------------ Scree graph ------------------------------
    d = np.shape(eigs)[0]  # num of eigenvalues
    scree_array = np.zeros([d, 2])
    for i in range(0, d):  # store the sorted eigs in ordered pairs for plottin
        scree_array[i, 0] = i
        scree_array[i, 1] = eigs[i]
    # ---------------------- Proportion of variance plot ----------------------
    den = np.sum(eigs)  # denominator of proportion of variance
    prop_var = np.zeros([d, 2])  # ordered pairs
    for i in range(0, d):
        prop_var[i, 0] = i
        num = np.sum(eigs[0:i])  # calculate sum up to the ith eigval
        prop_var[i, 1] = num/den
    return prop_var, scree_array


# Plot Scree stuff
def plot_scree(prop_var, scree_array, prop_limit, k):
    # ------------------------------ Scree graph ------------------------------
    el = len(prop_var)
    mx = np.ceil(np.max(scree_array))*1.1
    plt.plot(scree_array[:, 0]+1, scree_array[:, 1], c='0.25', linewidth=1)
    plt.scatter(scree_array[:, 0]+1, scree_array[:, 1],
                marker='+', c='k', s=50)
    plt.xlabel('Eigenvectors')
    plt.ylabel('Eigenvalues')
    plt.title('Scree graph for CFB Data')
    plt.xlim([0, el])
    plt.ylim([-5, mx])
    # ---------------------- Proportion of variance plot ----------------------
    plt.figure()
    plt.plot(prop_var[:, 0]+1, prop_var[:, 1], c='0.25', linewidth=1,
             label='_nolegend_')
    plt.scatter(prop_var[:, 0]+1, prop_var[:, 1], marker='+', c='k', s=50,
                label='_nolegend_')
    plt.xlabel('Eigenvectors')
    plt.ylabel('Prop. of Var')
    plt.title('Proportion of Variance')
    plt.xlim([0, el])
    plt.ylim([-0.1, 1.05])
    plt.vlines(k+1, -1, prop_limit, colors='k', linestyles='dotted')
    return None


# Project data onto k largest eigenvalues/eigenvectors
def project(sigma, m_vector, data, k):
    eig_vals, eig_vec = eig(sigma)
    sorted_eigs = np.sort(eig_vals)  # sorted low to high
    W = np.zeros([k, len(sigma)])
    for i in range(0, k):
        val = sorted_eigs[-(i+1)]  # store k largest eigenvectors
        W[i, :] = eig_vec.T[eig_vals == val]
    W = W.T
    z = np.zeros([len(data), k])
    m_vector = m_vector.reshape((np.shape(data)[1],))
    for i in range(0, len(data)):
        z[i, :] = W.T@(data[i, :]-m_vector)  # Eq 6.9
    return z, W


# Calculate the total error, specific error, avg error, and avg non26 error
def error(r, reg, thresh):
    diff = np.zeros(len(r))
    nonzero_avg = 0
    count = 0
    for i in range(0, len(r)):
        diff[i] = np.abs(r[i]-reg[i])
        if np.abs(r[i] - 26) > thresh:  # far enuf to assume not 26 confidently
            nonzero_avg += diff[i]
            count += 1
    return np.sum(diff), diff, np.sum(diff)/len(diff), nonzero_avg/count


# Plot ranking specific error in bar chart
def bar_chart(r, reg, diff, test_or_train, reg_type):
    t = np.zeros((26,))
    el = []
    for i in range(1, len(t)+1):
        idx = np.argwhere(reg == i)
        d = diff[idx]
        if len(d) == 0:
            t[i-1] = 0
        else:
            t[i-1] = np.sum(d)/len(d)
        el.append(i)
    plt.figure()
    plt.grid()
    plt.bar(el, t, zorder=10)
    plt.grid(zorder=0)
    plt.title(str(reg_type) + ' ' + str(test_or_train) +
              ' Regression Error per Ranking')
    plt.xlabel('Ranking')
    plt.ylabel('Avg Error')
    return None


# Create table to show estimated classes for relevant categories
def table(t_class, r_class, tt_class, rt_class, reg_type):
    headers = ['Category', 'True Training', 'Training Est',
               'True Testing', 'Testing Est']
    col = ['Top 4', 'Top 10', 'Top 25', 'Not Top 25']
    er = np.array([col, t_class.astype(int), r_class.astype(int),
                   tt_class.astype(int), rt_class.astype(int)], dtype=object).T
    df = pd.DataFrame(er, columns=headers)
    fig, ax = plt.subplots()
    t = ax.table(cellText=df.values, colLabels=df.columns,
                 loc='center', cellLoc='center')
    ax.set_title('Prediction Estimates of Common CFB Categories' +
                 ' after ' + reg_type + ' Regression')
    ax.axis('off')
    fig.tight_layout()
    t.set_fontsize(10)
    return None


# Classify each ranking regression to see if in top 4, top 10, top 25, or not
def classify(r, reg, thresh):
    r_class = np.zeros((4,))  # top 4, top 10, top 25, not top 25
    t_class = np.zeros((4,))
    t_class[3] = len(np.argwhere(reg == 26))
    t_class[2] = len(np.argwhere((reg <= 25) & (reg > 10)))
    t_class[1] = len(np.argwhere((reg <= 10) & (reg > 4)))
    t_class[0] = len(np.argwhere(reg <= 4))
    for i in range(0, len(reg)):
        if r[i] > 26 - thresh:
            r_class[3] += 1
        elif (r[i] <= 26 - thresh) & (r[i] > 10 - thresh):
            r_class[2] += 1
        elif (r[i] <= 10 - thresh) & (r[i] > 4-thresh):
            r_class[1] += 1
        else:
            r_class[0] += 1
    return r_class, t_class


# ------------------------------ k-NN Functions -------------------------------
# Uses kNN discriminant to estimate class, returns % accuracy
def kNN_estimator(knn, W, m, test_data, z, train_reg):
    r = np.zeros(len(test_data))
    for i in range(0, len(test_data)):
        z_inst = W.T@(test_data[i]-m.reshape(len(m),))  # transform test inst
        # distance from instance to all other train data PCA
        z_dist = np.zeros(len(z-z_inst))
        for j in range(0, len(z-z_inst)):
            # Euclidean Distance
            z_dist[j] = np.linalg.norm(z[j]-z_inst)
        z_i_sort = np.argsort(z_dist)  # index of sorted closest values
        rk = train_reg[z_i_sort][0:knn]  # values of sorted closest instances
        bins = np.bincount(rk.astype(int))  # most occuring reg values
        if (bins[-1] >= (knn*0.75)) & (len(bins) == 27):  # if 75% > are 26
            r[i] = 26  # just call it a 26
        else:  # if most of the values are not 26
            nonzero = np.argwhere(rk != 26)  # find the non 26 values
            r[i] = np.sum(rk[nonzero])/len(nonzero)  # avg them
            # the logic above is to essentially classify then regression.
            # first, if most nn are 26 indicating outside of top 25, call it a
            # 26. if not, only average the non26 values as if you include
            # the 26 values, the average gets skewed down. and 26 is essential
            # ly a class, not a regression value so no sense in calc with it
    return r


# -------------------------- Neural Network Functions -------------------------
# Normalize input space
def norm_x(x):
    xn = []
    for i in range(0, len(x)):
        xn.append(x[i]/np.max(x[i]))
    return np.array(xn)


# Initialize weights and biases
def init_weights(ins, hids, outs, kappa):
    a = np.zeros((ins+1, hids))  # Hidden weights
    b = np.zeros((hids+1, outs))  # Output weights
    cHid = np.zeros((ins+1, hids))  # weight changes
    cOut = np.zeros((ins+hids+1, outs))
    dHid = np.zeros((ins+1, hids))  # Error derivatives
    dOut = np.zeros((hids+ins+1, outs))
    eHid = np.zeros((ins+1, hids))  # Adaptive learning rates
    eOut = np.zeros((hids+ins+1, outs))
    fHid = np.zeros((ins+1, hids))  # Error derivative average
    fOut = np.zeros((hids+ins+1, outs))
    y = np.zeros(hids)  # Hidden node outputs
    z = np.zeros(outs)  # Output node outputs
    p = np.zeros(outs)  # dE/dv
    for j in range(0, hids):
        for i in range(0, ins+1):
            a[i, j] = 0.2*(random.random()-0.5)
            eHid[i, j] = kappa
    for k in range(0, outs):
        for j in range(0, hids+1):
            if j % 2 == 0:
                b[j, k] = 1
            else:
                b[j, k] = -1
    return a, b, cHid, cOut, dHid, dOut, eHid, eOut, fHid, fOut, y, z, p


# Run the neural network to approximate target function t, returns apprx
# data s and error for each epoch error
def NN(x, t, kappa, phi, theta, mu, ins, hids, outs, max_e):
    a, b, cHid, cOut, dHid, dOut, eHid, eOut, fHid, fOut, y, z, p = \
        init_weights(ins, hids, outs, kappa)
    epoch = 0
    error = np.zeros(max_e)
    error_sum = np.zeros((max_e, outs)).T
    examples = len(x.T)
    s = np.zeros((examples, outs)).T
    while epoch < max_e:
        for n in range(0, examples):  # each example in the data set
            for j in range(0, hids):
                u = a[ins, j]  # bias weight
                for i in range(0, ins):
                    u = u + (a[i, j]*x[i, n])  # weighted sum
                y[j] = 1/(1+np.exp(-u))  # logistic
            for k in range(0, outs):
                v = b[hids, k]  # bias weight
                for j in range(0, hids):
                    v = v + (b[j, k] * y[j])  # weighted sum
                z[k] = 1/(1+np.exp(-v))  # logistic
            s[:, n] = z[:]  # store estimated values
            error_sum[:, epoch] = error_sum[:, epoch] + \
                np.sum(np.abs(z[:]-t[:, n]))
            # Backpropagation
            q = np.zeros(hids)  # reset dE/du to 0
            for k in range(0, outs):  # calc error derivative on output node
                p[k] = (z[k]-t[k, n])*z[k]*(1-z[k])
                dOut[hids+1, k] = dOut[hids+1, k] + p[k]  # bias weight
                for j in range(0, hids):  # hid weights
                    dOut[j, k] = dOut[j, k] + p[k]*y[j]
                    q[j] = q[j] + p[k]*b[j, k]
            for j in range(0, hids):  # error derivative for hidden node
                q[j] = q[j]*y[j]*(1-y[j])
                dHid[ins, j] = dHid[ins, j] + q[j]  # bias weight
                for i in range(0, ins):
                    dHid[i, j] = dHid[i, j] + q[j]*x[i, n]
        for j in range(0, hids):  # end of examples, change hidden node weights
            for i in range(0, ins):
                if dHid[i, j]*fHid[i, j] > 0:
                    eHid[i, j] = eHid[i, j] + kappa
                else:
                    eHid[i, j] = eHid[i, j] * phi
                fHid[i, j] = theta*fHid[i, j] + (1-theta)*dHid[i, j]
                cHid[i, j] = mu*cHid[i, j] - (1-mu)*eHid[i, j]*dHid[i, j]
                a[i, j] = a[i, j] + cHid[i, j]
        for k in range(0, outs):  # change output node weights
            for j in range(0, hids+1):
                if dOut[j, k]*fOut[j, k] > 0:
                    eOut[j, k] = eOut[j, k] + kappa
                else:
                    eOut[j, k] = eOut[j, k] * phi
                fOut[j, k] = theta*fOut[j, k] + (1-theta)*dOut[j, k]
                cOut[j, k] = mu*cOut[j, k]-(1-mu)*eOut[j, k]*dOut[j, k]
                b[j, k] = b[j, k]+cOut[j, k]
        dHid[:, :] = 0  # reset error derivatives to 0
        dOut[:, :] = 0
        error[epoch] = np.sum(np.abs(s - t))
        epoch += 1
        if epoch % 50 == 0:
            print('Epoch ' + str(epoch) + ' out of ' + str(max_e))
    return s, error, a, b


# Use the weights in the trained NN to predict test data
def test_NN(a, b, test_data, hids, outs, ins, x):
    examples = len(test_data)
    y = np.zeros(hids)
    z = np.zeros(outs)
    s = np.zeros((examples, outs)).T
    for n in range(0, examples):
        for j in range(0, hids):
            u = a[ins, j]
            for i in range(0, ins):
                u = u + (a[i, j]*x[i, n])
            y[j] = 1/(1+np.exp(-u))
        for k in range(0, outs):
            v = b[hids, k]
            for j in range(0, hids):
                v = v + (b[j, k] * y[j])
            z[k] = 1/(1+np.exp(-v))
        s[:, n] = z[:]
    return s


# ---------------------------- 2022 CFP Prediction ----------------------------
# Import 2022 data, which has no CFP rank
def read_2022():
    data = np.array(pd.read_csv(r'./2022.csv'))[:, 1:].astype(float)
    names = np.array(pd.read_csv(r'./2022.csv'))[:, 0]
    return data, names


# ------------------------------------ Main -----------------------------------
def main():
    # Compile data and project
    thresh = 0.01  # threshold for regression accuracy and classification
    train_data, test_data, train_reg, test_reg = read_data(.80)
    sigma = np.cov(train_data.T)
    eigs, vecs = eig(sigma)
    prop_lim = 0.99
    k = proportion_variance(eigs, prop_lim)
    prop_var, scree_array = scree(eigs)
    plot_scree(prop_var, scree_array, prop_lim, k)
    m = mean(train_data)
    z, W = project(sigma, m, train_data, k)
    # Begin k-NN
    print('Beginning k-NN Algorithm')
    knn = 8
    r = kNN_estimator(knn, W, m, train_data, z, train_reg)
    r_class, t_class = classify(r, train_reg, thresh)
    e, diff, avg, nonzero_avg = error(r, train_reg, thresh)
    bar_chart(r, train_reg, diff, 'Training', 'k-NN')
    rt = kNN_estimator(knn, W, m, test_data, z, train_reg)
    rt_class, tt_class = classify(rt, test_reg, thresh)
    et, dt, at, nzt = error(rt, test_reg, thresh)
    bar_chart(rt, test_reg, dt, 'Testing', 'k-NN')
    table(t_class, r_class, tt_class, rt_class, 'k-NN')
    # Begin NN
    print('Beginning Neural Network')
    # NN structure and parameters
    ins = 14
    hids = 18
    outs = 1
    kappa = 0.1
    phi = 0.5
    theta = 0.7
    mu = 0.75
    # Format data appropriately, normalize values
    x = norm_x(train_data.T)  # normalize each feature
    t = (train_reg.reshape(len(train_reg), 1)/26).T  # normalize training vals
    max_e = 4250  # 4250 takes a long time but is accurate
    s, epoch_error, a, b = NN(x, t, kappa, phi, theta, mu, ins, hids,
                              outs, max_e)
    s = s[0]*26  # scale values back to 1-26 scale
    en, diffn, avgn, nonzero_avgn = error(s, train_reg, thresh)
    plt.figure()
    plt.plot(np.linspace(0, max_e, max_e), epoch_error)
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.title('Error vs Epoch')
    bar_chart(s, train_reg, diffn, 'Training', 'NN')
    xt = norm_x(test_data.T)
    stest = test_NN(a, b, test_data, hids, outs, ins, xt)
    stest = stest[0]*26
    ent, diffnt, avgnt, nonzero_avgnt = error(stest, test_reg, thresh)
    bar_chart(stest, test_reg, diffnt, 'Testing', 'NN')
    s_class = classify(s, train_reg, thresh)
    st_class = classify(stest, test_reg, thresh)
    table(t_class, s_class[0], tt_class, st_class[0], 'NN')
    # Begin prediction for 2022
    d_2022, teams = read_2022()
    xd = norm_x(d_2022.T)
    nn_2022 = test_NN(a, b, d_2022, hids, outs, ins, xd)
    nn_2022 = nn_2022[0]*26
    knn_2022 = kNN_estimator(knn, W, m, d_2022, z, train_reg)
    headers = ['Estimated CFP Rank', 'k-NN', 'NN']
    col = teams
    vals = np.array([col, knn_2022, nn_2022.reshape(len(nn_2022.T),)])
    df = np.array(pd.DataFrame(vals.T, columns=headers))
    nn_top4 = np.array(sorted(df, key=lambda x: x[2]))[0:4]
    knn_top4 = np.array(sorted(df, key=lambda x: x[1]))[0:4]
    col = [1, 2, 3, 4]
    vals = np.array([col, knn_top4[:, 0], nn_top4[:, 0]])
    df = pd.DataFrame(vals.T, columns=headers)
    fig, ax = plt.subplots()
    t = ax.table(cellText=df.values, colLabels=df.columns,
                 loc='center', cellLoc='center')
    ax.set_title('2022 CFP Predictions')
    ax.axis('off')
    fig.tight_layout()
    t.set_fontsize(10)
    return None


main()
