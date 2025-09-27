import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Read insurance data 
df = (pd.read_stata("0422survey.dta")[['id','takeup_survey','delay','intensive']]).dropna()
dt = (df.astype(int)).values
df_adj = (pd.read_stata("0422allinforawnet.dta")[['id','network_id']]).dropna()
dt_adj = (df_adj.astype(int)).values

# Construct network
n = len(dt)
E = np.zeros((n,n))
idlst = list(dt[:,0])
for line in dt_adj:
    a, b = line[0], line[1]
    if a in idlst:
        if b in idlst:
            ind1 = idlst.index(a)
            ind2 = idlst.index(b)
            E[ind1,ind2] = 1
            
Z = dt[:,3]
Y = dt[:,1]

# Calculate estimators and variance estimates    
r1 = 0.5
tau_dir = Y.dot(Z-r1) / (n*r1*(1-r1))
tau_ind = (Z-r1).dot(E.dot(Y)) / (n*r1*(1-r1))
se_dir = (np.power(Y,2).dot(Z) / (n*r1)**2 + np.power(Y,2).dot(1-Z) / (n*(1-r1))**2)**0.5
A = Y.dot(E)
se_ind = (np.power(A,2).dot(Z) / (n*r1)**2 + np.power(A,2).dot(1-Z) / (n*(1-r1))**2)**0.5
tau_tot = tau_dir + tau_ind
se_tot = (np.power(A+Y,2).dot(Z) / (n*r1)**2 + np.power(A+Y,2).dot(1-Z) / (n*(1-r1))**2)**0.5

e_vals, e_vecs = np.linalg.eig(E.dot(E.T))
e_vals, e_vecs = np.real(e_vals), np.real(e_vecs)
#np.save('e_vals.npy', np.real(e_vals))
#np.save('e_vecs.npy', np.real(e_vecs))
#e_vals = np.load('e_vals_insurance.npy')
#e_vecs = np.load('e_vecs_insurance.npy')

# Eigenvectors-adjusted estimators and variance estimates
sort_ind = np.argsort(e_vals)
VK = e_vecs[:,sort_ind[:-28:-1]]
Y_tilde = (sm.OLS(Y,np.hstack((np.diag(Z).dot(VK),np.diag(1-Z).dot(VK)))).fit()).resid
tau_ind_ev = (Z-r1).dot(E.dot(Y_tilde)) / (n*r1*(1-r1))
A = Y_tilde.dot(E)
se_ind_ev = (np.power(A,2).dot(Z) / (n*r1)**2 + np.power(A,2).dot(1-Z) / (n*(1-r1))**2)**0.5
tau_tot_ev = tau_dir + tau_ind_ev
se_tot_ev = (np.power(A+Y,2).dot(Z) / (n*r1)**2 + np.power(A+Y,2).dot(1-Z) / (n*(1-r1))**2)**0.5

# HAC variance estimation
EI = E+np.eye(n)
G = EI.dot(EI.T)
diag_G = np.diag(G)
HAC_vec = (2**diag_G[np.newaxis,:]-2**(diag_G[np.newaxis,:]-G)).dot(np.ones(n))/2**np.diag(G)
HAC_mat = 1-1/2**G
r0 = 1-r1
w_DIR = (Z-r1)/r1/r0
w_IND = E.dot(Z-r1)/r1/r0
w_TOT = w_DIR + w_IND
se_dir_HAC = ((np.power(Y*w_DIR,2).dot(HAC_vec)+((Y*w_DIR).T).dot(HAC_mat.dot(Y*w_DIR)))/n/n)**0.5
se_ind_HAC = ((np.power(Y*w_IND,2).dot(HAC_vec)+((Y*w_IND).T).dot(HAC_mat.dot(Y*w_IND)))/n/n)**0.5
se_ind_ev_HAC = ((np.power(Y_tilde*w_IND,2).dot(HAC_vec)+((Y_tilde*w_IND).T).dot(HAC_mat.dot(Y_tilde*w_IND)))/n/n)**0.5
se_tot_HAC = ((np.power(Y*w_TOT,2).dot(HAC_vec)+((Y*w_TOT).T).dot(HAC_mat.dot(Y*w_TOT)))/n/n)**0.5
se_tot_ev_HAC = ((np.power(Y_tilde*w_TOT,2).dot(HAC_vec)+((Y_tilde*w_TOT).T).dot(HAC_mat.dot(Y_tilde*w_TOT)))/n/n)**0.5

res = np.array([[tau_dir, tau_ind, tau_tot, tau_ind_ev, tau_tot_ev],
                [se_dir, se_ind, se_tot, se_ind_ev, se_tot_ev],
                [se_dir_HAC, se_ind_HAC, se_tot_HAC, se_ind_ev_HAC, se_tot_ev_HAC]])

# Print performance metrics
print(res)

# Plot top 100 eigenvalues
def plot_eigenvalue_100():
    ev_sorted = np.sort(e_vals)[::-1]
    
    top_100_e_vals = ev_sorted[:100]

    indices_to_annotate = [1, 5, 10, 20, 50, 100]
    values_to_annotate = [top_100_e_vals[i - 1] for i in indices_to_annotate]

    fig, axs = plt.subplots(figsize=(10, 6))
    axs.plot(range(1, 101), top_100_e_vals, marker='o', linestyle='-', color='b', markersize=4, linewidth=1, label='Eigenvalue')
    axs.plot(indices_to_annotate, values_to_annotate, marker='o', linestyle='', color='r')
    axs.set_title('Top 100 Eigenvalues', fontsize=18)
    axs.set_xlabel('Index', fontsize=16)
    axs.tick_params(axis='both', which='major', labelsize=14)
    axs.grid(True, linestyle='--', alpha=0.7)
    for idx, val in zip(indices_to_annotate, values_to_annotate):
        if idx != 100 and idx != 1:
            axs.annotate(f'({idx}, {val:.2f})', (idx, val), textcoords="offset points", xytext=(40,8), ha='center', fontsize=14)
        elif idx == 100:
            axs.annotate(f'({idx}, {val:.2f})', (idx, val), textcoords="offset points", xytext=(-15,8), ha='center', fontsize=14)
        else:
            axs.annotate(f'({idx}, {val:.2f})', (idx, val), textcoords="offset points", xytext=(50,0), ha='center', fontsize=14)
    axs.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig('insurance_eigenvalues_plot.eps')
    plt.show()
 
plot_eigenvalue_100()
