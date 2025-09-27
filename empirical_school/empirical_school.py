import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

total_df = pd.read_stata("37070-0001-Data.dta",convert_categoricals=False)
treated_school = total_df[total_df['SCHTREAT']==1]
schid = treated_school['SCHID']
schidnum = []
for i in set(schid.values):
    schidnum.append(len(treated_school[treated_school['SCHID']==i]))
sort_ind= np.argsort(schidnum)
if 1:
    schid_5largest = np.array(list(set(schid.values)))[(sort_ind[-5:])[::-1]] #24, 56, 22, 60, 58
    df_5largest = treated_school[treated_school['SCHID'].isin(schid_5largest)]
    df_dm = df_5largest[df_5largest['TREAT'].isin([0,1,2])] # Drop treatment missing
    #651(22), 805(24), 634(56), 581(58), 635(60)
    sch_dm = [22,24,56,58,60]
    ind_add = [0, 651, 651+805, 651+805+634, 651+805+634+581]
m = len(df_dm)
E_dm = np.zeros((m,m))

for i in range(5):
    df_i = df_dm[df_dm['SCHID']==sch_dm[i]]
    m_i = len(df_i)
    uidarr = list(df_i['ID'].values)
    start_i = ind_add[i]
    for j in range(m_i):
        infor_j = df_i.iloc[j]
        for k in range(10):
            name_k = str(int(infor_j['ST'+str(k+1)]))
            if name_k in uidarr:
                id_k = uidarr.index(name_k)
                E_dm[start_i+j, start_i+id_k] = 1

elig_bool = df_dm['TREAT'].isin([1,2]) #180+180
norm_bool = np.sum(E_dm[:,df_dm['TREAT'].isin([1,2]).values],axis=1) > 0 #1685
#sample_bool = elig_bool | norm_bool #1855
#sample_bool = elig_bool & norm_bool #150
sample_bool = elig_bool
df_tmp = df_dm[sample_bool]
E_tmp = E_dm[sample_bool][:,sample_bool]

y_not_missing = (df_tmp['WRISTW2'].isin([0,1])).values
df = df_tmp[y_not_missing]
E = E_tmp[y_not_missing][:,y_not_missing]

if 0:
    sch_spec = (df['SCHID']==60).values
    df = df[sch_spec]
    E = E[sch_spec][:,sch_spec]
    
Z = (df['TREAT']==1).values.astype(int)
Y = df['WRISTW2'].values.astype(int)
n = len(Z) #1609
r1 = np.sum(Z) / n
tau_dir = Y.dot(Z-r1) / (n*r1*(1-r1))
tau_ind = (Z-r1).dot(E.dot(Y)) / (n*r1*(1-r1))
se_dir = (np.power(Y,2).dot(Z) / (n*r1)**2 + np.power(Y,2).dot(1-Z) / (n*(1-r1))**2)**0.5
A = Y.dot(E)
se_ind = (np.power(A,2).dot(Z) / (n*r1)**2 + np.power(A,2).dot(1-Z) / (n*(1-r1))**2)**0.5
tau_tot = tau_dir + tau_ind
se_tot = (np.power(A+Y,2).dot(Z) / (n*r1)**2 + np.power(A+Y,2).dot(1-Z) / (n*(1-r1))**2)**0.5

#e_vals, e_vecs = np.linalg.eig(E.dot(E.T))
#e_vals, e_vecs = np.real(e_vals), np.real(e_vecs)
#np.save('e_vals_school.npy', e_vals)
#np.save('e_vecs_school.npy', e_vecs)
#e_vals = np.load('e_vals_school.npy')
#e_vecs = np.load('e_vecs_school.npy')
#sort_ind = np.argsort(e_vals)
#VK = e_vecs[:,sort_ind[:-24:-1]]
#Y_tilde = (sm.OLS(Y,np.hstack((np.diag(Z).dot(VK),np.diag(1-Z).dot(VK)))).fit()).resid

if 1:
    S = np.zeros((n,5))
    for i in range(5):
        S[:,i] = (df['SCHID']==sch_dm[i]).values.astype(int)
    reg_tmp = np.hstack((Z.reshape((n,1))*S,(1-Z.reshape((n,1)))*S))
    Y_tilde = (sm.OLS(Y,reg_tmp).fit()).resid    
    
tau_ind_ev = (Z-r1).dot(E.dot(Y_tilde)) / (n*r1*(1-r1))
A_ev = Y_tilde.dot(E)
se_ind_ev = (np.power(A_ev,2).dot(Z) / (n*r1)**2 + np.power(A_ev,2).dot(1-Z) / (n*(1-r1))**2)**0.5
tau_tot_ev = tau_dir + tau_ind_ev
se_tot_ev = (np.power(A_ev+Y,2).dot(Z) / (n*r1)**2 + np.power(A_ev+Y,2).dot(1-Z) / (n*(1-r1))**2)**0.5

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

print(res)

                
        
    
                            

