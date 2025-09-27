import numpy as np
import statsmodels.api as sm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import pandas as pd

# Simulation for Erdos-Renyi model
def simu_density(seed,n,rho,model,evadj_num=1,prob=0.5,hidden_prob=0.75,iter_num_treat=1000,para_mu=[1,1,1],para_sigma=[1,0.5,0.5]):
    """
    Parameters:
    seed : Random seed for reproducibility
    n : Number of units
    rho : Edge density for Erdos-Renyi model
    model : Outcome model type: 'linear' or 'logistic'
    evadj_num : Number of eigenvectors for adjustment
    prob : Treatment assignment probability
    hidden_prob : Probability that edges in the source network are unobserved
    iter_num_treat : Number of Monte Carlo iterations for treatment assignments
    para_mu : Mean value of outcome model parameters  
    para_sigma : Standard deviations of outcome model parameters 
    
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    E = rng.choice([0,1],size=(n,n),p=[1-rho,rho])
    for tmp_idx in range(n):
        E[tmp_idx,tmp_idx] = 0
    #print('rho:', np.sum(E) / n / n) # Print density of the observed network

    EI = E+np.eye(n)
    G = EI.dot(EI.T)
    diag_G = np.diag(G)
    HAC_vec = (2**diag_G[np.newaxis,:]-2**(diag_G[np.newaxis,:]-G)).dot(np.ones(n))/2**np.diag(G)
    HAC_mat = 1-1/2**G
    
    mu_alpha, mu_beta, mu_gamma = para_mu
    sigma_alpha, sigma_beta, sigma_gamma = para_sigma
    alpha = mu_alpha*(1+rng.normal(loc=0,scale=1,size=n)*sigma_alpha)
    beta = mu_beta*(1+rng.standard_t(3,size=n)*sigma_beta)
    gamma = mu_gamma*(1+rng.standard_t(3,size=(n,n))*sigma_gamma)
    
    hidden_matrix = rng.choice([0,1],size=(n,n),p=[hidden_prob,1-hidden_prob])
    tilde_E = E*hidden_matrix
    Q = normalize(tilde_E,norm='l1',axis=1)
    gamma_tilde = gamma*Q
    #print('tilde_rho:', np.sum(tilde_E) / n / n) # Print density of the source network

    e_vals, e_vecs = np.linalg.eig(E.dot(E.T))
    e_vals, e_vecs = np.real(e_vals), np.real(e_vecs)
    sort_ind = np.argsort(e_vals)

    if model == 'linear': # Calculate true effects
        DATE = np.mean(beta)
        IATE = np.sum(gamma_tilde) / n
        TATE = DATE + IATE
    elif model == 'logistic':
        DATE_arr, IATE_arr = np.zeros(100000),np.zeros(100000)
        for tmp in range(100000):
            if tmp % 10000==0:
                print(tmp)
            Z = rng.choice([0,1],size=n,p=[1-prob,prob])
            Z_tilde = Z-prob
            U = alpha + beta * Z_tilde + gamma_tilde.dot(Z_tilde)
            Y = 1/(1+np.exp(-U))
            DATE_arr[tmp] = Y.dot(Z-prob) / (n*prob*(1-prob)) 
            IATE_arr[tmp] = (Z-prob).dot((E.T).dot(Y)) / (n*prob*(1-prob))
        DATE = np.mean(DATE_arr)
        IATE = np.mean(IATE_arr)
        TATE = DATE + IATE
        

    res_point = np.zeros((5,iter_num_treat)) # Estimators of tau_DIR, tau_IND, tau_IND^ev, tau_TOT, tau_TOT^ev
    res_var = np.zeros((5,iter_num_treat)) #Estimators of Var(hat_tau_DIR), Var(hat_tau_IND), Var(hat_tau_IND^ev), Var(hat_tau_TOT), Var(hat_tau_TOT^ev)
    res_var_HAC = np.zeros((5,iter_num_treat)) #Estimators (HAC type) of Var(hat_tau_DIR), Var(hat_tau_IND), Var(hat_tau_IND^ev), Var(hat_tau_TOT), Var(hat_tau_TOT^ev)
    
    for i in range(iter_num_treat): # Run Bernoulli design
        #if i % 100==0:
            #print(i)
        Z = rng.choice([0,1],size=n,p=[1-prob,prob])
        r1 = prob
        r0 = 1 - r1
        Z_tilde = Z-r1
        U = alpha + beta * Z_tilde + gamma_tilde.dot(Z_tilde)
        if model == 'linear':
            Y = U
        elif model == 'logistic':
            Y = 1/(1+np.exp(-U))

        hat_tau_DIR = Y.dot(Z-r1) / (n*r1*r0) 
        hat_tau_IND = (Z-r1).dot((E.T).dot(Y)) / (n*r1*r0)

        VK = e_vecs[:,sort_ind[:-(evadj_num+1):-1]]
        Y_ev = (sm.OLS(Y,np.hstack((np.diag(Z).dot(VK),np.diag(1-Z).dot(VK)))).fit()).resid
        hat_tau_IND_ev = (Z-r1).dot((E.T).dot(Y_ev)) / (n*r1*r0)
        
        res_point[0][i] = hat_tau_DIR
        res_point[1][i] = hat_tau_IND
        res_point[2][i] = hat_tau_IND_ev
        res_point[3][i] = hat_tau_DIR + hat_tau_IND
        res_point[4][i] = hat_tau_DIR + hat_tau_IND_ev

        res_var[0][i] = (np.power(Y,2).dot(Z) / (n*r1)**2 + np.power(Y,2).dot(1-Z) / (n*r0)**2)
        A, A_ev = Y.dot(E), Y_ev.dot(E)
        res_var[1][i] = (np.power(A,2).dot(Z) / (n*r1)**2 + np.power(A,2).dot(1-Z) / (n*r0)**2)
        res_var[2][i] = (np.power(A_ev,2).dot(Z) / (n*r1)**2 + np.power(A_ev,2).dot(1-Z) / (n*r0)**2)
        res_var[3][i] = (np.power(A+Y,2).dot(Z) / (n*r1)**2 + np.power(A+Y,2).dot(1-Z) / (n*r0)**2)
        res_var[4][i] = (np.power(A_ev+Y,2).dot(Z) / (n*r1)**2 + np.power(A_ev+Y,2).dot(1-Z) / (n*r0)**2)

        # For HAC variance estimators, here only consider r1 = r0 = 0.5
        w_DIR = (Z-r1)/r1/r0
        w_IND = E.dot(Z-r1)/r1/r0
        w_TOT = w_DIR + w_IND
        res_var_HAC[0][i] = (np.power(Y*w_DIR,2).dot(HAC_vec)+((Y*w_DIR).T).dot(HAC_mat.dot(Y*w_DIR)))/n/n
        res_var_HAC[1][i] = (np.power(Y*w_IND,2).dot(HAC_vec)+((Y*w_IND).T).dot(HAC_mat.dot(Y*w_IND)))/n/n
        res_var_HAC[2][i] = (np.power(Y_ev*w_IND,2).dot(HAC_vec)+((Y_ev*w_IND).T).dot(HAC_mat.dot(Y_ev*w_IND)))/n/n
        res_var_HAC[3][i] = (np.power(Y*w_TOT,2).dot(HAC_vec)+((Y*w_TOT).T).dot(HAC_mat.dot(Y*w_TOT)))/n/n
        res_var_HAC[4][i] = (np.power(Y_ev*w_TOT,2).dot(HAC_vec)+((Y_ev*w_TOT).T).dot(HAC_mat.dot(Y_ev*w_TOT)))/n/n

    res_SE, res_SE_HAC = np.power(res_var, 0.5), np.power(res_var_HAC, 0.5)

    if 0:
        # Print performance metrics
        print('n='+str(n), 'rho='+str(rho), 'model='+model+':')
        def res_print(res_SE_arr):
            print('tau_DIR:', round(DATE,3))
            print('hat_tau', 'SD', 'SE', 'cover')
            print(list(map(lambda x: round(x,3),[np.mean(res_point[0]), np.var(res_point[0])**0.5, np.mean(res_SE_arr[0]), np.mean(np.abs(res_point[0]-DATE)<1.96*res_SE_arr[0])])))
            print('tau_IND:', round(IATE,3))
            print('hat_tau', 'SD', 'SE', 'cover')
            print(list(map(lambda x: round(x,3),[np.mean(res_point[1]), np.var(res_point[1])**0.5, np.mean(res_SE_arr[1]), np.mean(np.abs(res_point[1]-IATE)<1.96*res_SE_arr[1])])))
            print('hat_tau^ev', 'SD', 'SE', 'cover')
            print(list(map(lambda x: round(x,3),[np.mean(res_point[2]), np.var(res_point[2])**0.5, np.mean(res_SE_arr[2]), np.mean(np.abs(res_point[2]-IATE)<1.96*res_SE_arr[2])])))
            print('tau_TOT:', round(TATE,3))
            print('hat_tau', 'SD', 'SE', 'cover')
            print(list(map(lambda x: round(x,3),[np.mean(res_point[3]), np.var(res_point[3])**0.5, np.mean(res_SE_arr[3]), np.mean(np.abs(res_point[3]-TATE)<1.96*res_SE_arr[3])])))
            print('hat_tau^ev', 'SD', 'SE', 'cover')
            print(list(map(lambda x: round(x,3),[np.mean(res_point[4]), np.var(res_point[4])**0.5, np.mean(res_SE_arr[4]), np.mean(np.abs(res_point[4]-TATE)<1.96*res_SE_arr[4])])))
            return None
        res_print(res_SE)
        res_print(res_SE_HAC)

    # Return performance metrics for different variance estimators
    return [DATE, IATE, TATE,
            [np.mean(res_point[0]), np.var(res_point[0])**0.5, np.mean(res_SE[0]), np.mean(np.abs(res_point[0]-DATE)<1.96*res_SE[0]), np.mean(res_SE_HAC[0]), np.mean(np.abs(res_point[0]-DATE)<1.96*res_SE_HAC[0])],
            [np.mean(res_point[1]), np.var(res_point[1])**0.5, np.mean(res_SE[1]), np.mean(np.abs(res_point[1]-IATE)<1.96*res_SE[1]), np.mean(res_SE_HAC[1]), np.mean(np.abs(res_point[1]-IATE)<1.96*res_SE_HAC[1])],
            [np.mean(res_point[2]), np.var(res_point[2])**0.5, np.mean(res_SE[2]), np.mean(np.abs(res_point[2]-IATE)<1.96*res_SE[2]), np.mean(res_SE_HAC[2]), np.mean(np.abs(res_point[2]-IATE)<1.96*res_SE_HAC[2])],
            [np.mean(res_point[3]), np.var(res_point[3])**0.5, np.mean(res_SE[3]), np.mean(np.abs(res_point[3]-TATE)<1.96*res_SE[3]), np.mean(res_SE_HAC[3]), np.mean(np.abs(res_point[3]-TATE)<1.96*res_SE_HAC[3])],
            [np.mean(res_point[4]), np.var(res_point[4])**0.5, np.mean(res_SE[4]), np.mean(np.abs(res_point[4]-TATE)<1.96*res_SE[4]), np.mean(res_SE_HAC[4]), np.mean(np.abs(res_point[4]-TATE)<1.96*res_SE_HAC[4])]]

if __name__ == "__main__":
    main_seed, seed_iter = 2025, 50
    main_rng = np.random.Generator(np.random.PCG64(main_seed))
    seeds = main_rng.integers(0, 2**32-1, size=seed_iter, dtype=np.uint32)

    all_output = np.zeros((seed_iter,15,7))
    model = 'linear' # Set to 'logistic' for nonlinear outcome model results
    for run_idx, seed in enumerate(seeds): # Run simulation for 50 finite populations
        print('run time: '+str(run_idx+1))
        
        output = np.zeros((15,7))       
        for rho_idx, rho in enumerate([0.001,0.01,0.1]):
            res = simu_density(seed,1000,rho,model)
            output[0+5*rho_idx,0] = res[0]
            output[1+5*rho_idx:3+5*rho_idx,0] = res[1]
            output[3+5*rho_idx:5+5*rho_idx,0] = res[2]
            output[0+5*rho_idx,1:] = res[3]
            output[1+5*rho_idx,1:] = res[4]
            output[2+5*rho_idx,1:] = res[5]
            output[3+5*rho_idx,1:] = res[6]
            output[4+5*rho_idx,1:] = res[7]
        all_output[run_idx] = output

    output_res = np.median(all_output, axis=0)
    df_output = pd.DataFrame(output_res)
    df_output.to_excel('res_'+model+'_rep50.xlsx', index=False, float_format="%.3f")

        

