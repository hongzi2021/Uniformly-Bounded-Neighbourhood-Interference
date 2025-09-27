import numpy as np
import statsmodels.api as sm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import pandas as pd

# Simulation under partial interference setting   
def simu_cluster(seed,nm,m,model,prob=0.5,hidden_prob=0.75,iter_num_treat=1000,para_mu=[1,1,0.5],para_sigma=[1,0.5,0.5]):
    """
    Parameters:
    seed : Random seed for reproducibility
    nm: cluster size
    m: number of clusters
    model : Outcome model type: 'linear' or 'logistic'
    prob : Treatment assignment probability
    hidden_prob : Probability that edges in the source network are unobserved
    iter_num_treat : Number of Monte Carlo iterations for treatment assignments
    para_mu : Mean value of outcome model parameters  
    para_sigma : Standard deviations of outcome model parameters 
    
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n = nm * m
    K = 3
    num_strata = m//K
    
    E = np.kron(np.eye(m), np.ones((nm, nm)))
    for tmpi in range(n): 
        E[tmpi,tmpi] = 0
    #print('rho:', np.sum(E) / n / n) # Print density of the observed network
        
    mu_k = [-1,0,1] # Generate outcome model parameters
    alpha = np.zeros(n) 
    beta = np.zeros(n)
    gamma = np.zeros((n,n))
    mu_alpha, mu_beta, mu_gamma = para_mu
    sigma_alpha, sigma_beta, sigma_gamma = para_sigma
    for cl_tmp in range(K):
        mean_k = mu_k[cl_tmp]
        alpha[cl_tmp*(n//K):(cl_tmp+1)*(n//K)] = rng.normal(loc=mean_k+mu_alpha,scale=sigma_alpha,size=n//K)
        beta[cl_tmp*(n//K):(cl_tmp+1)*(n//K)] = mean_k+mu_beta+rng.standard_t(3,size=n//K)*sigma_beta
        for gamma_tmp in range(m//K):
            gamma[cl_tmp*(n//K)+gamma_tmp*nm:cl_tmp*(n//K)+(gamma_tmp+1)*nm,cl_tmp*(n//K)+gamma_tmp*nm:cl_tmp*(n//K)+(gamma_tmp+1)*nm] = mu_gamma+rng.standard_t(3,size=(nm,nm))*sigma_gamma
    
    hidden_matrix = rng.choice([0,1],size=(n,n),p=[hidden_prob,1-hidden_prob]) 
    E_tilde = E*hidden_matrix                   
    Q = normalize(E_tilde,norm='l1',axis=1)
    gamma_tilde = gamma*Q
    #print('rho_tilde:', np.sum(E_tilde) / n / n) # Print density of the source network
    
    S = np.zeros((n,K)) 
    for s_tmp in range(K):
        S[s_tmp*(n//K):(s_tmp+1)*(n//K), s_tmp] = 1            

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

        reg_tmp = np.hstack((Z.reshape((n,1))*S,(1-Z.reshape((n,1)))*S))
        Y_ev = (sm.OLS(Y,reg_tmp).fit()).resid
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
        
    res_SE = np.power(res_var, 0.5)

    res_cl = np.zeros((2,iter_num_treat)) # Estimators of tau_TOT^cl, Var(hat_tau_TOT^cl)
    for i in range(iter_num_treat): # Run cluster randomization
        Z_tmp = rng.choice([0,1],size=m,p=[1-prob,prob])
        Z = np.repeat(Z_tmp,nm)
        r1 = prob
        r0 = 1 - r1
        Z_tilde = Z-r1
        U = alpha + beta * Z_tilde + gamma_tilde.dot(Z_tilde)
        if model == 'linear':
            Y = U
        elif model == 'logistic':
            Y = 1/(1+np.exp(-U))
        Y_cl = np.zeros(m)
        for j in range(m):
            Y_cl[j] = np.mean(Y[j*nm:(j+1)*nm])
        res_cl[0][i] = Y_cl.dot(Z_tmp-r1) / (m*r1*r0)
        res_cl[1][i] = (np.power(Y_cl,2).dot(Z_tmp)/(m*r1)**2 + np.power(Y_cl,2).dot(1-Z_tmp)/(m*r0)**2)**0.5

    if 0:
        # Print performance metrics
        print('Bernoulli', 'model='+model+':')
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
        #res_point(res_SE_HAC)

        print('cluster', 'model='+model+':')
        print(list(map(lambda x: round(x,3),[np.mean(res_cl[0]),np.var(res_cl[0])**0.5, np.mean(res_cl[1]), np.mean(np.abs(res_cl[0]-TATE)<1.96*res_cl[1])])))

    # Return results for all estimators   
    return [DATE, IATE, TATE,
            [np.mean(res_point[0]), np.var(res_point[0])**0.5, np.mean(res_SE[0]), np.mean(np.abs(res_point[0]-DATE)<1.96*res_SE[0])],
            [np.mean(res_point[1]), np.var(res_point[1])**0.5, np.mean(res_SE[1]), np.mean(np.abs(res_point[1]-IATE)<1.96*res_SE[1])],
            [np.mean(res_point[2]), np.var(res_point[2])**0.5, np.mean(res_SE[2]), np.mean(np.abs(res_point[2]-IATE)<1.96*res_SE[2])],
            [np.mean(res_point[3]), np.var(res_point[3])**0.5, np.mean(res_SE[3]), np.mean(np.abs(res_point[3]-TATE)<1.96*res_SE[3])],
            [np.mean(res_point[4]), np.var(res_point[4])**0.5, np.mean(res_SE[4]), np.mean(np.abs(res_point[4]-TATE)<1.96*res_SE[4])],
            [np.mean(res_cl[0]),np.var(res_cl[0])**0.5, np.mean(res_cl[1]), np.mean(np.abs(res_cl[0]-TATE)<1.96*res_cl[1])]]

if __name__ == "__main__":
    main_seed, seed_iter = 2026, 50
    main_rng = np.random.Generator(np.random.PCG64(main_seed))
    seeds = main_rng.integers(0, 2**32-1, size=seed_iter, dtype=np.uint32)

    all_output = np.zeros((seed_iter,6,5))
    model = 'linear' # Set to 'logistic' for nonlinear outcome model results
    for run_idx, seed in enumerate(seeds): # Run simulation for 50 finite populations
        print('run time: '+str(run_idx+1))

        output = np.zeros((6,5))
        res = simu_cluster(seed,20,180,model)
        output[0,0] = res[0]
        output[1:3,0] = res[1]
        output[3:6,0] = res[2]
        for tmp_idx in range(6):
            output[tmp_idx,1:] = res[tmp_idx+3]
        all_output[run_idx] = output
        
    output_res = np.median(all_output, axis=0)
    df_output = pd.DataFrame(output_res)
    df_output.to_excel('res_'+model+'_rep50.xlsx', index=False, float_format="%.3f")
