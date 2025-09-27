import numpy as np
import statsmodels.api as sm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import pandas as pd

# Simulation under two-sided market setting 
def simu_twosided(seed,nr,nc,model,prob=0.5,hidden_prob=0.75,iter_num_treat=1000,para_mu=[1,1,0.5],para_sigma=[1,0.5,0.5]):
    """
    Parameters:
    seed : Random seed for reproducibility
    nr, nc : Number of rows and columns in the two-sided market grid
    model : Outcome model type: 'linear' or 'logistic'
    prob : Treatment assignment probability
    hidden_prob : Probability that edges in the source network are unobserved
    iter_num_treat : Number of Monte Carlo iterations for treatment assignments
    para_mu : Mean value of outcome model parameters  
    para_sigma : Standard deviations of outcome model parameters 
    
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n = nr*nc
    E=np.block([[np.ones((nc,nc)) if i==j else np.zeros((nc,nc)) for i in range(nr)] for j in range(nr)])
    for i in range(n):
        for j in range(1,nr):
            E[i,(i+j*nc)%n]=1
    for tmp_idx in range(n):
        E[tmp_idx,tmp_idx] = 0
    #print('rho:', np.sum(E) / n / n) # Print density of the observed network

    mu_alpha, mu_beta, mu_gamma = para_mu # Generate outcome model parameters 
    sigma_alpha, sigma_beta, sigma_gamma = para_sigma
    para_adj = np.empty((nr, nc))    
    mu_k = np.array([[-2,0], [0,2]]) 
    size_row, size_col = nr // 2, nc // 2
    for tmp_i in range(2):
        for tmp_j in range(2):
            para_adj[tmp_i*size_row:(tmp_i+1)*size_row, tmp_j*size_col:(tmp_j+1)*size_col] = mu_k[tmp_i, tmp_j]
    str_effect = para_adj.reshape(n) 
    alpha = rng.normal(loc=mu_alpha,scale=sigma_alpha,size=n) 
    alpha += str_effect
    beta = mu_beta+rng.standard_t(3,size=n)*sigma_beta
    beta += str_effect
    gamma = mu_gamma+rng.standard_t(3,size=(n,n))*sigma_gamma
    
    hidden_matrix = rng.choice([0,1],size=(n,n),p=[hidden_prob,1-hidden_prob])
    tilde_E = E*hidden_matrix
    Q = normalize(tilde_E,norm='l1',axis=1)
    gamma_tilde = gamma*Q
    #print('tilde_rho:', np.sum(tilde_E) / n / n) # Print density of the source network

    c_col = np.zeros((nr,nc)) 
    c_col[:,:nc//2] = 1
    S = np.zeros((n,3)) 
    S[:,0] = 1
    S[:,1][:n//2] = 1
    S[:,2] += c_col.reshape(n)

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

    if 0:
        # Print performance metrics for Bernoulli design
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

    res_smrd = np.zeros((5,iter_num_treat)) # Estimators of tau_DIR, tau_IND, Var(hat_tau_DIR^SMRD), Var(hat_tau_IND^SMRD), Var(hat_tau_TOT^SMRD)
    nr1, nc1 = nr//2, nc//2
    for i in range(iter_num_treat): # Run simple multiple randomization design
        #if i % 100 == 0:
            #print(i)
        Zr= np.zeros(nr)
        Zr[:nr1] = np.ones(nr1)
        rng.shuffle(Zr)
        Zc= np.zeros(nc)
        Zc[:nr1] = np.ones(nc1)
        rng.shuffle(Zc)
        Ztr_mat, Zib_mat, Zis_mat, Zcc_mat = Zr.reshape((nr,1)).dot(Zc.reshape((1,nc))), Zr.reshape((nr,1)).dot((1-Zc).reshape((1,nc))), (1-Zr).reshape((nr,1)).dot(Zc.reshape((1,nc))), (1-Zr).reshape((nr,1)).dot((1-Zc).reshape((1,nc)))
        Z_mat = [Ztr_mat, Zib_mat, Zis_mat, Zcc_mat]
        Ztr = (Ztr_mat).reshape(n)
        Zib = (Zib_mat).reshape(n)
        Zis = (Zis_mat).reshape(n)
        Zcc = (Zcc_mat).reshape(n)
        Z = Ztr
        r1 = prob
        r0 = 1 - r1
        Z_tilde = Z-r1
        U = alpha + beta * Z_tilde + gamma_tilde.dot(Z_tilde)
        if model == 'linear':
            Y = U
        elif model == 'logistic':
            Y = 1/(1+np.exp(-U))
        Y_mat = Y.reshape((nr,nc))
        ytr = np.sum(Y*Ztr)/np.sum(Ztr)
        yib = np.sum(Y*Zib)/np.sum(Zib)
        yis = np.sum(Y*Zis)/np.sum(Zis)
        ycc = np.sum(Y*Zcc)/np.sum(Zcc)
        res_smrd[0][i] = ytr-(nr1-1)/nr1*yib-(nc1-1)/nc1*yis+((nr1-1)/nr1+(nc1-1)/nc1-1)*ycc
        res_smrd[1][i] = (nr-1)/nr1*(yib-ycc) + (nc-1)/nc1*(yis-ycc)

        I_gamma = np.array([nr1, nr1, nr-nr1, nr-nr1])
        J_gamma = np.array([nc1, nc-nc1, nc1, nc-nc1])
        sigmar, sigmac, sigmarc = np.zeros(4), np.zeros(4), np.zeros(4)
        varr_group, varc_group = np.zeros(4), np.zeros(4)
        for gtmp in range(4):
            Y_tmp = (Y_mat[Z_mat[gtmp]==1]).reshape((I_gamma[gtmp],J_gamma[gtmp]))
            yr_bar_tmp = np.mean(Y_tmp, axis=1)
            yc_bar_tmp = np.mean(Y_tmp, axis=0)
            varr_group[gtmp] = np.mean(np.power(Y_tmp - np.mean(Y_tmp, axis=1, keepdims=True),2))
            varc_group[gtmp] = np.mean(np.power(Y_tmp - np.mean(Y_tmp, axis=0, keepdims=True),2))
            sigmar[gtmp] = np.var(yr_bar_tmp)
            sigmac[gtmp] = np.var(yc_bar_tmp)
            Ydif = Y_tmp - np.mean(Y_tmp, axis=1, keepdims=True) - np.mean(Y_tmp, axis=0, keepdims=True)
            sigmarc[gtmp] = np.var(Ydif)
        ar = (nr-I_gamma)/nr/I_gamma
        ac = (nc-J_gamma)/nc/J_gamma
        sigma_arr = ((ar*sigmar+ac*sigmac+ar*ac*sigmarc)/(1-ar-ac+ar*ac) - ar/(1-ar)*(nc-J_gamma)/nc/(J_gamma-1)*varr_group - ac/(1-ac)*(nr-I_gamma)/nr/(I_gamma-1)*varc_group)
        beta_smrd_dir = np.array([1,-(nr1-1)/nr1,-(nc1-1)/nc1,(nr1-1)/nr1+(nc1-1)/nc1-1])
        beta_smrd_ind = np.array([0,(nr-1)/nr1,(nc-1)/nc1,-(nr-1)/nr1-(nc-1)/nc1])
        beta_smrd_tot = beta_smrd_dir+beta_smrd_ind
        res_smrd[2][i] = np.abs(beta_smrd_dir).dot(np.power(sigma_arr,0.5))
        res_smrd[3][i] = np.abs(beta_smrd_ind).dot(np.power(sigma_arr,0.5))
        res_smrd[4][i] = np.abs(beta_smrd_tot).dot(np.power(sigma_arr,0.5))

    if 0:
        # Print performance metrics for simple multiple randomization design
        print('SMRD', 'model='+model+':')
        print('tau_DIR:', round(DATE,3))
        print('smrd:')
        print(list(map(lambda x: round(x,3),[np.mean(res_smrd[0]), np.var(res_smrd[0])**0.5,np.mean(res_smrd[2]),np.mean(np.abs(res_smrd[0]-DATE)<1.96*res_smrd[2])])))
        print('tau_IND:', round(IATE,3))
        print('smrd:')
        print(list(map(lambda x: round(x,3),[np.mean(res_smrd[1]), np.var(res_smrd[1])**0.5,np.mean(res_smrd[3]),np.mean(np.abs(res_smrd[1]-IATE)<1.96*res_smrd[3])])))
        print('tau_TOT:', round(TATE,3))
        print('smrd:')
        print(list(map(lambda x: round(x,3),[np.mean(res_smrd[0]+res_smrd[1]), np.var(res_smrd[0]+res_smrd[1])**0.5,np.mean(res_smrd[4]),np.mean(np.abs(res_smrd[0]+res_smrd[1]-TATE)<1.96*res_smrd[4])])))

    # Return results for all estimators
    return [DATE, IATE, TATE, 
            [np.mean(res_point[0]), np.var(res_point[0])**0.5, np.mean(res_SE[0]), np.mean(np.abs(res_point[0]-DATE)<1.96*res_SE[0])],
            [np.mean(res_smrd[0]), np.var(res_smrd[0])**0.5,np.mean(res_smrd[2]),np.mean(np.abs(res_smrd[0]-DATE)<1.96*res_smrd[2])],
            [np.mean(res_point[1]), np.var(res_point[1])**0.5, np.mean(res_SE[1]), np.mean(np.abs(res_point[1]-IATE)<1.96*res_SE[1])],
            [np.mean(res_point[2]), np.var(res_point[2])**0.5, np.mean(res_SE[2]), np.mean(np.abs(res_point[2]-IATE)<1.96*res_SE[2])],
            [np.mean(res_smrd[1]), np.var(res_smrd[1])**0.5,np.mean(res_smrd[3]),np.mean(np.abs(res_smrd[1]-IATE)<1.96*res_smrd[3])],
            [np.mean(res_point[3]), np.var(res_point[3])**0.5, np.mean(res_SE[3]), np.mean(np.abs(res_point[3]-TATE)<1.96*res_SE[3])],
            [np.mean(res_point[4]), np.var(res_point[4])**0.5, np.mean(res_SE[4]), np.mean(np.abs(res_point[4]-TATE)<1.96*res_SE[4])],
            [np.mean(res_smrd[0]+res_smrd[1]), np.var(res_smrd[0]+res_smrd[1])**0.5,np.mean(res_smrd[4]),np.mean(np.abs(res_smrd[0]+res_smrd[1]-TATE)<1.96*res_smrd[4])]]
    
if __name__ == "__main__":
    main_seed, seed_iter = 2026, 50
    main_rng = np.random.Generator(np.random.PCG64(main_seed))
    seeds = main_rng.integers(0, 2**32-1, size=seed_iter, dtype=np.uint32)

    all_output = np.zeros((seed_iter,8,5))
    model = 'linear' # Set to 'logistic' for nonlinear outcome model results
    for run_idx, seed in enumerate(seeds): # Run simulation for 50 finite populations
        print('run time: '+str(run_idx+1))

        output = np.zeros((8,5))
        res = simu_twosided(seed,60,60,model)
        output[0:2,0] = res[0]
        output[2:5,0] = res[1]
        output[5:8,0] = res[2]
        for tmp_idx in range(8):
            output[tmp_idx,1:] = res[tmp_idx+3]
        all_output[run_idx] = output
        
    output_res = np.median(all_output, axis=0)
    df_output = pd.DataFrame(output_res)
    df_output.to_excel('res_'+model+'_rep50.xlsx', index=False, float_format="%.3f")
