import numpy as np
import statsmodels.api as sm
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import normalize
import pandas as pd

# Simulation based on the email data
def simu_email(seed,model,evadj_num,prob=0.5,hidden_prob=0.75,iter_num_treat=1000,para_mu=[1,1,1],para_sigma=[1,0.5,0.5]):
    """
    Parameters:
    seed : Random seed for reproducibility
    model : Outcome model type: 'linear' or 'logistic'
    evadj_num : Number of eigenvectors for adjustment
    prob : Treatment assignment probability
    hidden_prob : Probability that edges in the source network are unobserved
    iter_num_treat : Number of Monte Carlo iterations for treatment assignments
    para_mu : Mean value of outcome model parameters  
    para_sigma : Standard deviations of outcome model parameters 
    
    """
    rng = np.random.Generator(np.random.PCG64(seed))
    n = 1005
    E = np.zeros((n,n))
    file_email = open('email-Eu-core.txt','r')
    line = file_email.readline()
    while line:
        pair = (line.strip()).split(' ')
        E[int(pair[0])][int(pair[1])] = 1
        line = file_email.readline()
    file_email.close()
    for tmp_idx in range(n):
        E[tmp_idx,tmp_idx] = 0
    #print('rho:', np.sum(E) / n / n) # Print density of the observed network

    EI = E+np.eye(n)
    G = EI.dot(EI.T)
    diag_G = np.diag(G)
    HAC_vec = (2**diag_G[np.newaxis,:]-2**(diag_G[np.newaxis,:]-G)).dot(np.ones(n))/2**np.diag(G)
    HAC_mat = 1-1/2**G

    mu_alpha, mu_beta, mu_gamma = para_mu # Generate outcome model parameters 
    sigma_alpha, sigma_beta, sigma_gamma = para_sigma
    alpha = mu_alpha+rng.normal(loc=0,scale=1,size=n)*sigma_alpha
    beta = mu_beta+rng.standard_t(3,size=n)*sigma_beta
    gamma = mu_gamma+rng.standard_t(3,size=(n,n))*sigma_gamma

    e_vals, e_vecs = np.linalg.eig(E.dot(E.T))
    e_vals, e_vecs = np.real(e_vals), np.real(e_vecs)
    sort_ind = np.argsort(e_vals)
    
    hidden_matrix = rng.choice([0,1],size=(n,n),p=[hidden_prob,1-hidden_prob])
    tilde_E = E*hidden_matrix
    Q = normalize(tilde_E,norm='l1',axis=1)
    gamma_tilde = gamma*Q
    #print('tilde_rho:', np.sum(tilde_E) / n / n) # Print density of the source network

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
        # Print performance metrics for current eigenvector adjustment number
        print('evadj_num='+str(evadj_num), 'model='+model+':')
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
        #res_print(res_SE_HAC)

    # Return performance metrics for indirect and total effects with and without adjustments
    return [[np.mean(res_point[1])-IATE, np.var(res_point[1])**0.5, np.mean(res_SE[1]), np.mean(np.abs(res_point[1]-IATE)<1.96*res_SE[1]), np.mean(res_SE_HAC[1]), np.mean(np.abs(res_point[1]-IATE)<1.96*res_SE_HAC[1])],
            [np.mean(res_point[3])-TATE, np.var(res_point[3])**0.5, np.mean(res_SE[3]), np.mean(np.abs(res_point[3]-TATE)<1.96*res_SE[3]), np.mean(res_SE_HAC[3]), np.mean(np.abs(res_point[3]-TATE)<1.96*res_SE_HAC[3])],
            [np.mean(res_point[2])-IATE, np.var(res_point[2])**0.5, np.mean(res_SE[2]), np.mean(np.abs(res_point[2]-IATE)<1.96*res_SE[2]), np.mean(res_SE_HAC[2]), np.mean(np.abs(res_point[2]-IATE)<1.96*res_SE_HAC[2])],
            [np.mean(res_point[4])-TATE, np.var(res_point[4])**0.5, np.mean(res_SE[4]), np.mean(np.abs(res_point[4]-TATE)<1.96*res_SE[4]), np.mean(res_SE_HAC[4]), np.mean(np.abs(res_point[4]-TATE)<1.96*res_SE_HAC[4])]]

# Plot top 100 eigenvalues of the email network
def plot_eigenvalue_100():
    n = 1005
    E = np.zeros((n,n))
    file_email = open('email-Eu-core.txt','r')
    line = file_email.readline()
    while line:
        pair = (line.strip()).split(' ')
        E[int(pair[0])][int(pair[1])] = 1
        line = file_email.readline()
    file_email.close()
    for tmp_idx in range(n):
        E[tmp_idx,tmp_idx] = 0
    
    e_vals, e_vecs = np.linalg.eig(E.dot(E.T))
    e_vals = np.real(e_vals)
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
    plt.savefig('top_100_eigenvalues_plot.eps')
    plt.show()

if __name__ == "__main__":
    # Plot top 100 eigenvalues
    plot_eigenvalue_100()

    # Run simulation for 50 finite populations
    main_seed, seed_iter = 2025, 50
    main_rng = np.random.Generator(np.random.PCG64(main_seed))
    seeds = main_rng.integers(0, 2**32-1, size=seed_iter, dtype=np.uint32)

    all_output = np.zeros((seed_iter,11,12))
    for run_idx, seed in enumerate(seeds):
        print('run time: '+str(run_idx+1))
        
        output = np.zeros((11,12))
        for evadj_num in range(11):
            if evadj_num == 0:
                res_IATE0, res_TATE0 = simu_email(seed,'linear',evadj_num+1)[:2]
                output[evadj_num,:6]=res_IATE0
                output[evadj_num,6:]=res_TATE0
            else:   
                res_tmp = simu_email(seed,'linear',evadj_num)[2:]
                output[evadj_num,:6]=res_tmp[0]
                output[evadj_num,6:]=res_tmp[1]
                
        all_output[run_idx] = output

    output_res = np.median(all_output, axis=0)
    df_output = pd.DataFrame(output_res)
    df_output.to_excel('res_linear_rep50.xlsx', index=False, float_format="%.3f")

    # Analyze results and generate plots
    df_output = pd.read_excel('res_linear_rep50.xlsx')
    output_res = df_output.values
    
    summary_TATE = np.zeros((6,11))
    bias_TATE = output_res[:,6]
    SD_TATE = output_res[:,7]
    SE_TATE = output_res[:,8]
    CP_TATE = output_res[:,9]
    CP_HAC_TATE = output_res[:,11]
    summary_TATE[0] = bias_TATE
    summary_TATE[1] = SD_TATE
    summary_TATE[2] = np.power(np.power(bias_TATE,2)+np.power(SD_TATE,2),0.5)
    summary_TATE[3] = SE_TATE
    summary_TATE[4] = CP_TATE
    summary_TATE[5] = CP_HAC_TATE
    
    k_values = np.arange(11)
    metrics = ['Bias', 'SD', 'RMSE', 'SE']
    fig, axs = plt.subplots(1, 2, figsize=(12, 7))

    for i in range(4): 
        if metrics[i] == 'Bias':
            axs[0].plot(k_values, abs(summary_TATE[i]), marker='o', label=f'Absolute {metrics[i]}')
        else:
            axs[0].plot(k_values, summary_TATE[i], marker='o', label=f'{metrics[i]}')
 
    axs[0].set_xlabel('Number of eigenvectors', fontsize=16)  
    axs[0].tick_params(axis='both', labelsize=14) 
    axs[0].grid(True)
    axs[0].legend(fontsize=14) 

    axs[1].plot(k_values, summary_TATE[4], marker='D', color='purple', label='CP')
    axs[1].plot(k_values, summary_TATE[5], marker='s', color='purple', label='CP (HAC)')
    axs[1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% Reference')

    axs[1].set_xlabel('Number of eigenvectors', fontsize=16)
    axs[1].tick_params(axis='both', labelsize=14)
    axs[1].grid(True)
    axs[1].legend(fontsize=14) 
    axs[1].set_ylim(0.85, 1.01) 
    
    plt.tight_layout()  
    plt.savefig('eigenvector_number_plot_rep50.eps')
    plt.show()

