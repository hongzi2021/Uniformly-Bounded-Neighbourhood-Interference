Replication Code for paper "Causal inference under uniformly bounded neighbourhood interference". 
The code implements the methods discussed in the paper and reproduces the main tables and figures.

Authors:
Xin Lu, Hongzi Li and Hanzhong Liu
Department of Statistics and Data Science, Tsinghua University, 100084, Beijing, China

Dependencies:
- Python 3.10.10
- numpy 1.24.2
- pandas 1.5.3
- statsmodels 0.14.0
- matplotlib 3.7.0
- seaborn 0.12.2
- scikit-learn 1.3.0
- networkx 3.1

Description:
- Python files ('.py') in each folder implement the numerical simulations and experimental data analyses from the main text and Supplementary Material
- Stata files ('.dta') in 'empirical_insurance' contain the insurance experiment data used in the main text Section 7.2
- The file 'empirical_school/37070-0001-Data.dta' contains the anti-conflict intervention data used in Supplementary Material Section B.4

Usage:
1. Run 'simu_twosided/simu_twosided.py' to generate the results in Table 1.
2. Run 'simu_email/simu_email.py' to generate Figures 1-2.
3. Run 'empirical_insurance/empirical_insurance.py' to generate the results in Table 2 and Figure S.1.
4. Run 'simu_partial/simu_partial.py' to generate the results in Table S.1.
5. Run 'simu_density/simu_density.py' to generate the results in Tables S.2-S.3.
6. Run 'empirical_school/empirical_school.py' to generate the results in Table S.4.