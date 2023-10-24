import numpy as np

lambda_clienti_pe_ora = 20
timp_mediu_plasare = 2  
deviatie_standard_plasare = 0.5  
timp_maxim = 15  

def simuleaza_servire(alpha, timp_maxim):
    t_servire = []
    for _ in range(10000):  
        t_pregatire = np.random.exponential(alpha)
        t_plasare = np.random.normal(loc=timp_mediu_plasare, scale=deviatie_standard_plasare)
        total = t_pregatire + t_plasare
        t_servire.append(total)
    
    t_servire = np.array(t_servire)
    prob_servire_in_timp = np.mean(t_servire <= timp_maxim)    
    return prob_servire_in_timp

alpha_max = 12.0
alpha_min = 0.01

while  True:
    alpha_test = (alpha_min + alpha_max) / 2
    probabilitate_servire = simuleaza_servire(alpha_test, timp_maxim)
    if probabilitate_servire < 0.95:
        alpha_max = alpha_test
    else:       
        rez = alpha_test
        break
print(f" {rez:.2f} ")

##########ex3

alpha = rez  
timp_mediu_plasare = 2  
timp_mediu_asteptare = alpha + timp_mediu_plasare
print(f"Timpul mediu asteptare servire: {timp_mediu_asteptare:.2f} ")
