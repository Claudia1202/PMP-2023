import numpy as np

lam = 20
mean_time_order = 2  
std_dev_time_order = 0.5  

alpha = 1
nr_clients = np.random.poisson(lam)
time_order = np.random.normal(mean_time_order, std_dev_time_order, nr_clients)
time_cooking_order = np.random.exponential(alpha, nr_clients)

total = np.sum(time_order) + np.sum(time_cooking_order)

print(f"Clienti: {nr_clients}")
print(f"Timp total servire: {total}")

