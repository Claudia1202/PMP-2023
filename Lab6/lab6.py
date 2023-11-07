import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

theta_val = [0.2, 0.5]
Y_val = [0, 5, 10]

dictt = {}  #dictionar in care stocam

for Y in Y_val:
    for th in theta_val:
        with pm.Model() as model:
            n = pm.Poisson('n', mu=10)

            pm.Binomial('obs', n=n, p=th, observed=Y)
            trace = pm.sample(1000, tune=1000, cores=1, chains=1)
            dictt[(Y, th)] = trace

for key, c in dictt.items():
    az.plot_posterior(c)
    plt.title(f"Y={key[0]}, theta={key[1]}")
    plt.show()
