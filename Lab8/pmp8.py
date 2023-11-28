import pandas as pd
import matplotlib.pyplot as plt
import pymc as pm
import numpy as np
import arviz as az

data = pd.read_csv('Prices.csv')

price = data['Price']
speed = data['Speed']
ram = data['Ram']
hard_drive = data['HardDrive']
premium = data['Premium']

with pm.Model() as model:
    
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    b1 = pm.Normal('b1', mu=0, sigma=10)
    b2 = pm.Normal('b2', mu=0, sigma=10)

    eps = pm.HalfCauchy('eps', beta=5)
    nu = pm.Deterministic('nu', alpha + b1 * speed + b2 * np.log(hard_drive))
    y = pm.Normal('price_pred', mu=nu, sigma=eps, observed=price)
    idata_g = pm.sample(50, tune=50, cores=1, return_inferencedata=True)    
    trace = pm.sample(2000, return_inferencedata=True)

az.plot_trace(trace, var_names=['alpha', 'b1', 'b2', 'eps'])
print(az.summary(trace, var_names=['alpha', 'b1', 'b2', 'eps']))
plt.show()

az.plot_posterior(idata_g, var_names='b1', hdi_prob=0.95)
plt.show()
az.plot_posterior(idata_g, var_names='b2', hdi_prob=0.95)
plt.show()


