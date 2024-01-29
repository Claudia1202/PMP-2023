import pandas as pd
import pymc as pm
import numpy as np


##ex1
# a. setul de date
val = pd.read_csv('BostonHousing.csv')
val = val[['medv', 'rm', 'crim', 'indus']]

# b
with pm.Model() as model:

    X = val[['rm', 'crim', 'indus']].values  #var independente
    y = val['medv'].values   #var dependenta
    intercept = pm.Normal('Intercept', mu=0, sigma=10)
    slopes = pm.Normal('slopes', mu=0, sigma=10, shape=3)  # 3 pt rm, crim, indus

    mu = intercept + pm.math.dot(X, slopes)
    sigma = pm.Uniform('sigma', lower=0, upper=20)
    y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=y)  #likelihood 
    trace = pm.sample(5000, tune=2000)  #sampling

#c
summary = pm.summary(trace, var_names=['slopes', 'Intercept', 'sigma']).round(2)
print(summary)

# var cu cea mai mare influenta
max_effect_var = summary.loc['slopes']['mean'].idxmax()
print(max_effect_var)

# d.
ppc = pm.sample_posterior_predictive(trace, samples=5000, model=model)  #distrib predictiv posterioara
medv_pred = ppc['Y_obs']  
medv_hdi = pm.hdi(medv_pred, hdi_prob=0.5)

print("Intervalul de predictie HDI:", medv_hdi)


