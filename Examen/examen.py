import pandas as pd
import pymc as pm
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

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


##2.

def posterior_grid(grid_points=50, trials=5, success=1):
  
    grid = np.linspace(0, 1, grid_points)   #intervalul pt theta
    apriori = np.repeat(1 / grid_points, grid_points)  #prob a priori
    likelihood = stats.binom.pmf(success, trials, p=grid)  #verosim
    posterior = likelihood *apriori  #verosim * prob a priori
    posterior /= posterior.sum() 
    return grid, posterior

#parametrii
success = 1  
grid_points = 50 
trials_values = [5, 10, 20]

#grafic
plt.figure(figsize=(12, 8))

for trials in trials_values:
    grid, posterior = posterior_grid(grid_points, trials, success)
    plt.plot(grid, posterior, 'o-', label=f'Trials = {trials}')

plt.title(f'Grafic')
plt.xlabel('theta')
plt.legend()
plt.show()
