import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

with pm.Model() as model:
    data = pd.read_csv('auto-mpg.csv')

    data['horsepower'] = pd.to_numeric(data['horsepower'], errors='coerce')

    data.dropna(subset=['horsepower'], inplace=True)

    horsepower_data = data['horsepower'].values.astype(float)

    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta = pm.Normal('beta', mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)

    mu = alpha + beta * data['horsepower']

    mpg = pm.Normal('mpg', mu=mu, sigma=sigma, observed=data['mpg'])

with model:
    trace = pm.sample(1000, tune=1000)

az.summary(trace).round(2)

az.plot_trace(trace)
plt.show()

plt.scatter(data['horsepower'], data['mpg'], label='Date observate')

az.plot_posterior_predictive(trace, samples=100, var_name='mpg', color='red', alpha=0.1)

hdi = az.hdi(trace['mpg'], hdi_prob=0.95)
plt.fill_between(data['horsepower'], hdi[0], hdi[1], color='lightblue', alpha=0.4, label='95%HDI')

plt.title('Regresie liniara Bayesiana')
plt.xlabel('CP')
plt.ylabel('mpg')
plt.legend()
plt.show()
