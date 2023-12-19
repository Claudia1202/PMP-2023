import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt

clusters = 3
n_cluster = [200, 150, 150] 
n_total = sum(n_cluster)
means = [5, 0, 2] 
std_devs = [2, 2, 1]  
mix = np.random.normal(np.repeat(means, n_cluster), np.repeat(std_devs, n_cluster))

az.plot_kde(np.array(mix))
plt.show()

models = []
data = []
for n in [2, 3, 4]:
    with pm.Model() as model:
        weights = pm.Dirichlet('weights', a=np.ones(n))
        means = pm.Normal('means', mu=np.linspace(mix.min(), mix.max(), n), sigma=10, shape=n)
        sigma = pm.HalfNormal('sigma', sigma=10)
        y_obs = pm.NormalMixture('y_obs', w=weights, mu=means, sigma=sigma, observed=mix)
        trace = pm.sample(1000, return_inferencedata=True, random_seed=111, cores=1, idata_kwargs={'log_likelihood': True})  
        p = pm.sample_posterior_predictive(trace)
        models.append((model, trace))
        data.append(trace)

compared_loocv = az.compare({'model1': data[0], 'model2': data[1], 'model3': data[2]}, method='stacking', ic='loo', scale='deviance')
print(compared_loocv)

compared_waic = az.compare({'model1': data[0], 'model2': data[1], 'model3': data[2]}, method='stacking', ic='waic', scale='deviance')
print(compared_waic)
