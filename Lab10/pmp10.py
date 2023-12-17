import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt


dummy_data = np.loadtxt('/content/dummy.csv')
x = dummy_data[:, 0]
y = dummy_data[:, 1]

order = 5
x_poly = np.vstack([x**i for i in range(1, order + 1)])
x_standardized = (x_poly - x_poly.mean(axis=1, keepdims=True)) / x_poly.std(axis=1, keepdims=True)
y_standardized = (y - y.mean()) / y.std()

def create_and_sample_model(sigma):
    with pm.Model() as model:
        a = pm.Normal('alfa', mu=0, sigma=1)
        beta = pm.Normal('beta', mu=0, sigma=sigma, shape=order)
        eps = pm.HalfNormal('eps', 5)
        niu = a + pm.math.dot(beta, x_standardized)
        y_pred = pm.Normal('y_pred', mu=niu, sigma=eps, observed=y_standardized)
        return pm.sample(200, return_inferencedata=True)

idata_sigma_10 = create_and_sample_model(10)
idata_sigma_100 = create_and_sample_model(100)
idata_sigma_array = create_and_sample_model(np.array([10, 0.1, 0.1, 0.1, 0.1]))

plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.scatter(x_standardized[0], y_standardized)
az.plot_posterior(idata_sigma_10)
plt.title('sigma=10')

plt.subplot(3, 1, 2)
plt.scatter(x_standardized[0], y_standardized)
az.plot_posterior(idata_sigma_100)
plt.title('sigma=100')

plt.subplot(3, 1, 3)
plt.scatter(x_standardized[0], y_standardized)
az.plot_posterior(idata_sigma_array)
plt.title('sigma=np.array([10, 0.1, 0.1, 0.1, 0.1])')

plt.xlabel('x')
plt.ylabel('y')
plt.tight_layout()
plt.show()