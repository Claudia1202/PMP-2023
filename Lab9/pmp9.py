import pymc as pm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('Admission.csv')

admitted = data['Admission'].values
gre = data['GRE'].values
gpa = data['GPA'].values

with pm.Model() as model:
    beta0 = pm.Normal('beta0', mu=0, sigma=10)
    beta1 = pm.Normal('beta1', mu=0, sigma=10)
    beta2 = pm.Normal('beta2', mu=0, sigma=10)
    pi = pm.math.sigmoid(beta0 + beta1*gre + beta2*gpa)
    obs = pm.Bernoulli('obs', p=pi, observed=admitted)
    trace = pm.sample(2000, tune=1000)

decision_boundary = -trace['beta0'] / trace['beta1']

hdi = pm.hpd(decision_boundary, hdi_prob=0.94)

print(f"Granita de decizie medie: {np.mean(decision_boundary)}")
print(f"Intervalul 94% HDI: {hdi}")

plt.hist(decision_boundary, bins=30, alpha=0.5)
plt.axvline(np.mean(decision_boundary), color='red', lw=2)
plt.axvline(hdi[0], color='black', linestyle='--')
plt.axvline(hdi[1], color='black', linestyle='--')
plt.xlabel('Granita de decizie')
plt.ylabel('Frecventa')
plt.show()
