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

#3 + 4:

data = {'GRE': 550, 'GPA': 3.5}
p = np.exp(-(trace.posterior['beta0'] + trace.posterior['beta1'] * data['GRE'] + trace.posterior['beta2'] * data['GPA']))
prob_post = 1 / (1+p)
hdi__prob = pm.hdi(prob_post, hdi_prob=0.9)
print(hdi__prob)


dataa = {'GRE': 500, 'GPA': 3.2}
pp = np.exp(-(trace.posterior['beta0'] + trace.posterior['beta1'] * dataa['GRE'] + trace.posterior['beta2'] * dataa['GPA']))
prob__post = 1 / (1 + pp)
hdi__prob2 = pm.hdi(prob__post, hdi_prob2=0.9)
print(hdi__prob2)
