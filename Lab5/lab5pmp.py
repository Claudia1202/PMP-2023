import pymc3 as pm
import pandas as pd

data = pd.read_csv('trafic.csv')

minute = data['minut']
nr_masini = data['nr. masini'].values

ore_modif = [7, 8, 16, 19]

with pm.Model() as model:
    lambd = pm.Exponential('lambda', 1.2)

    traffic = pm.Poisson('traffic', mu=lambd, obs=nr_masini)

    d_lambd = pm.Normal("d_lambd", mu=0, sigma=1, shape=len(ore_modif))
    
    for i, hour in enumerate(ore_modif):
        trafic_mediu =  pm.math.set_subtensor(nr_masini[i], d_lambd[ore_modif.index(minute // 60)])

    trafic_observat = pm.Poisson("trafic_obs", trafic_mediu, obs=nr_masini)


with model:
    
    trace = pm.sample(1000, tune=1000)

pm.plot_trace(trace)
