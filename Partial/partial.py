from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import pymc as pm
import arviz as az

#ex1

#p(j1-stema)=2/3
#p(j0-stema)=1/2

j0 = stats.bernoulli.rvs(p=0.5, size=10000)
j1 = stats.bernoulli.rvs(p=0.66, size=10000)




#b)
# J1 = J1-castiga
# J0 = J0-castiga

#Sansele de castig ale lui J0 sunt determinate de prima aruncare(daca incepe sau nu primul) si de aruncarea cu moneda proprie(nemasluita)
#Sansele de castig ale lui J1 sunt determinate de prima aruncare(daca incepe sau nu primul) si de aruncarea cu moneda proprie(masluita)

bayesian_model = BayesianNetwork([('MonedaJ1', 'J1'), ('MonedaJ0', 'J0'), ('J0-primul', 'J0'), ('J1-primul', 'J1')])
#cream nodurile retelei

cpd_MonedaJ1 = TabularCPD(variable='MonedaJ1', variable_card=2, values=[[0.66], [0.33]])  #moneda masluita: 0.66-0.33
cpd_MonedaJ0 = TabularCPD(variable='MonedaJ0', variable_card=2, values=[[0.5], [0.5]])    #moneda nemasluita: 0.5-0.5
cpd_J1primul = TabularCPD(variable='J1-primul', variable_card=2, values=[[0.5], [0.5]])
cpd_J0primul = TabularCPD(variable='J0-primul', variable_card=2, values=[[0.5], [0.5]])

cpd_J0 = TabularCPD(variable='J0', variable_card=2, values=[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5]],  evidence=['MonedaJ0', 'J0-primul'], evidence_card=[2, 2])
cpd_J1 = TabularCPD(variable='J1', variable_card=2, values=[[0.66, 0.66,  ], [0.33, 0.5]],  evidence=['MonedaJ1', 'J1-primul'], evidence_card=[2, 2])


#adaugam nodurile la retea
bayesian_model.add_cpds(cpd_MonedaJ1, cpd_MonedaJ0, cpd_J1, cpd_J0, cpd_J0primul, cpd_J1primul)
assert bayesian_model.check_model()

#desenul retelei
pos = nx.circular_layout(bayesian_model)
nx.draw(bayesian_model, pos=pos, with_labels=True, node_size=4000, font_weight='bold', node_color='skyblue')
plt.show()

#c)
#prob a posteriori ca j0, respectiv j1 sa fi fost primul stiind ca j0, resp j1 a castigat
infer = VariableElimination(bayesian_model)
prob_j0_primul = infer.query(variables=['J0-primul'], evidence={'J0': 1})
print(prob_j0_primul)

prob_j1_primul = infer.query(variables=['J1-primul'], evidence={'J1': 1})
print(prob_j1_primul)

# #ex2

timpi = stats.poisson.rvs(10, size=100)

with pm.Model() as model:


  mu = pm.Uniform('mu', lower=5, upper=10)
  sigma = pm.HalfNormal('sigma', sigma=10)
  #am modelat timpul mediu de asteptare cu distributia normala
  timp_asteptare = pm.Normal('timp_asteptare', mu=mu, sigma=sigma, observed=timpi)
  idata = pm.sample(1000, return_inferencedata=True)
  pm.sample_posterior_predictive(idata, model=model, extend_inferencedata=True)
  ax1 = az.plot_ppc(idata, num_pp_samples=100, figsize=(12, 6), mean=False)



