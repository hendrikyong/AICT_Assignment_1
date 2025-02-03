'''
Deliverables:
•	Bayesian network structure and implementation.
•	Inference results for different scenarios.
•	Analysis and discussion of the model’s accuracy and limitations.

Road Surface Conditions (RC): Dry, wet
Weather (W): Sunny, Rainy, Foggy
Time of Day (T): Morning, Afternoon, Evening
Day of the Week (D): Weekday, Weekend
Road Accidents (RA): None, Minor, Major
Historical Congestion Level (H): Low, Medium, High

W influences RC (because if rainy weather then road surface conditions are wet)
W and RC influences RA (because if theres rainy weather and wet road surface confitions, then road accidents are more likely to happen)
W, RA, T, D influences H (because if theres rainy weather, road accidents, morning time, weekday, then historical congestion level is more likely to be high)
'''

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#define bayesian model structure 
#this model is constructed with a list of directed edges that define the relationships between different variables. 
model = BayesianNetwork([
    ('W', 'RC'),   #for this example, weather influences road conditions
    ('W', 'RA'),   #weather and road conditions influence road accidents
    ('RC', 'RA'),  
    ('W', 'H'),    #weather, road accidents, time of day, and day of the week influence historical congestion level
    ('RA', 'H'),   
    ('T', 'H'),    
    ('D', 'H')     
])


#conditional probability distributions (CPDs) for each node
cpd_T = TabularCPD(variable='T', variable_card=3,
                   values=[[1/3], [1/3], [1/3]])  # P(Morning), P(Afternoon), P(Evening)

cpd_D = TabularCPD(variable='D', variable_card=2,
                   values=[[0.5], [0.5]])  # P(Weekday), P(Weekend)

cpd_W = TabularCPD(variable='W', variable_card=3, values=[[0.6], [0.3], [0.1]]) # Sunny, Rainy, Foggy

cpd_RC = TabularCPD(variable='RC', variable_card=2, 
                    values=[[0.8, 0.2, 0.5],  # Dry: P(Dry | Sunny)... etc
                            [0.2, 0.8, 0.5]],  # Wet: P(Wet | Sunny)... etc
                    evidence=['W'], evidence_card=[3])

cpd_RA = TabularCPD(variable='RA', variable_card=3, 
                    values=[[0.9, 0.7, 1/3, 0.1, 0.1, 0.1],  # P(None | Sunny, Dry), P(None | Sunny, Wet), P(None | Rainy, Dry), P(None | Rainy, Wet), P(None | Foggy, Dry), P(None | Foggy, Wet)
                            [0.08, 0.2, 1/3, 0.4, 0.3, 0.2],  # P(Minor | Sunny, Dry), P(Minor | Sunny, Wet),  P(Minor | Rainy, Dry), P(Minor | Rainy, Wet), P(Minor | Foggy, Dry), P(Minor | Foggy, Wet)
                            [0.02, 0.1, 1/3, 0.5, 0.6, 0.7]], # P(Major | Sunny, Dry), P(Major | Sunny, Wet), P(Major | Rainy, Dry), P(Major | Rainy, Wet), P(Major | Foggy, Dry), P(Major | Foggy, Wet)
                    evidence=['W', 'RC'], evidence_card=[3, 2])

#not possible to have P(None OR Minor OR Major | Rainy, Dry) because it is not possible to have rainy weather and dry road conditions at the same time 

cpd_H = TabularCPD(variable='H', variable_card=3,  # Low, Medium, High
                   values=[
                       # for low
                       [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01,
                        0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01,
                        0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01,
                        0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01, 0.01, 0.01, 0.01,
                        0.7, 0.6, 0.5, 0.4, 0.3, 0.2],
                       # for medium
                       [0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2,
                        0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2,
                        0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2,
                        0.2, 0.3, 0.4, 0.4, 0.5, 0.5, 0.6, 0.6, 0.5, 0.4, 0.3, 0.2,
                        0.2, 0.3, 0.4, 0.4, 0.5, 0.5],
                       # for high
                       [0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.35, 0.49, 0.59, 0.69, 0.79,
                        0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.35, 0.49, 0.59, 0.69, 0.79,
                        0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.35, 0.49, 0.59, 0.69, 0.79,
                        0.1, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.35, 0.49, 0.59, 0.69, 0.79,
                        0.1, 0.1, 0.1, 0.2, 0.2, 0.3]
                   ],
                   evidence=['W', 'RA', 'T', 'D'], evidence_card=[3, 3, 3, 2])


# Add CPDs to the model
model.add_cpds(cpd_T, cpd_D, cpd_W, cpd_RC, cpd_RA, cpd_H)

# Validate the model
model.check_model()
print(model)

inference = VariableElimination(model)
# T = Time of day 0 = Morning, 1 = Afternoon, 2 = Evening
# W = Weather 0 = Sunny, 1 = Rainy, 2 = Foggy

result = inference.query(variables=['H'], evidence={'T': 1, 'W': 0})
congestion_mapping = {0: "Low", 1: "Medium", 2: "High"}

for index, prob in enumerate(result.values):
    print(f"Congestion Level: {congestion_mapping[index]}, Probability: {prob:.2%}")

