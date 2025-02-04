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

cpd_W = TabularCPD(variable='W', variable_card=3,
                   values=[[0.5], [0.35], [0.15]],  # Sunny, Rainy, Foggy
                   state_names={'W': ['Sunny', 'Rainy', 'Foggy']})

cpd_RC = TabularCPD(variable='RC', variable_card=2,
                     values=[[0.95, 0.3, 0.4],  # Good
                             [0.05, 0.7, 0.6]],  # Bad
                     evidence=['W'], evidence_card=[3],
                     state_names={'RC': ['Good', 'Bad'], 'W': ['Sunny', 'Rainy', 'Foggy']})

cpd_RA = TabularCPD(variable='RA', variable_card=2,
                     values=[[0.98, 0.7, 0.8, 0.5, 0.7, 0.4],  # No Accident
                             [0.02, 0.3, 0.2, 0.5, 0.3, 0.6]],  # Accident
                     evidence=['W', 'RC'], evidence_card=[3, 2],
                     state_names={'RA': ['No Accident', 'Accident'],
                                  'W': ['Sunny', 'Rainy', 'Foggy'],
                                  'RC': ['Good', 'Bad']})
cpd_T = TabularCPD(variable='T', variable_card=3,
                   values=[[0.3], [0.4], [0.3]],  # Morning, Afternoon, Evening
                   state_names={'T': ['Morning', 'Afternoon', 'Evening']})

cpd_D = TabularCPD(variable='D', variable_card=2,
                   values=[[5/7], [2/7]],  # Weekday, Weekend
                   state_names={'D': ['Weekday', 'Weekend']})

import numpy as np

# Define states
weather_states = ["Sunny", "Rainy", "Foggy"]
accident_states = ["No Accident", "Accident"]
time_states = ["Morning", "Afternoon", "Evening"]
day_states = ["Weekday", "Weekend"]

# Probabilities storage
prob_low, prob_medium, prob_high = [], [], []

# Adjust congestion probabilities for specific weekend afternoon conditions
for w in weather_states:
    for ra in accident_states:
        for t in time_states:
            for d in day_states:
                # Base probabilities
                if w == "Sunny":
                    low, medium, high = 0.9, 0.08, 0.02
                elif w == "Rainy":
                    low, medium, high = 0.4, 0.4, 0.2
                else:  # Foggy
                    low, medium, high = 0.25, 0.50, 0.25 

                if ra == "Accident":
                    low *= 0.2  
                    medium *= 1.3  
                    high *= 2.5  

                if t == "Morning":
                    low *= 0.5
                    medium *= 1.2
                    high *= 2.0
                elif t == "Evening":
                    low *= 0.6
                    medium *= 1.2 
                    high *= 1.5  

                if d == "Weekday":
                    low *= 0.75
                    medium *= 1.1
                    high *= 1.4
                elif d == "Weekend" and t == "Afternoon" and ra == "No Accident" and w == "Sunny":
                    low *= 0.5  
                    medium *= 1.5  
                    high *= 2.0  

                total = low + medium + high
                low /= total
                medium /= total
                high /= total

                prob_low.append(low)
                prob_medium.append(medium)
                prob_high.append(high)

# Convert to NumPy arrays
prob_values = np.array([prob_low, prob_medium, prob_high])

cpd_H = TabularCPD(
    variable='H', variable_card=3,
    values=prob_values,
    evidence=['W', 'RA', 'T', 'D'],
    evidence_card=[3, 2, 3, 2],
    state_names={'H': ['Low', 'Medium', 'High'],
                 'W': ['Sunny', 'Rainy', 'Foggy'],
                 'RA': ['No Accident', 'Accident'],
                 'T': ['Morning', 'Afternoon', 'Evening'],
                 'D': ['Weekday', 'Weekend']}
)

# Add CPDs to the model
model.add_cpds(cpd_W, cpd_RC, cpd_RA, cpd_T, cpd_D, cpd_H)

# Check if model is valid
assert model.check_model(), "The model is not valid!"

inference = VariableElimination(model)
congestion_mapping = {0: "Low", 1: "Medium", 2: "High"}


test_cases = [
    {"T": "Morning", "W": "Rainy"},  
    {"T": "Afternoon", "W": "Sunny"},  
    {"T": "Evening", "W": "Foggy"},  
    {"T": "Morning", "W": "Rainy", "RA": "Accident", "D": "Weekday"},  
    {"T": "Afternoon", "W": "Sunny", "RA": "No Accident", "D": "Weekday"},
    {"T": "Afternoon", "W": "Sunny", "RA": "No Accident", "D": "Weekend"},
]

for i, evidence in enumerate(test_cases, 1):
    result = inference.query(variables=['H'], evidence=evidence)
    
    print(f"\nTest Case {i}: {evidence}")
    for index, prob in enumerate(result.values):
        print(f"  Congestion Level: {congestion_mapping[index]}, Probability: {prob:.2%}")
