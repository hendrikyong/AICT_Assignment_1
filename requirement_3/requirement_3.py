#import necessary library for Bayesian Networks
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import math

#create a Bayesian Network model with nodes and edges
model = BayesianNetwork([
    ('W', 'RC'),  #weather (W) affects road condition (RC)
    ('W', 'RA'),  #weather (W) affects the likelihood of a road accident (RA)
    ('RC', 'RA'), #road condition (RC) also affects the likelihood of a road accident (RA)
    ('W', 'H'),   #weather (W) affects traffic congestion (H)
    ('RA', 'H'),  #road accidents (RA) contribute to traffic congestion (H)
    ('T', 'H'),   #time of day (T) influences traffic congestion (H)
    ('D', 'H')    #day of the week (D) affects traffic congestion (H)
])


# Create a NetworkX graph from the Bayesian Network
graph = nx.DiGraph()
graph.add_edges_from(model.edges())

# Draw the graph
plt.figure(figsize=(8, 6))
pos = nx.spring_layout(graph)  # Layout for better positioning
nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=3000, edge_color='gray', font_size=12, font_weight='bold', arrows=True)

# Show the diagram
plt.title("Bayesian Network Structure", fontsize=14)
plt.show()

#define the conditional probability distribution (CPD) for weather (W)
cpd_W = TabularCPD(variable='W', variable_card=3,
                   values=[[0.5], [0.35], [0.15]], #probabilities for Sunny, Rainy, and Foggy
                   state_names={'W': ['Sunny', 'Rainy', 'Foggy']})

#define the CPD for road condition (RC) given weather (W)
cpd_RC = TabularCPD(variable='RC', variable_card=2,
                     values=[[0.95, 0.3, 0.4],  #P(RC=Good | W=Sunny), P(RC=Good | W=Rainy), P(RC=Good | W=Foggy)
                             [0.05, 0.7, 0.6]], #P(RC=Bad | W=Sunny), P(RC=Bad | W=Rainy), P(RC=Bad | W=Foggy)
                     evidence=['W'], evidence_card=[3],
                     state_names={'RC': ['Good', 'Bad'], 'W': ['Sunny', 'Rainy', 'Foggy']})

#define the CPD for road accident (RA) given weather (W) and road condition (RC)
cpd_RA = TabularCPD(variable='RA', variable_card=2,
                     values=[[0.98, 0.8, 0.7, 0.3, 0.6, 0.4],   #P(RA=No Accident | W=Sunny, RC=Good), P(RA=No Accident | W=Sunny, RC=Bad), #P(RA=No Accident | W=Rainy, RC=Good), P(RA=No Accident | W=Rainy, RC=Bad),
 #P(RA=No Accident | W=Foggy, RC=Good), P(RA=No Accident | W=Foggy, RC=Bad)
                             [0.02, 0.2, 0.3, 0.7, 0.4, 0.6]],  # P(RA=Accident | W=Sunny, RC=Good), P(RA=Accident | W=Sunny, RC=Bad), #P(RA=Accident | W=Rainy, RC=Good), P(RA=Accident | W=Rainy, RC=Bad), #P(RA=Accident | W=Foggy, RC=Good), P(RA=Accident | W=Foggy, RC=Bad)
                     evidence=['W', 'RC'], evidence_card=[3, 2],
                     state_names={'RA': ['No Accident', 'Accident'],
                                  'W': ['Sunny', 'Rainy', 'Foggy'],
                                  'RC': ['Good', 'Bad']})

#define the CPD for time of day (T)
cpd_T = TabularCPD(variable='T', variable_card=3,
                   values=[[0.3], [0.4], [0.3]],  #probabilities for Morning, Afternoon, and Evening
                   state_names={'T': ['Morning', 'Afternoon', 'Evening']})

#define the CPD for day of the week (D)
cpd_D = TabularCPD(variable='D', variable_card=2,
                   values=[[5/7], [2/7]],  #probabilities for Weekday and Weekend
                   state_names={'D': ['Weekday', 'Weekend']})

#====================================================================================
# Define the CPD for traffic congestion (H) based on multiple factors
#====================================================================================

#define possible states for different factors that affect traffic congestion
weather_states = ["Sunny", "Rainy", "Foggy"]
accident_states = ["No Accident", "Accident"]
time_states = ["Morning", "Afternoon", "Evening"]
day_states = ["Weekday", "Weekend"]

#lists to store probability values for each congestion level
prob_low, prob_medium, prob_high = [], [], []

#loop through all possible combinations of weather, accident, time, and day states
for w in weather_states: #loop through different weather conditions
    for ra in accident_states: #loop through different accident conditions
        for t in time_states: #loop through different times of day
            for d in day_states: #loop through different days of the week
                #default probabilities low, med, high congestion levels based on weather conditions
                base_probabilities = {
                    "Sunny": [0.9, 0.08, 0.02],
                    "Rainy": [0.3, 0.4, 0.3],
                    "Foggy": [0.25, 0.50, 0.25]
                }
                #extract base probabilities based on current weather condition
                low, medium, high = base_probabilities[w]

                #define adjustment factors for different conditions
                #these factors are used to adjust the base probabilities based on the current conditions
                accident_factor = {"No Accident": [1.0, 1.0, 1.0], #no change to probability if there is no accident 
                                   "Accident": [0.2, 1.3, 2.5] #increase probability of Medium and High congestion if there is an accident
                                   }
                time_factor = {"Morning": [0.5, 1.2, 2.0], #higher probability of Medium and High congestion in the morning
                               "Afternoon": [1.0, 1.0, 1.0], #no adjustment for afternoon
                               "Evening": [0.6, 1.2, 1.5] #higher probability of Medium and High congestion in the evening
                               }
                day_factor = {"Weekday": [0.75, 1.1, 1.4], #higher probability of Medium and High congestion on weekdays
                              "Weekend": [0.9, 1.05, 1.2]  #smoother traffic on weekends
                              }

                #additional adjustment if it's a weekend afternoon
                #if weekend afternoon, slightly increase Medium and High congestion probabilities due to people going out
                if d == "Weekend" and t == "Afternoon":
                    day_factor["Weekend"] = [0.5, 1.6, 1.8]  # Slightly increase Medium and High

                #apply accident adjustment factors based on base probability 
                low *= accident_factor[ra][0] #adjust Low congestion probability based on accident presence
                medium *= accident_factor[ra][1] #adjust Medium congestion probability based on accident presence
                high *= accident_factor[ra][2] #adjust High congestion probability based on accident presence

                #apply time-of-day adjustment
                low *= time_factor[t][0] #adjust Low congestion probability based on time of day
                medium *= time_factor[t][1] #adjust Medium congestion probability based on time of day
                high *= time_factor[t][2] #adjust High congestion probability based on time of day

                #apply weekday/weekend adjustment
                low *= day_factor[d][0] #adjust Low congestion probability based on day of the week
                medium *= day_factor[d][1] #adjust Medium congestion probability based on day of the week
                high *= day_factor[d][2] #adjust High congestion probability based on day of the week

                #normalize probabilities so they sum to 1 (important for probability distributions)
                total = low + medium + high #calculate sum of adjusted probability
                low /= total #normalize Low congestion probability
                medium /= total #normalize Medium congestion probability
                high /= total   #normalize High congestion probability

                #store and append the calculated probabilities to the respective lists 
                prob_low.append(low)
                prob_medium.append(medium)
                prob_high.append(high)

#convert the probability lists into a numpy array
prob_values = np.array([prob_low, prob_medium, prob_high])

#define the CPD for traffic congestion (H)
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

#add the Conditional Probability Distributions (CPDs) to the Bayesian Network model
model.add_cpds(cpd_W, cpd_RC, cpd_RA, cpd_T, cpd_D, cpd_H)

#validate the model to ensure it follows Bayesian network properties
assert model.check_model(), "The model is not valid!"

#create an inference object using Variable Elimination for probabilistic reasoning
inference = VariableElimination(model)

#mapping congestion levels from numerical indices to readable labels so the display later will show low med high instead of 012
congestion_mapping = {0: "Low", 1: "Medium", 2: "High"}

#define a set of test cases with different evidence (conditions) to query the model
test_cases = [
    {"T": "Morning", "W": "Rainy"},  
    {"T": "Afternoon", "W": "Sunny"},  
    {"T": "Evening", "W": "Foggy"},  
    {"T": "Morning", "W": "Rainy", "RA": "Accident", "D": "Weekday"},  
    {"T": "Afternoon", "W": "Sunny", "RA": "No Accident", "D": "Weekday"},
    {"T": "Afternoon", "W": "Sunny", "RA": "No Accident", "D": "Weekend"},
]

#query the model for each test case and display the results
for i, evidence in enumerate(test_cases, 1):
    result = inference.query(variables=['H'], evidence=evidence)
    #display the results for each test case
    print(f"\nTest Case {i}: {evidence}")
    for index, prob in enumerate(result.values):
        print(f"  Congestion Level: {congestion_mapping[index]}, Probability: {prob:.2%}")




#====================================================================================
# Advanced Simulated Annealing
#====================================================================================
def simulated_annealing(routes, distance_matrix, max_iterations=1000, initial_temp=100, cooling_rate=0.99):
    """
    Optimizes vehicle routes using Simulated Annealing to minimize total travel time.
    :param routes: List of initial vehicle routes (each route is a list of nodes/locations).
    :param distance_matrix: A matrix representing travel time between locations.
    :param max_iterations: Maximum number of iterations.
    :param initial_temp: Starting temperature for the annealing process.
    :param cooling_rate: Rate at which temperature decreases.
    :return: Optimized routes and corresponding total travel time.
    """
    def total_travel_time(routes):
        """Computes the total travel time for given routes."""
        total_time = 0
        for route in routes:
            for i in range(len(route) - 1):
                total_time += distance_matrix[route[i]][route[i + 1]] #add time between consecutive locations
        return total_time
    
    #initialize the current solution (routes) and the best solution found so far
    current_routes = routes.copy()
    best_routes = routes.copy()

    #calculate the total travel time (cost) of the initial routes
    current_cost = total_travel_time(current_routes)
    best_cost = current_cost

    #set initial temperature for simulated annealing
    temperature = initial_temp
    
    #main loop: iterate for a maximum of max_iterations times
    for iteration in range(max_iterations):
        #make a copy of the current routes to explore a new solution
        new_routes = [route.copy() for route in current_routes]
        #select a random route to modify
        route_idx = random.randint(0, len(new_routes) - 1)
        #swap two locations within the selected route if it has at least two locations
        if len(new_routes[route_idx]) > 2:
            i, j = random.sample(range(len(new_routes[route_idx])), 2)
            #swapping route position
            new_routes[route_idx][i], new_routes[route_idx][j] = new_routes[route_idx][j], new_routes[route_idx][i]
        
        #calculate the total travel time of the modified routes
        new_cost = total_travel_time(new_routes)
        #determine the difference between the new solution's cost and the current solution's cost
        cost_difference = new_cost - current_cost
        
        #if the new solution is better (lower cost), accept it
        #otherwise, accept it with a probability that decreases as temperature drops
        if cost_difference < 0 or random.uniform(0, 1) < math.exp(-cost_difference / temperature):
            current_routes = new_routes
            current_cost = new_cost
            #update the best solution found so far if the new solution is better
            if new_cost < best_cost:
                best_routes = new_routes
                best_cost = new_cost
        
        #reduce the temperature by the cooling rate to gradually decrease exploration
        temperature *= cooling_rate
    
    #return the best routes and their corresponding total travel time
    return best_routes, best_cost

#distance matrix representing travel time between locations (symmetric matrix)
distance_matrix = np.array([
    [0, 10, 15, 20], #row then column so to travel from location 3 to location 1 the way to read this is 
    #row 3 index 1 in this case
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0] #row 3, index 1 is 25 
])

#define initial vehicle routes (list of lists, each representing a sequence of stops)
#-vehicle 1 follows the route [0 → 1 → 2 → 3]
#-vehicle 2 follows the route [3 → 2 → 1 → 0]
#each vehicle's route is represented as a list of locations (or stops).
#goal is to optimize these routes to minimize total travel time, adjusting the order of stops if needed.
initial_routes = [[0, 1, 2, 3], 
                  [3, 2, 1, 0],
                  [0, 2, 1, 3]]

#calculate the initial total travel time for the routes
initial_time = sum(
        sum(distance_matrix[route[i]][route[i + 1]] for i in range(len(route) - 1))
    for route in initial_routes
)

#run the Simulated Annealing optimization algorithm to find better routes
optimized_routes, optimized_time = simulated_annealing(initial_routes, distance_matrix)


#print the results before and after optimization
print("\nBefore Optimization:")
print("Initial Routes:", initial_routes)
print("Total Travel Time Before Optimization:", initial_time)

print("\nAfter Optimization:")
print("Optimized Routes:", optimized_routes)
print("Total Travel Time After Optimization:", optimized_time)

