'''
3.	Traffic Prediction with Bayesian Networks 
You are to model and predict traffic congestion using a Bayesian network based on real-world factors such as weather, time of day, and historical traffic data.

Task Details:
	Construct a Bayesian network with variables such as:
•	Weather (W): Sunny, Rainy, Foggy.
•	Time of Day (T): Morning, Afternoon, Evening.
•	Historical Congestion Level (H): Low, Medium, High.
•	Current Congestion Level (C): Low, Medium, High.
	Use conditional probabilities to predict the likelihood of congestion for a given time and weather condition.

Deliverables:
•	Bayesian network structure and implementation.
•	Inference results for different scenarios.
•	Analysis and discussion of the model’s accuracy and limitations.
'''

'''
Vehicle Count (VC): The number of vehicles passing a specific point per unit time.
Average Vehicle Speed (AS): Average speed of vehicles in a segment of the road.
Road Surface Conditions (RC): Dry, wet
Weather (W): Sunny, Rainy, Foggy
Time of Day (T): Morning, Afternoon, Evening
Day of the Week (D): Weekday, Weekend
Road Incidents (RI): None, Minor, Major
Historical Congestion Level (H): Low, Medium, High
'''

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

#bayesian model
model = BayesianNetwork([
    ("weather", "historical_congestion_level"),
    ("time_of_day", "historical_congestion_level"),
    ("historical_congestion_level", "current_congestion_level")
])


#define cpd
#cpd for weather
cpd_weather = TabularCPD(variable="weather", 
                         variable_card=3, 
                         values=[[0.5], [0.3], [0.2]],
                         state_names={"weather": ["sunny", "rainy", "foggy"]})

#cpd for time_of_day
cpd_time_of_day = TabularCPD(variable="time_of_day", 
                              variable_card=3, 
                              values=[[1/3], [1/3], [1/3]],
                              state_names={"time_of_day": ["morning", "afternoon", "evening"]})

#cpd for historical_congestion_level
cpd_historical_congestion_level = TabularCPD(
    variable="historical_congestion_level",
    variable_card=3,
    values=[
        [0.6, 0.8, 0.7, 0.4, 0.3, 0.35, 0.5, 0.6, 0.55],  # High
        [0.3, 0.15, 0.2, 0.4, 0.5, 0.45, 0.4, 0.3, 0.35],  # Medium
        [0.1, 0.05, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1],    # Low
    ],
    evidence=["weather", "time_of_day"],
    evidence_card=[3, 3],
    state_names={
        "historical_congestion_level": ["high", "medium", "low"],
        "weather": ["sunny", "rainy", "foggy"],
        "time_of_day": ["morning", "afternoon", "evening"]
    }
)

#cpd for current_congestion_level
cpd_current_congestion_level = TabularCPD(
    variable="current_congestion_level",
    variable_card=3,
    values=[
        [0.9, 0.6, 0.3],  
        [0.1, 0.3, 0.4],  
        [0.0, 0.1, 0.3],  
    ],
    evidence=["historical_congestion_level"],
    evidence_card=[3],
    state_names={
        "current_congestion_level": ["high", "medium", "low"],
        "historical_congestion_level": ["high", "medium", "low"]
    }
)

#add CPDs to the model
model.add_cpds(cpd_weather, cpd_time_of_day, cpd_historical_congestion_level, cpd_current_congestion_level)

#model check
assert model.check_model()

#inf
inference = VariableElimination(model)

#get user input
weather_input = input("Enter the weather (sunny, rainy, foggy): ").strip().lower()
time_of_day_input = input("Enter the time of day (morning, afternoon, evening): ").strip().lower()

#validate
if weather_input not in ["sunny", "rainy", "foggy"] or time_of_day_input not in ["morning", "afternoon", "evening"]:
    print("Invalid input. Please enter valid weather and time of day.")
else:
    #else use query to inf the current congestion level
    query_result = inference.query(
        variables=["current_congestion_level"],
        evidence={"weather": weather_input, "time_of_day": time_of_day_input}
    )

    #state names is like high med low 
    state_names = cpd_current_congestion_level.state_names["current_congestion_level"]
    probabilities = query_result.values 

    print("\nPredicted probabilities for Current Congestion Level:")
    for level_name, prob in zip(state_names, probabilities):
        print(f"  {level_name}: {prob:.4f}")
