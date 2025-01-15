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

from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
model = BayesianModel()


#defining conditional probability distributions 
#need to input probability when decided 
cpd_weather = TabularCPD(variable='Weather', variable_card=3, values=[[], [], []])
cpd_time_of_day = TabularCPD(variable='Time of Day', variable_card=3, values=[[], [], []])
cpd_historical_congestion_level = TabularCPD(variable='Historical Congestion Level', variable_card=3, values=[[], [], []])
cpd_current_congestion_level = TabularCPD(variable='Current Congestion Level', variable_card=3, values=[[], [], []])