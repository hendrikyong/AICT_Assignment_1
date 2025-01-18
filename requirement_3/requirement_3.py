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
import pandas as pd

df = pd.read_csv('requirement_3/data.csv')
# print(df.head())
