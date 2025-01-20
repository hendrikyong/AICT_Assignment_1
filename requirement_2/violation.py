from itertools import product
from logic import *

# Knowledge base for traffic rules
knowledge = And()

# Load traffic data from CSV
import csv
traffic_data = []
with open("traffic_data_2.csv", "r") as file:
    reader = csv.DictReader(file)
    traffic_data = list(reader)

# Extract unique sets of data from CSV
vehicles = set(row["vehicle_id"] for row in traffic_data)
speeds = set(int(row["speed"]) for row in traffic_data)
locations = set(row["location"] for row in traffic_data)
times = set(int(row["time"]) for row in traffic_data)
signals = set(row["signal"] for row in traffic_data)

# Define symbols for logic
speed_symbols = {
    (vehicle, speed): Symbol(f"{vehicle}_Speed{speed}")
    for vehicle, speed in product(vehicles, speeds)
}
location_symbols = {
    (vehicle, location, time): Symbol(f"{vehicle}_{location}_T{time}")
    for vehicle, location, time in product(vehicles, locations, times)
}
signal_symbols = {
    (vehicle, signal): Symbol(f"{vehicle}_Signal{signal}")
    for vehicle, signal in product(vehicles, signals)
}
violation_symbols = {
    vehicle: Symbol(f"{vehicle}_SpeedViolation")
    for vehicle in vehicles
}
move_symbols = {
    vehicle: Symbol(f"{vehicle}_Moves")
    for vehicle in vehicles
}

# Add traffic rules to the knowledge base
for vehicle in vehicles:
    # Speed Rule: If speed > 60, then SpeedViolation is True
    for speed in speeds:
        speed_symbol = speed_symbols[(vehicle, speed)]
        speed_violation = violation_symbols[vehicle]
        if speed > 60:
            knowledge.add(Implication(speed_symbol, speed_violation))
        else:
            knowledge.add(Implication(speed_symbol, Not(speed_violation)))

    # Signal Rule: If signal is Red and vehicle moves, then SignalViolation is True
    signal_violation = Symbol(f"{vehicle}_SignalViolation")
    for signal in signals:
        signal_symbol = signal_symbols[(vehicle, signal)]
        if signal == "Red":
            knowledge.add(Implication(And(signal_symbol, move_symbols[vehicle]), signal_violation))

    # Location Consistency Rule: A vehicle cannot be in two locations at the same time
    for t in times:
        for loc1, loc2 in product(locations, repeat=2):
            if loc1 != loc2:
                loc1_symbol = location_symbols[(vehicle, loc1, t)]
                loc2_symbol = location_symbols[(vehicle, loc2, t)]
                knowledge.add(Not(And(loc1_symbol, loc2_symbol)))

# Add facts from the CSV data
print("Adding facts from CSV:")
for row in traffic_data:
    vehicle = row["vehicle_id"]
    speed = int(row["speed"])
    location = row["location"]
    time = int(row["time"])
    signal = row["signal"]

    # Add speed, location, signal, and movement facts
    knowledge.add(speed_symbols[(vehicle, speed)])
    knowledge.add(location_symbols[(vehicle, location, time)])
    knowledge.add(signal_symbols[(vehicle, signal)])
    knowledge.add(move_symbols[vehicle])  # Assume all vehicles can move for now

    # Debug facts
    print(f"Added: {speed_symbols[(vehicle, speed)]}")
    print(f"Added: {location_symbols[(vehicle, location, time)]}")
    print(f"Added: {signal_symbols[(vehicle, signal)]}")
    print(f"Added: {move_symbols[vehicle]}")

# Debug: Print the entire knowledge base
print("\nKnowledge Base:")
for conjunct in knowledge.conjuncts:
    print(conjunct)

# Perform inference to check for violations
print("\nInference Results:")
for vehicle in vehicles:
    speed_violation = violation_symbols[vehicle]
    signal_violation = Symbol(f"{vehicle}_SignalViolation")
    print(f"Checking Speed Violation for {vehicle}: {model_check(knowledge, speed_violation)}")
    print(f"Checking Signal Violation for {vehicle}: {model_check(knowledge, signal_violation)}")
