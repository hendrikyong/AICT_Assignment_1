import csv
from logic import *
from datetime import datetime

csv_path = "traffic_data_2.csv"
data = []
with open(csv_path, "r") as file:
    reader = csv.DictReader(file)
    data = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]  # Strip whitespace

if data:
    print("Columns in data:", data[0].keys())

for row in data:
    vehicle = row.get("Vehicle", None)
    speed = row.get("Speed", None)
    location = row.get("Location", None)
    timestamp = row.get("Timestamp", None)
    if not vehicle or not speed or not location or not timestamp:
        print(f"Corrupt data: Missing fields in row {row}")
    try:
        int(speed)
        datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        print(f"Corrupt data: Invalid speed or timestamp in row {row}")

for row in data:
    vehicle = row["Vehicle"]
    speed = int(row.get("Speed", -1))
    if speed < 0 or speed > 200: 
        print(f"Invalid speed: {vehicle} has an unrealistic speed of {speed} km/h.")

vehicles = [row["Vehicle"] for row in data]
speed_symbols = []
bus_lane_violation_symbols = []
red_light_violation_symbols = []
erp_violation_symbols = []
school_zone_violation_symbols = []

for row in data:
    vehicle = row["Vehicle"]
    speed = int(row["Speed"])
    timestamp = row.get("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    is_school_zone = row.get("SchoolZone", "0") == "1"
    vehicle_type = row.get("VehicleType", "Car")

    if speed > 60:
        speed_symbols.append(Symbol(f"{vehicle}_OverSpeed"))

    hour = int(timestamp.split(" ")[1].split(":")[0])
    if vehicle_type != "Bus" and row.get("BusLane", "0") == "1" and (7 <= hour < 9 or 17 <= hour < 20):
        bus_lane_violation_symbols.append(Symbol(f"{vehicle}_BusLaneViolation"))

    if row.get("RedLight", "0") == "1":
        red_light_violation_symbols.append(Symbol(f"{vehicle}_RedLightViolation"))

    erp_balance = float(row.get("ERPBalance", "0"))
    erp_charge = float(row.get("ERPCharge", "0"))
    if erp_balance < erp_charge:
        erp_violation_symbols.append(Symbol(f"{vehicle}_ERPViolation"))

    if is_school_zone and speed > 40:
        school_zone_violation_symbols.append(Symbol(f"{vehicle}_SchoolZoneViolation"))

# knowledge base
traffic_rules = And()

# speed > 60
for row in data:
    vehicle = row["Vehicle"]
    speed = int(row["Speed"])
    if speed > 60:
        traffic_rules.add(Symbol(f"{vehicle}_OverSpeed"))

# bus lane during restricted hours
for row in data:
    vehicle = row["Vehicle"]
    vehicle_type = row.get("VehicleType", "Car")
    timestamp = row.get("Timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    hour = int(timestamp.split(" ")[1].split(":")[0])
    if vehicle_type != "Bus" and row.get("BusLane", "0") == "1" and (7 <= hour < 9 or 17 <= hour < 20):
        traffic_rules.add(Symbol(f"{vehicle}_BusLaneViolation"))

for row in data:
    vehicle = row["Vehicle"]
    if row.get("RedLight", "0") == "1":
        traffic_rules.add(Symbol(f"{vehicle}_RedLightViolation"))

# balance less than charge
for row in data:
    vehicle = row["Vehicle"]
    erp_balance = float(row.get("ERPBalance", "0"))
    erp_charge = float(row.get("ERPCharge", "0"))
    if erp_balance < erp_charge:
        traffic_rules.add(Symbol(f"{vehicle}_ERPViolation"))

# speed > 40
for row in data:
    vehicle = row["Vehicle"]
    speed = int(row["Speed"])
    is_school_zone = row.get("SchoolZone", "0") == "1"
    if is_school_zone and speed > 40:
        traffic_rules.add(Symbol(f"{vehicle}_SchoolZoneViolation"))

# Inference to detect violations
print("Detected Violations:")

# Check for speed violations
for symbol in speed_symbols:
    if model_check(traffic_rules, symbol):
        print(f"Violation detected: {symbol}")

# Check for bus lane violations
for symbol in bus_lane_violation_symbols:
    if model_check(traffic_rules, symbol):
        print(f"Violation detected: {symbol}")

# Check for red light violations
for symbol in red_light_violation_symbols:
    if model_check(traffic_rules, symbol):
        print(f"Violation detected: {symbol}")

# Check for ERP violations
for symbol in erp_violation_symbols:
    if model_check(traffic_rules, symbol):
        print(f"Violation detected: {symbol}")

# Check for school zone violations
for symbol in school_zone_violation_symbols:
    if model_check(traffic_rules, symbol):
        print(f"Violation detected: {symbol}")
