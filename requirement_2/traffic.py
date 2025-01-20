import csv
from logic import *

# Define variables for traffic rules
SpeedViolation = Symbol("SpeedViolation")
SignalViolation = Symbol("SignalViolation")
StopSignViolation = Symbol("StopSignViolation")
LaneViolation = Symbol("LaneViolation")
SchoolZoneViolation = Symbol("SchoolZoneViolation")
DuplicateEntry = Symbol("DuplicateEntry")

# Knowledge base to store the rules
knowledge = And()

# Rules
knowledge.add(Implication(Symbol("UrbanSpeedAbove50"), SpeedViolation))
knowledge.add(Implication(Symbol("ExpresswaySpeedAbove90"), SpeedViolation))
knowledge.add(Implication(Symbol("MovingAtRedLight"), SignalViolation))
knowledge.add(Implication(And(Symbol("StopSign"), Symbol("SpeedAbove0")), StopSignViolation))
knowledge.add(Implication(Symbol("WrongLane"), LaneViolation))
knowledge.add(Implication(And(Symbol("SchoolZone"), Symbol("SpeedAbove30")), SchoolZoneViolation))
knowledge.add(Implication(Symbol("HasDuplicates"), DuplicateEntry))


def check_speed_violation(location, speed):
    """Check for speed violations based on location type."""
    if location == "UrbanRoad" and speed > 50:
        return True, "UrbanSpeedAbove50"
    if location == "Expressway" and speed > 90:
        return True, "ExpresswaySpeedAbove90"
    return False, None


def check_signal_violation(signal, speed):
    """Check for red light violations."""
    return signal.lower() == "red" and speed > 0


def check_stop_sign_violation(stop_sign, speed):
    """Check for stop sign violations."""
    return stop_sign == "Yes" and speed > 0


def check_lane_violation(lane, assigned_lane):
    """Check for lane violations."""
    return lane != assigned_lane


def check_school_zone_violation(school_zone, speed):
    """Check for school zone violations."""
    return school_zone == "Yes" and speed > 30


def check_duplicate_entries(records):
    """Check for duplicate entries."""
    seen = set()
    duplicates = set()
    for record in records:
        # Use a tuple of the record values to track duplicates
        entry = tuple(record.values())
        if entry in seen:
            duplicates.add(record["VehicleID"])
        seen.add(entry)
    return duplicates


def process_dataset(file_path):
    """Process the dataset and check for traffic violations."""
    with open(file_path, "r") as file:
        reader = csv.DictReader(file)
        violations = {}
        records = list(reader)

        # Check each record
        for record in records:
            vehicle_id = record["VehicleID"]
            location = record["Location"]
            speed = float(record["Speed (km/h)"])
            signal_status = record["SignalStatus"]
            stop_sign = record["StopSign"]
            lane = record["Lane"]
            assigned_lane = record["AssignedLane"]
            school_zone = record["SchoolZone"]

            if vehicle_id not in violations:
                violations[vehicle_id] = []

            # Check speed violation
            speed_violation, symbol = check_speed_violation(location, speed)
            if speed_violation:
                violations[vehicle_id].append("Speed Violation")
                knowledge.add(Symbol(symbol))

            # Check signal violation
            if check_signal_violation(signal_status, speed):
                violations[vehicle_id].append("Signal Violation")
                knowledge.add(Symbol("MovingAtRedLight"))

            # Check stop sign violation
            if check_stop_sign_violation(stop_sign, speed):
                violations[vehicle_id].append("Stop Sign Violation")
                knowledge.add(Symbol("StopSign"))

            # Check lane violation
            if check_lane_violation(lane, assigned_lane):
                violations[vehicle_id].append("Lane Violation")
                knowledge.add(Symbol("WrongLane"))

            # Check school zone violation
            if check_school_zone_violation(school_zone, speed):
                violations[vehicle_id].append("School Zone Violation")
                knowledge.add(Symbol("SchoolZone"))

        # Check duplicate entries
        duplicate_vehicles = check_duplicate_entries(records)
        for vehicle_id in duplicate_vehicles:
            violations[vehicle_id].append("Duplicate Entry")
            knowledge.add(Symbol("HasDuplicates"))

        # Separate vehicles with and without violations
        vehicles_with_violations = {vehicle: violations[vehicle] for vehicle in violations if violations[vehicle]}
        vehicles_without_violations = [vehicle for vehicle in violations if not violations[vehicle]]

        # Output violations summary
        print("\nViolations Summary:")
        print("Vehicles with Violations:")
        for vehicle, violation_list in vehicles_with_violations.items():
            print(f"  Vehicle {vehicle}: {', '.join(violation_list)}")

        print("\nVehicles without Violations:")
        for vehicle in vehicles_without_violations:
            print(f"  Vehicle {vehicle}")


# File path to the dataset
file_path = "traffic_data.csv"

# Process the dataset
process_dataset(file_path)
