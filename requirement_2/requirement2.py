import csv
from datetime import datetime
import itertools
import logic

DEBUG = False

def get_traffic_rules():
    """
    violations include:
      1. RedLightViolation
      2. SpeedingViolation
      3. SchoolZoneSpeedingViolation
      4. ResidentialSpeedingViolation
      5. ExpresswaySpeedingViolation
      6. BusLaneViolation
      7. IllegalParkingViolation
      8. ERPViolation
      9. UnauthorizedUTurnViolation

      inconsistencies include:
      1. unusual speeds (extremely high/negative)
      2. vehicles in the different location at the same time
    """

    # traffic rules for each location
    rules_by_location = {
        "Tuas Expressway": [
            ("ExpresswaySpeedingViolation",
             "On Tuas Expressway, if speed exceeds 90 km/h then violation",
             "(~expressway OR speed_within_expressway OR ExpresswaySpeedingViolation)"),
            ("ERPViolation",
             "On Tuas Expressway, if ERP is active and ERP balance is below required charge then violation",
             "(~erp_active OR ~erp_charge_violation OR ERPViolation)")
        ],
        "Orchard Road": [
            ("BusLaneViolation",
             "On Orchard Road, if the vehicle is not a bus and the current time is within enforced bus lane hours then violation",
             "(is_bus OR ~in_bus_lane_hours OR BusLaneViolation)"),
            ("IllegalParkingViolation",
             "On Orchard Road, if the vehicle is parked for more than 5 minutes then violation",
             "(~illegal_parking OR IllegalParkingViolation)")
        ],
        "CBD": [
            ("RedLightViolation",
             "In the CBD, if the traffic light is red and the vehicle is moving then violation",
             "(~red OR ~speed_above_5 OR RedLightViolation)"),
            ("UnauthorizedUTurnViolation",
             "In the CBD, if the vehicle makes a U-turn then violation",
             "(~made_uturn OR UnauthorizedUTurnViolation)")
        ],
        "School Street": [
            ("SchoolZoneSpeedingViolation",
             "On School Street, if in a school zone during enforcement hours and speed > 40 km/h then violation",
             "(~school_zone OR ~in_school_hours OR ~speed_above_40 OR SchoolZoneSpeedingViolation)"),
            ("PedestrianSafetyViolation",
             "On School Street, if pedestrians are detected and speed exceeds 30 km/h then violation",
             "(~pedestrians_present OR ~speed_above_30 OR PedestrianSafetyViolation)")
        ],
        "City Link": [
            ("ResidentialSpeedingViolation",
             "On City Link, if speed exceeds 50 km/h then violation",
             "(~residential OR speed_within_residential OR ResidentialSpeedingViolation)")
        ],
        "Jurong East": [
            ("HeavyVehicleViolation",
             "On Jurong East, if the vehicle is a heavy vehicle and operating during restricted hours then violation",
             "(~is_heavy_vehicle OR ~restricted_hours OR HeavyVehicleViolation)")
        ],
        "Changi Airport": [
            ("UnauthorizedParkingViolation",
             "At Changi Airport, if vehicle is parked outside designated parking zones then violation",
             "(~illegal_parking OR UnauthorizedParkingViolation)")
        ],
        "ALL": [
            ("SpeedingViolation",
             "If the vehicle's speed exceeds 60 km/h then violation",
             "(~speed_above_60 OR SpeedingViolation)"),
            ("IllegalParkingViolation",
             "If the vehicle has been parked for more than 5 minutes then violation",
             "(~illegal_parking OR IllegalParkingViolation)"),
            ("ERPViolation",
             "If ERP is active and ERP balance is below the required charge then violation",
             "(~erp_active OR ~erp_charge_violation OR ERPViolation)"),
            ("UnrealisticSpeedViolation",
             "If vehicle speed exceeds 150 km/h in non-expressway areas then violation",
             "(~residential OR ~speed_above_150 OR UnrealisticSpeedViolation)")
        ]
    }
    return rules_by_location


def nnf(sentence):
    # convert into negated normal form (like after demorgan's law)
    if isinstance(sentence, logic.Symbol):
        return sentence
    elif isinstance(sentence, logic.Not):
        operand = sentence.operand
        if isinstance(operand, logic.Symbol):
            return sentence  # already in NNF
        elif isinstance(operand, logic.Not):
            return nnf(operand.operand)  # ¬(¬A) becomes A.
        elif isinstance(operand, logic.And):
            return nnf(logic.Or(*[logic.Not(s) for s in operand.conjuncts]))
        elif isinstance(operand, logic.Or):
            return nnf(logic.And(*[logic.Not(s) for s in operand.disjuncts]))
        else:
            raise Exception("nnf: Unhandled operand type in Not.")
    elif isinstance(sentence, logic.And):
        return logic.And(*[nnf(s) for s in sentence.conjuncts])
    elif isinstance(sentence, logic.Or):
        return logic.Or(*[nnf(s) for s in sentence.disjuncts])
    elif isinstance(sentence, logic.Implication):
        return nnf(logic.Or(logic.Not(sentence.antecedent), sentence.consequent))
    elif isinstance(sentence, logic.Biconditional):
        return nnf(logic.And(logic.Implication(sentence.left, sentence.right),
                             logic.Implication(sentence.right, sentence.left)))
    else:
        raise Exception("nnf: Unhandled sentence type")


def complement_literal(literal):
    return literal[1:] if literal.startswith("~") else "~" + literal

def resolve_clause(ci, cj):
    resolvents = set()
    for literal in ci:
        comp = complement_literal(literal)
        if comp in cj:
            resolvent = (ci - {literal}) | (cj - {comp})
            resolvents.add(frozenset(resolvent))
    return resolvents

def to_clause(sentence):
    sentence = nnf(sentence)
    if isinstance(sentence, logic.Symbol):
        return frozenset({sentence.name})
    elif isinstance(sentence, logic.Not):
        return frozenset({f"~{sentence.operand.name}"})
    elif isinstance(sentence, logic.Or):
        lits = set()
        for disj in sentence.disjuncts:
            lits |= to_clause(disj)
        return frozenset(lits)
    else:
        raise Exception("to_clause: Unexpected sentence type (expected Symbol, Not, or Or)")

def resolution_model_check(knowledge, query):
    # convert knowledge base into a set of clauses
    # convert negated query into clause and add
    # apply resolution until empty or no new clauses
    clauses = {to_clause(s) for s in knowledge}
    neg_query = nnf(logic.Not(query))
    if isinstance(neg_query, logic.And):
        for conjunct in neg_query.conjuncts:
            clauses.add(to_clause(conjunct))
    else:
        clauses.add(to_clause(neg_query))
    
    new = set()
    while True:
        pairs = list(itertools.combinations(clauses, 2))
        for (Ci, Cj) in pairs:
            resolvents = resolve_clause(Ci, Cj)
            if frozenset() in resolvents:
                return True  # empty clause
            new = new.union(resolvents)
        if new.issubset(clauses):
            return False  # no new clause
        clauses = clauses.union(new)

def resolution_inference(knowledge, query):
    return resolution_model_check(knowledge, query)


def build_kb(vehicle):
    kb = set()
    try:
        t = datetime.strptime(vehicle["timestamp"].strip(), "%H:%M").time()
    except Exception:
        t = None

    # bus lane hours, non buses cannot in bus lane (7:30-9:30, 17:00-20:00)
    if t is not None:
        bus_start_morn = datetime.strptime("07:30", "%H:%M").time()
        bus_end_morn   = datetime.strptime("09:30", "%H:%M").time()
        bus_start_even = datetime.strptime("17:00", "%H:%M").time()
        bus_end_even   = datetime.strptime("20:00", "%H:%M").time()
        in_bus_lane_hours = ((bus_start_morn <= t <= bus_end_morn) or 
                             (bus_start_even <= t <= bus_end_even))
    else:
        in_bus_lane_hours = False

    # school zone hours, vehicles shouldn't go past 60kmh (6:30-7:45, 12:00-14:30, 18:00-19:00)
    if t is not None and vehicle["school_zone"]:
        school_start_1 = datetime.strptime("06:30", "%H:%M").time()
        school_end_1   = datetime.strptime("07:45", "%H:%M").time()
        school_start_2 = datetime.strptime("12:00", "%H:%M").time()
        school_end_2   = datetime.strptime("14:30", "%H:%M").time()
        school_start_3 = datetime.strptime("18:00", "%H:%M").time()
        school_end_3   = datetime.strptime("19:00", "%H:%M").time()
        in_school_hours = ((school_start_1 <= t <= school_end_1) or
                           (school_start_2 <= t <= school_end_2) or
                           (school_start_3 <= t <= school_end_3))
    else:
        in_school_hours = False

    facts = {
        "red": vehicle["traffic_light"].strip().lower() == "red",
        "speed_above_5": vehicle["speed"] > 5,
        "speed_above_30": vehicle["speed"] > 30,
        "speed_above_40": vehicle["speed"] > 40,
        "speed_above_60": vehicle["speed"] > 60,
        "is_bus": vehicle["is_bus"],
        "illegal_parking": vehicle["parked_duration"] > 5,
        "erp_active": vehicle["erp_active"],
        "erp_charge_violation": vehicle["erp_balance"] < vehicle["charge_amount"],
        "school_zone": vehicle["school_zone"],
        "in_school_hours": in_school_hours, 
        "in_bus_lane_hours": in_bus_lane_hours,
        "expressway": vehicle["zone_type"].strip().lower() == "expressway",
        "speed_within_expressway": vehicle["speed"] <= 90,
        "residential": vehicle["zone_type"].strip().lower() == "residential",
        "speed_within_residential": vehicle["speed"] <= 50,
        "made_uturn": vehicle["made_uturn"] 
    }
    for symbol, value in facts.items():
        if value:
            kb.add(logic.Symbol(symbol))
        else:
            kb.add(logic.Not(logic.Symbol(symbol)))
    if DEBUG:
        print("Knowledge Base for vehicle", vehicle["vehicle_id"])
        for clause in kb:
            print("  ", clause.formula())
        print("-" * 40)
    return kb

def load_vehicle_data(filename):
    vehicles = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                row["speed"] = float(row["speed"])
                row["parked_duration"] = float(row["parked_duration"])
                row["erp_balance"] = float(row["erp_balance"])
                row["charge_amount"] = float(row["charge_amount"])
            except ValueError:
                print(f"Error converting numeric fields in row: {row}")
            row["is_bus"] = row["is_bus"].strip().lower() == "true"
            row["erp_active"] = row["erp_active"].strip().lower() == "true"
            row["school_zone"] = row["school_zone"].strip().lower() == "true"
            row["made_uturn"] = row["made_uturn"].strip().lower() == "true"
            vehicles.append(row)
    return vehicles

def load_traffic_rules(filename):
    return {}

def check_inconsistencies(vehicles):
    inconsistencies = {}
    time_format = "%H:%M"
    for vehicle in vehicles:
        vid = vehicle["vehicle_id"].strip() if vehicle["vehicle_id"].strip() else "Unknown"
        if vehicle["location"].strip() == "" or vehicle["timestamp"].strip() == "":
            inconsistencies.setdefault(vid, set()).add("MissingData: location or timestamp missing")
    vehicle_groups = {}
    for vehicle in vehicles:
        vid = vehicle["vehicle_id"].strip() if vehicle["vehicle_id"].strip() else "Unknown"
        vehicle_groups.setdefault(vid, []).append(vehicle)
    for vid, records in vehicle_groups.items():
        try:
            records_sorted = sorted(records, key=lambda v: datetime.strptime(v["timestamp"], time_format))
        except Exception:
            continue
        for i in range(len(records_sorted)):
            for j in range(i+1, len(records_sorted)):
                rec1 = records_sorted[i]
                rec2 = records_sorted[j]
                t1 = datetime.strptime(rec1["timestamp"], time_format)
                t2 = datetime.strptime(rec2["timestamp"], time_format)
                diff = abs((t2 - t1).total_seconds()) / 60.0
                if diff < 10 and rec1["location"].strip() != rec2["location"].strip():
                    inconsistencies.setdefault(vid, set()).add("ConflictingTimeLocation: records within 10 minutes at different locations")
                if rec1["traffic_light"].strip().lower() != rec2["traffic_light"].strip().lower():
                    inconsistencies.setdefault(vid, set()).add("ConflictingRecords: inconsistent traffic light status")
    for vehicle in vehicles:
        vid = vehicle["vehicle_id"].strip() if vehicle["vehicle_id"].strip() else "Unknown"
        if vehicle.get("zone_type", "").strip().lower() != "expressway" and vehicle["speed"] > 150:
            inconsistencies.setdefault(vid, set()).add("UnrealisticSpeed: speed > 150 km/h in non-expressway setting")
    for vehicle in vehicles:
        vid = vehicle["vehicle_id"].strip() if vehicle["vehicle_id"].strip() else "Unknown"
        if vehicle["speed"] < 0:
            inconsistencies.setdefault(vid, set()).add("NegativeSpeed: speed is negative")
    return inconsistencies

def evaluate_vehicle(vehicle, rules_by_location, debug=False):
    violations = []
    kb = build_kb(vehicle)
    vehicle_loc = vehicle["location"].strip()
    applicable_rules = rules_by_location.get(vehicle_loc, []) + rules_by_location.get("ALL", [])
    
    for violation, description, cnf_clause in applicable_rules:
        clause_str = cnf_clause.strip("()")
        literals = []
        for lit in clause_str.split("OR"):
            lit = lit.strip()
            if lit.startswith("~"):
                literals.append(logic.Not(logic.Symbol(lit[1:])))
            else:
                literals.append(logic.Symbol(lit))
        if len(literals) == 1:
            rule_clause = literals[0]
        else:
            rule_clause = logic.Or(*literals)
        # add rule clause to copy of knowledge base
        kb_with_rule = kb.copy()
        kb_with_rule.add(rule_clause)
        if debug:
            print("Evaluating rule:", violation)
            print("Rule clause:", rule_clause.formula())
            print("KB with rule added:")
            for s in kb_with_rule:
                print("  ", s.formula())
        # check if knowledge base entails violation
        query = logic.Symbol(violation)
        result = resolution_inference(kb_with_rule, query)
        if debug:
            print("Result of resolution-based inference for", violation, ":", result)
            print("-" * 40)
        if result:
            violations.append(violation)
    return list(set(violations))


def main():
    vehicles = load_vehicle_data("vehicle_data.csv")
    rules_by_location = get_traffic_rules() 
    
    violations_report = {}
    for vehicle in vehicles:
        vid = vehicle["vehicle_id"].strip() if vehicle["vehicle_id"].strip() else "Unknown"
        vehicle_violations = evaluate_vehicle(vehicle, rules_by_location, debug=DEBUG)
        violations_report.setdefault(vid, []).extend(vehicle_violations)
    for vid in violations_report:
        violations_report[vid] = list(set(violations_report[vid]))
        
    inconsistencies_report = check_inconsistencies(vehicles)
    
    print("Traffic Violation Report:")
    for vid, vio in violations_report.items():
        if vio:
            print(f"  Vehicle {vid} has violations: {', '.join(vio)}")
        else:
            print(f"  Vehicle {vid} has no violations.")
    
    print("\nInconsistencies Found:")
    if inconsistencies_report:
        for vid, issues in inconsistencies_report.items():
            print(f"  Vehicle {vid}: {', '.join(issues)}")
    else:
        print("  No inconsistencies found.")

if __name__ == "__main__":
    main()
