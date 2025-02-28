# 🚀 AI for Intelligent Transportation System 🚗📍

## 📌 Project Overview
This project is part of the **Artificial Intelligence Concepts & Techniques (AICT) Assignment**. Our goal is to develop an **Intelligent Transportation System** that optimizes vehicle routing, ensures compliance with traffic rules, and predicts congestion patterns.


**Key AI Techniques Used:**  
✅ **Route Planning with Search Algorithms** (BFS, DFS, GBFS, A*)  
✅ **Logical Inference for Traffic Rules** (Propositional Logic & Resolution)  
✅ **Traffic Prediction with Bayesian Networks**  


## 🔹 Project Details  
**📌 Assignment Name:** AICT Assignment 2024/25  
**📌 Real-World Problem:** Intelligent Transportation System  
**📌 Frameworks & Libraries:** Python `networkx`, `pgmpy`, `matplotlib`, `pandas`, `itertools`, `csv`, `datetime`  
**📌 Team Size:** 3 members  


Our system models a **city map as a graph** where:  
✅ **Nodes** = Intersections (locations)  
✅ **Edges** = Roads (weighted by time, congestion, or distance)  


## 👥 Team Members & Roles

| Name         | Task & Contribution                         | Student ID  |
|-------------|--------------------------------------------|-------------|
| 🧑‍💻 ZhiHeng | Route Planning With Search Algorithms (Requirement 2.1) | S10241579H |
| 🧑‍💻  Cheryl | Logical Inference for Traffic Rules (Requirement 2.2) | S10258146H |
| 🧑‍💻  Hendrik | Traffic Prediction With Bayesian Network (Requirement 2.3)| S10241624J |


## 📌 **1️⃣ Route Planning with Search Algorithms**
This module determines the **shortest path** between two locations using four different search algorithms.

### **Implemented Search Algorithms**
1️⃣ **Breadth-First Search (BFS)** - Explores level-by-level, guarantees shortest path in an unweighted graph.  
2️⃣ **Depth-First Search (DFS)** - Explores deeply first, but does not guarantee the shortest path.  
3️⃣ **Greedy Best-First Search (GBFS)** - Uses heuristics to guide the search but may not always find the shortest path.  
4️⃣ **A* Search (A\*)** - Combines **path cost + heuristic**, ensuring an optimal shortest path.  

### **Graph Representation**
```python
graph = {
        "WL": [("Y", 1), ("PG", 6), ("YCK", 4)],
        "Y": [("WL", 1), ("YCK", 2)],
        "PG": [("WL", 6), ("PS", 2)],
        "YCK": [("WL", 4), ("Y", 2), ("PG", 5), ("TP", 3)],
        "TP": [("YCK", 3), ("CA", 1)],
        "PS": [("PG", 2), ("YCK", 5), ("CA", 2)],
        "CA": [("PS", 2), ("TP", 1)]
}
```


## 📌 **2️⃣ Logical Inference for Traffic Rules**
This program takes has traffic rules built into it, taking in a file "vehicle_data.csv". From there it uses propositional logic and resolution-based inference to deduce violations and inconsistencies in the data. using logic.py in the process.

## 📌 **3️⃣ Traffic Prediction with Bayesian Networks**
This program models a city's traffic network using a Bayesian Network to predict congestion levels based on factors like weather, road conditions, accidents, time of day, and day of the week.
Additionally, it implements Simulated Annealing to optimize vehicle routes, minimizing total travel time while considering road congestion.

## ✅ How to Run the Code  
### Requirement 1 - Route Planning  
```python
python route_planning.py
```  
1️⃣ Enter the starting location (e.g., WL)  
2️⃣ Enter the destination location (e.g., CA)  
3️⃣ View the shortest path, cost, and runtime for each algorithm  

### Requirement 2 - Logical Inference for Traffic Rules  
```python
python requirement2.py
```

### Requirement 3 - Traffic Prediction with Bayesian Networks  
```python
python requirement_3.py
```

### Advanced done by Cheryl
```python
python advanced.py
```

## 🔗 Reference  
📌 AICT Assignment Document  
📌 networkx, pgmpy, matplotlib Documentation  
📌 Python Search & Optimization Algorithms
