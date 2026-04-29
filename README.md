# Hydraulic-safety-system-simulation
This project is a simulation tool that analyzes the safety and resilience of aircraft hydraulic systems.

I modeled the hydraulic network as a graph using Python, where components like pumps, valves, and actuators are nodes, and hydraulic lines are edges.

The system simulates random failures in components and lines, applies isolation logic to prevent failure propagation, and then evaluates how well pressure is maintained across the system.

![image alt](https://github.com/swarnimakhare/Hydraulic-safety-system-simulation/blob/a3196b8a1fafa3867c3d5e7cd279fbb6a2902b84/Screenshot%202026-04-20%20211727.png)


I use Monte Carlo simulations to run many failure scenarios and compute metrics like actuator functionality, pressure distribution, redundancy, and a composite safety score.

![image alt](https://github.com/swarnimakhare/Hydraulic-safety-system-simulation/blob/acb4c95368dc9bd12b92fc68dbb88e398a30079d/Screenshot%202026-04-20%20211825.png)

Additionally, I implemented a genetic algorithm that improves the network design by adding connections to increase resilience under failure conditions.

Overall, the goal is to compare different hydraulic architectures and identify designs that are more robust and fault-tolerant.

