# Deep Reinforcement Learning Approach for Urban Traffic Signal Control

## Overview
This repository presents a **centralized deep reinforcement learning (DRL) framework** for adaptive traffic signal control in urban road networks.  
The system learns optimal traffic light phase decisions by interacting with a traffic simulation environment, with the objective of minimizing congestion, vehicle waiting time, and queue lengths.

The project is designed as a **research-oriented prototype**, suitable for academic evaluation, experimentation, and further extension.

---

## Key Objectives
- Replace static traffic light timing with **adaptive, learning-based control**
- Optimize traffic flow using **Deep Reinforcement Learning**
- Evaluate system performance using a simulated urban traffic environment
- Provide a modular and extensible codebase for experimentation

---

## System Architecture
The system follows a standard reinforcement learning pipeline:

1. **Traffic Simulator**
   - Simulates urban traffic dynamics
   - Provides environment state (vehicle density, queue lengths, etc.)

2. **RL Environment Wrapper**
   - Converts simulation state into an RL-compatible format
   - Computes reward signals based on traffic efficiency metrics

3. **Deep RL Agent**
   - Learns optimal traffic signal actions
   - Uses neural networks to approximate decision policies

4. **Training & Evaluation Pipeline**
   - Training loop for policy optimization
   - Evaluation module for performance assessment

---

## Reinforcement Learning Formulation
- **State Space**
  - Encodes traffic conditions such as vehicle counts, waiting times, or lane occupancy

- **Action Space**
  - Traffic signal phase selection or phase switching decisions

- **Reward Function**
  - Designed to penalize congestion and delay
  - Encourages smoother traffic flow and reduced waiting time

---

## Installation
### Prerequisites
- Python 3.7+
- Traffic simulator (e.g., SUMO)
- Required Python packages

### Setup
```bash
git clone https://github.com/cosmicbit/Deep-Reinforcement-Approach-on-Traffic-System.git
cd Deep-Reinforcement-Approach-on-Traffic-System
pip install -r requirements.txt
