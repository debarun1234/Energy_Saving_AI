# Wireless Network Optimization

This Python project focuses on optimizing energy consumption and channel allocation in a wireless network using Genetic Algorithm (GA) and Particle Swarm Optimization (PSO). It simulates a wireless network with a specified number of users and base stations, aiming to minimize energy consumption while meeting traffic load requirements.

## Table of Contents

- [Introduction](#introduction)
- [Simulation Parameters](#simulation-parameters)
- [Optimization Objectives](#optimization-objectives)
- [Genetic Algorithm](#genetic-algorithm)
- [Particle Swarm Optimization](#particle-swarm-optimization)
- [Channel Optimization](#channel-optimization)
- [Usage](#usage)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Wireless networks are essential for modern communication but consume significant energy. This project addresses the challenge of optimizing energy consumption while ensuring that the network can handle the traffic load from multiple users. It employs two optimization algorithms, Genetic Algorithm and Particle Swarm Optimization, to determine the optimal configuration of base stations.

## Simulation Parameters

The simulation is configured with the following parameters:

- Number of Users: 500
- Number of Base Stations: 100
- Number of Iterations: 400
- Population Size: 600
- Energy Threshold: 200

Synthetic data for energy consumption and traffic load are generated based on these parameters.

## Optimization Objectives

Two primary optimization objectives (fitness functions) are defined:

1. **Energy Objective**: Minimize energy consumption while penalizing solutions that exceed a specified energy constraint. This objective considers both energy and traffic load.

2. **Traffic Objective**: Maximize total traffic load, focusing solely on the traffic load aspect.

## Genetic Algorithm

The Genetic Algorithm (GA) is employed to find an optimal solution. It operates through the evolution of a population of solutions, involving selection, crossover, and mutation.

## Particle Swarm Optimization

Particle Swarm Optimization (PSO) is another optimization technique used to find the optimal solution. It simulates the behavior of particles in a swarm to search for the best solution.

## Channel Optimization

After running the optimization algorithms, a channel optimization step determines whether each base station should be active or shut down based on a specified energy threshold.

## Usage

To use this code, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/wireless-network-optimization.git
