# Reinforcement Learning to Optimize Disaster Recovery Efficiency with DES and GIS

**Aidan Schmitigal & Yun-Kuei Lin** · Oregon State University, College of Engineering  
Advisor: Alan Fern · Project Partner: Dr. Joseph Louis (Civil & Construction Engineering)

## Overview

This project is about whether a learned policy can optimize responses in a simulated disaster response scenario. We built a Discrete Event Simulation environment grounded in real GIS road network data from Clatsop County, Oregon, and used it to train, evaluate, and compare a range of dispatch strategies including hand-crafted heuristics, Policy Switcher, Proximal Policy Optimization models, Imitation Learning, and PPO initialized from an imitation model. Our main finding is that an imitation-initialized PPO agent outperforms both the from-scratch PPO baseline and the Policy Switcher it was trained to imitate, suggesting that a strong behavioral prior removes the exploration bottleneck that prevents RL from learning in this domain.

## Approach

**[Simulation environment](SimPyTest/simulation.py).** The environment is built with SimPy and wraps GIS road network data for Clatsop County, OR. Resources (trucks and excavators), depots, dump sites, and landslide disasters are all positioned on the real road network. Travel times follow actual routing distances. The simulator is wrapped as a Gymnasium environment for use in training and evaluating ML models. Three difficulty levels: easy, medium, and hard, all vary the number and scale of disasters spawned per season, with seasonal variation in both frequency and size.

**[Heuristic baselines](SimPyTest/policies.py).** Hand-crafted dispatch policies serve as baselines.

**[Policy Switcher](SimPyTest/policies_tournament.py).** Since no single heuristic performs well across all decision points, the Policy Switcher evaluates all candidate policies at every step via simulation-based lookahead and selects the action taken by the best-performing policy.

**Imitation model.** A supervised classification model is trained on state-action trajectories collected from the Policy Switcher. The architecture is a Deep Sets scoring network: a shared MLP encoder processes each (resource, disaster) pair, a masked mean pooling operation aggregates the embeddings into a global context vector, and a candidate-scoring head produces per-action logits.

**[PPO from scratch](ppo/ppo_dispatch.py).** A MaskablePPO agent using the same Deep Sets policy architecture is trained via a difficulty curriculum (easy → medium → hard), with a dense per-step reward derived from the change in the training objective score plus a small action-quality heuristic bonus.

**[PPO initialized from the imitation model](ppo_init/train_critic.py).** The actor network is initialized from the pretrained imitation model weights. The critic is first trained alone for 400 iterations (with the actor frozen) to produce stable value estimates before any policy updates. PPO then fine-tunes both components.

## Results

Scores are the average training objective across 10 held-out seeds at medium difficulty. Higher (less negative) is better.

| Method              | Score       |
| ------------------- | ----------- |
| **PPO Init** (ours) | **−40,388** |
| Policy Switcher     | −43,062     |
| Balanced Ratio      | −43,474     |
| Imitation Model     | −46,055     |
| PPO from scratch    | −46,100     |

The Policy Switcher requires ~2,293 ms per decision. Both the Imitation Model and PPO Init operate at ~3 ms.

## Objective Function

$$\text{score} = -\left( P_{\text{fail}} + 100{,}000 \cdot N_{\text{unresolved}} + H_{\text{closure}} + \frac{T_{\text{disasters}}}{60} + \frac{C_{\text{total}}}{1000} \right)$$

where $P_{\text{fail}}$ = 1,000,000 for any non-success terminal outcome, $N_{\text{unresolved}}$ is unresolved disasters, $H_{\text{closure}}$ is total weighted closure-hours, $T_{\text{disasters}}$ is total minutes with at least one active disaster, and $C_{\text{total}}$ is total spending in dollars.

## Running the Code

**Evaluate heuristic policies**

```bash
python benchmark.py --suite quick
```

**Train the imitation model**

```bash
python -m scripts.training.mlp.train_dispatch_ml
```

**Train PPO from scratch**

```bash
python -m scripts.training.ppo.train_ppo
```

**Train PPO initialized from imitation model**

```bash
python -m scripts.training.ppo_init.train_critic
```

**GIS setup.** Road network data for Clatsop County is expected at `maps/tl_2024_41007_roads/tl_2024_41007_roads.shp`. Download the TIGER/Line shapefile for Clatsop County, OR from the [US Census Bureau](https://www.census.gov/cgi-bin/geo/shapefiles/index.php) and place it at that path. GIS can be disabled for all runs by omitting the `--gis` flag or passing `gis_config=None` in the scenario config.

## Acknowledgments

This project was completed as a capstone at Oregon State University. We thank Dr. Joseph Louis (School of Civil and Construction Engineering, OSU) for domain expertise on disaster response operations and landslide data, and Alan Fern for advising the reinforcement learning methodology.

Landslide location data is derived from the [Oregon Department of Geology and Mineral Industries Statewide Landslide Information Database (SLIDO)](https://www.oregongeology.org/slido/), April 2024.
