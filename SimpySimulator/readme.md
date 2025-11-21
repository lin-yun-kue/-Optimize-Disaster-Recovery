Some run stats:

```bash
python main.py --seeds=1
```

```python
class SimulationConfig:
    DROP_OFF_LOCATION = (75, 75)
    DEPOT_LOCATION = (0, 0)
    HOSPITAL_LOCATION = (-75, -75)

    # Ticks should be in minutes

    # Disaster Specifics
    LANDSLIDE_MIN_SIZE = 150
    LANDSLIDE_MAX_SIZE = 250
    NUM_STARTING_LANDSLIDES = 10
    NUM_LANDSLIDES = 15

    NUM_TRUCKS = 100
    NUM_EXCAVATORS = 15
    NUM_FIRE_TRUCKS = 4
    NUM_AMBULANCES = 5
```

> Tournament is the policy switching that runs all policies for every decision and picks the best one.

POLICY               | SUCCESS %  | AVG TIME   | STDEV      | MIN      | MAX
---------------------|------------|------------|------------|----------|-------------
tournament           | 100.0    % | 2324.04    | 0.00       | 2324     | 2324
balanced_ratio       | 100.0    % | 3022.78    | 0.00       | 3023     | 3023
random               | 100.0    % | 3134.17    | 0.00       | 3134     | 3134
first_priority       | 100.0    % | 3478.69    | 0.00       | 3479     | 3479
split_excavator      | 100.0    % | 3482.48    | 0.00       | 3482     | 3482
smart_split          | 100.0    % | 3489.35    | 0.00       | 3489     | 3489
cost_function        | 100.0    % | 3611.96    | 0.00       | 3612     | 3612
chain_gang           | 100.0    % | 4154.77    | 0.00       | 4155     | 4155
smallest_job_first   | 100.0    % | 4535.75    | 0.00       | 4536     | 4536
closest_neighbor     | 0.0      % | N/A        | N/A        | N/A      | N/A
gravity              | 0.0      % | N/A        | N/A        | N/A      | N/A


