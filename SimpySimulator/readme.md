# Some run stats:

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

| POLICY               | SUCCESS %  | TIME       |
|----------------------|------------|------------|
| tournament           | 100.0    % | 2324.04    |
| balanced_ratio       | 100.0    % | 3022.78    |
| random               | 100.0    % | 3134.17    |
| first_priority       | 100.0    % | 3478.69    |
| split_excavator      | 100.0    % | 3482.48    |
| smart_split          | 100.0    % | 3489.35    |
| cost_function        | 100.0    % | 3611.96    |
| chain_gang           | 100.0    % | 4154.77    |
| smallest_job_first   | 100.0    % | 4535.75    |
| closest_neighbor     | 0.0      % | DNF        |
| gravity              | 0.0      % | DNF        |


# More

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

    NUM_TRUCKS = 50
    NUM_EXCAVATORS = 10
    NUM_FIRE_TRUCKS = 4
    NUM_AMBULANCES = 5
```


| POLICY               | SUCCESS %  | TIME       |
|----------------------|------------|------------|
| tournament           | 100.0    % | 2914.65    |
| cost_function        | 100.0    % | 3345.83    |
| smart_split          | 100.0    % | 3658.72    |
| balanced_ratio       | 100.0    % | 3701.50    |
| chain_gang           | 100.0    % | 3915.30    |
| split_excavator      | 100.0    % | 4187.39    |
| first_priority       | 100.0    % | 4489.75    |
| smallest_job_first   | 100.0    % | 4548.02    |
| random               | 0.0      % | DNF        |
| closest_neighbor     | 0.0      % | DNF        |
| gravity              | 0.0      % | DNF        |




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

    NUM_TRUCKS = 50
    NUM_EXCAVATORS = 10
    NUM_FIRE_TRUCKS = 4
    NUM_AMBULANCES = 5

ompleted tournament in 1751.7917369544305 seconds.

--- END OF SEED 9  ---


=====================================================================================
POLICY               | SUCCESS %  | AVG TIME   | STDEV      | MIN      | MAX
-------------------------------------------------------------------------------------
tournament           | 100.0    % | 1512.15    | 146.64     | 1289     | 1752
=====================================================================================

=====================================================================================
POLICY               | SUCCESS %  | AVG TIME   | STDEV      | MIN      | MAX
-------------------------------------------------------------------------------------
balanced_ratio       | 100.0    % | 1599.15    | 156.64     | 1389     | 1865
split_excavator      | 100.0    % | 1658.87    | 148.96     | 1470     | 1913
cost_function        | 100.0    % | 1660.19    | 156.79     | 1407     | 1867
smart_split          | 100.0    % | 2003.98    | 206.89     | 1638     | 2285
first_priority       | 100.0    % | 3351.01    | 221.31     | 3017     | 3778
smallest_job_first   | 100.0    % | 3451.77    | 257.66     | 3039     | 3893
chain_gang           | 100.0    % | 3501.99    | 334.89     | 3037     | 4037
random               | 0.0      % | N/A        | N/A        | N/A      | N/A
closest_neighbor     | 0.0      % | N/A        | N/A        | N/A      | N/A
gravity              | 0.0      % | N/A        | N/A        | N/A      | N/A
=====================================================================================