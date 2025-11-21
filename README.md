# CARLA Garage + Simplex

This repository documents the custom modifications made to adapt **CARLA Garage** for our **Dependable Systems and Networks (DSN)** research experiments, focusing on safety, fault tolerance, and runtime assurance in autonomous driving.

---

## Installation

This repository includes multiple **submodules**, such as the [`perception_simplex`](team_code/safe_agent/perception_simplex) module.  
To ensure all dependencies are correctly initialized, clone the repository **recursively**:

```bash
# Clone with submodules
git clone --recurse-submodules git@github.com:MikaYeghi/carla_garage.git
cd carla_garage
````

If you’ve already cloned the repository without `--recurse-submodules`, initialize the submodules manually:

```bash
git submodule update --init --recursive
```

To update submodules later to their latest committed versions:

```bash
git submodule update --remote --merge
```

---

## Running

To evaluate the DSN agents locally (without using the CARLA leaderboard submission interface), run:

```bash
PYTHONPATH=$PYTHONPATH:/home/acrl-uiuc/Mikael/SS/carla_garage/team_code \
SAVE_PATH="/home/acrl-uiuc/Mikael/SS/carla_garage/results/OvertakeSafeAgent" \
python leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --agent-config model_ckpt/pretrained_models/all_towns/ \
  --agent team_code/safe_agent/safe_agent.py \
  --routes leaderboard/data/DSN/Overtake.xml
```

Logs will be stored in the directory specified by `SAVE_PATH`.
Visualization will be stored there too if an environmental variable `VISUALIZE=1` is supplied.

`SAFETY=1` enables teh safety layer. Otherwise only the mission layer is running.

---

## Modifications

Modifications compared to the original version:

1. **Disabled background traffic** — to ensure controlled and deterministic evaluation
   ([see line 405](leaderboard/leaderboard/scenarios/route_scenario.py#L405))
2. **Added a Faulty Sensor Agent** ([`team_code/faulty_sensor_agent.py`](team_code/faulty_sensor_agent.py))
   Simulates sensor degradation to study perception faults and robustness under degraded sensing conditions.
3. **Added a Safe Agent** ([`team_code/safe_agent/safe_agent.py`](team_code/safe_agent/safe_agent.py))
   Implements a runtime assurance mechanism inspired by [Perception Simplex](https://arxiv.org/abs/2209.01710) for fault-tolerant decision-making.
4. **Integrated Perception Simplex Submodule** ([`team_code/safe_agent/perception_simplex`](team_code/safe_agent/perception_simplex))
   Provides reusable components for modeling and testing perception-level safety architectures.
5. **Updated `.gitignore`** ([`.gitignore`](.gitignore))
   Excludes DSN experiment logs and generated result files for cleaner version control.

---

> **Note:** The original CARLA Garage documentation has been moved to [README_OLD.md](README_OLD.md).
