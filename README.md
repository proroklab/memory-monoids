# Revisiting Recurrent Reinforcement Learning

This repository contains the code for "Revisiting Recurrent Reinforcement Learning".

- `collector/`` contains the sample collection code
- `experiments/`` contains experiments as yaml files
- `memory/`` contains memory model implementations
- `plotting/`` contains plotting tools and scripts used to generate all plots
- `buffer.py` contains replay buffers for SBB and TBB
- `losses.py` contains loss and update functions
- `segment_dqn.py` runs a SBB double DQN, given a SBB experiment. For example, `python segment_dqn.py experiments/cartpole_easy/segment_s5_10_100.yaml` 
- `tape_dqn.py` runs a TBB double DQN, given a tape experiment. For example, `python tape_dqn.py experiments/cartpole_easy/tape_s5.yaml`
- `utils.py` contains various utilities
- `modules.py` contains definitions of the models as well as various utilities
- `returns.py` implements the discounted return as a memory monoid, records the time taken, and ensures the memory monoid is correct
- `run_experiments.sh` is a way to run many experiments at once
- `requirements.txt` should contain necessary packages to run the scripts (without versions, to avoid dependency hell)
- `requirements_freeze.txt` contains the exact dependency verions for the experiments (if `requirements.txt`` does not work)