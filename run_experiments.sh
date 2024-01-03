#!/bin/bash
NUM_SEEDS=10
# CMD=echo # Set to echo for debug
CMD=

# Tape
#for i in {1..10}; do $CMD python tape_dqn.py experiments/batch_sizes/cartpole_easy/tape_linattn.yaml -w -p 'rdqn_paper' -n "tape_linattn_$i" --seed $i; done

for i in {1..10}; do $CMD python tape_dqn.py experiments/batch_sizes/cartpole_easy/tape_ffm.yaml -w -p 'rdqn_paper' -n "tape_ffm_$i" --seed $i; done

# Segment
#for i in {1..10}; do $CMD python segment_dqn.py experiments/batch_sizes/cartpole_easy/segment_linattn_10_400.yaml -w -p 'rdqn_paper' -n "segment_linattn_$i" --seed $i; done
