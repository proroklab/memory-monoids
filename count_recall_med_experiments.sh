#!/bin/bash
NUM_SEEDS=10
BATCH_SIZES=("10_100" "20_50" "50_20" "100_10") 
# CMD=echo # Set to echo for debug
CMD=

# Tape
for i in $(seq 1 $NUM_SEEDS); do $CMD python tape_dqn.py experiments/batch_sizes/count_recall_medium/tape_linattn.yaml -w -p 'rdqn_paper' -n "tape_linattn_$i" --seed $i; done

#for i in $(seq 1 $NUM_SEEDS); do $CMD python tape_dqn.py experiments/batch_sizes/count_recall_medium/tape_ffm.yaml -w -p 'rdqn_paper' -n "tape_ffm_$i" --seed $i; done

# Segment
for bs in ${BATCH_SIZES[@]}; do
    #for i in $(seq 1 $NUM_SEEDS); do $CMD python segment_dqn.py experiments/batch_sizes/count_recall_medium/segment_linattn_${bs}.yaml -w -p 'rdqn_paper' -n "segment_linattn_${bs}_${i}" --seed $i; done

    # Segment FFM
    #for i in $(seq 1 $NUM_SEEDS); do $CMD python segment_dqn.py experiments/batch_sizes/count_recall_medium/segment_ffm_${bs}.yaml -w -p 'rdqn_paper' -n "segment_ffm_${bs}_${i}" --seed $i; done

done
