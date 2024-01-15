#!/bin/bash
SEED_START=${SEED_START:-1}
SEED_END=${SEED_END:-10}
# Use as BATCH_SIZES="100_1 200_2"
BATCH_SIZES=${BATCH_SIZES:-"10_100 20_50 50_20 100_10"} 
ROOT_DIR=${ROOT_DIR:-./}
CONFIG_DIR=${CONFIG_DIR:-cartpole_easy}
MODELS=${MODELS:-"linattn ffm s5 lru"}
TYPES=${TYPES:-"tape segment"}
WANDB_GROUP=${WANDB_GROUP:-"rdqn_paper"}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.2}
CHUNK_SIZE=${CHUNK_SIZE:-4}
CMD_DELIM=${SERIAL:-";"} # Use ; or & to do serial or parallel
# Set to echo for debug
EXECUTE_CMD=${EXECUTE:-echo}
# Or for the real thing
#EXECUTE_CMD=sbatch /home/sm2558/ser_paper/experiment.wilkes
DEBUG=${DEBUG:-}


# String to list
IFS=' ' read -r -a BATCH_SIZES <<< "${BATCH_SIZES}"
IFS=' ' read -r -a MODELS <<< "${MODELS}"
IFS=' ' read -r -a TYPES <<< "${TYPES}"

CMDS=()
for SEED in $(seq $SEED_START $SEED_END); do
    for MODEL in ${MODELS[@]}; do
        for TYPE in ${TYPES[@]}; do
            if [ "$TYPE" == "tape" ]; then
                    CMD="$DEBUG python $ROOT_DIR/tape_dqn.py ${ROOT_DIR}/experiments/${CONFIG_DIR}/${TYPE}_${MODEL}.yaml -w -p $WANDB_GROUP -n ${CONFIG_DIR}_${MODEL}_${TYPE}_${SEED} --seed ${SEED} ${CMD_DELIM}"
                    CMDS+=("$CMD")
            elif [ "$TYPE" == "segment" ]; then
                for BATCH_SIZE in ${BATCH_SIZES[@]}; do
                    CMD="$DEBUG python $ROOT_DIR/segment_dqn.py ${ROOT_DIR}/experiments/${CONFIG_DIR}/${TYPE}_${MODEL}_${BATCH_SIZE}.yaml -w -p $WANDB_GROUP -n ${CONFIG_DIR}_${MODEL}_${TYPE}_${BATCH_SIZE}_${SEED} --seed ${SEED} ${CMD_DELIM}"
                    CMDS+=("$CMD")
                done
            fi
        done
    done
done
for ((i=0; i<${#CMDS[@]}; i+=CHUNK_SIZE)); do
    # slice of array
    #export CURRENT_CMD=${CMDS[@]:i:CHUNK_SIZE}
    ${EXECUTE_CMD} "${CMDS[@]:i:CHUNK_SIZE}"
done