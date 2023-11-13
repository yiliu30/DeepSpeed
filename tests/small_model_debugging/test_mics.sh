#!/bin/bash

deepspeed test_mics_config.py --mics_shard_size=1

deepspeed test_mics_config.py --mics_shard_size=2

# for debugging the hierarchical params gathering
export NDEV_PER_NODE=2
deepspeed --include="localhost:0,2" test_model.py 
deepspeed --include="localhost:2,4" test_in_pruner.py
#--mics_shard_size=2 --mics_hierarchical_params_gather
