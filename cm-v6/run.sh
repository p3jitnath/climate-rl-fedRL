#!/bin/bash

# 0. Perform cleanup
# Stop all pending ray processes
ray stop --force

# Check for keys matching "SIG*" pattern in Redis on port 6380
sig_keys=$(redis-cli -p 6380 KEYS "SIG*")
# Check for keys matching "actor*" pattern in Redis on port 6380
actor_keys=$(redis-cli -p 6380 KEYS "actor*")

# If there are SIG* keys, delete them
if [[ -n "$sig_keys" ]]; then
  echo "Found SIG* keys: $sig_keys"
  echo "$sig_keys" | xargs redis-cli -p 6380 DEL
  echo "SIG* keys deleted."
else
  echo "No keys found matching pattern 'SIG*'."
fi

# If there are actor* keys, delete them
if [[ -n "$actor_keys" ]]; then
  echo "Found actor* keys: $actor_keys"
  echo "$actor_keys" | xargs redis-cli -p 6380 DEL
  echo "actor* keys deleted."
else
  echo "No keys found matching pattern 'actor*'."
fi

# 1. Run smartsim
python run_smartsim.py
