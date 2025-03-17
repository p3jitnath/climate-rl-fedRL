# Define the SmartSim experiment artifact to clean
ARTIFACT="SCM_FLWR_Orchestrator"
ARTIFACT_PATH="/gws/nopw/j04/ai4er/users/pn341/climate-rl-fedrl/cm-v5/$ARTIFACT"

# Remove the specified artifact directory
echo "Cleaning SmartSim artifacts at: $ARTIFACT_PATH"
rm -rf "$ARTIFACT_PATH"
echo "Cleanup complete"
