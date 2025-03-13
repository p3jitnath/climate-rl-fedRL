import csv
import random

import cm
import numpy as np
import redis


def update_params():
    """
    Updates the forcing, feedback factor, and heat capacity within a certain range.

    Returns
    -------
    dict
        A dictionary containing the updated 'forcing', 'feedback_factor', and 'heat_capacity'.
    """
    updated_forcing = random.uniform(3.5, 4.0)  # Random forcing within a range
    updated_feedback_factor = random.uniform(
        0.4, 0.6
    )  # Random feedback factor within a range
    updated_heat_capacity = random.uniform(
        950.0, 1050.0
    )  # Random heat capacity within a range

    return {
        "forcing": updated_forcing,
        "feedback_factor": updated_feedback_factor,
        "heat_capacity": updated_heat_capacity,
    }


# Connect to Redis
r = redis.Redis(host="localhost", port=6379, db=0)

# Store initial parameters in Redis (only done once initially)
r.set("forcing", 3.7)  # Example forcing value
r.set("feedback_factor", 0.5)  # Initial feedback factor
r.set("heat_capacity", 1000.0)  # Heat capacity of the system
r.set("temperature_t", 15.0)  # Initial temperature in degrees Celsius

# Prepare the CSV file to store results
csv_file = "cm_results.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.writer(file)
    # Write the header row
    writer.writerow(
        [
            "Step",
            "Forcing",
            "Feedback Factor",
            "Heat Capacity",
            "Temperature (°C)",
        ]
    )

    # Run the model for several steps (e.g., 5 steps)
    for step in range(5):

        # Read parameters from Redis
        forcing = float(r.get("forcing"))
        feedback_factor = float(r.get("feedback_factor"))
        heat_capacity = float(r.get("heat_capacity"))
        temperature_t = float(r.get("temperature_t"))

        # Run one step of the climate model
        temperature_t1 = cm.forward(
            forcing, feedback_factor, heat_capacity, temperature_t
        )

        # Print result of the step
        print(f"[Step {step:03}] New temperature: {temperature_t1:.4f} °C")

        # Store the updated temperature back in Redis
        r.set("temperature_t", temperature_t1)

        # Update the parameters randomly and store them back in Redis
        new_params = update_params()
        r.set("forcing", new_params["forcing"])
        r.set("feedback_factor", new_params["feedback_factor"])
        r.set("heat_capacity", new_params["heat_capacity"])

        # Save the results to the CSV file
        writer.writerow(
            [
                step,
                new_params["forcing"],
                new_params["feedback_factor"],
                new_params["heat_capacity"],
                temperature_t1,
            ]
        )
