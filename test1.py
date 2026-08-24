import random
import simpy
import pandas as pd

DATA_FILE = "bearing_des_dataset_FINAL.xlsx"
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ------------------------------------------------------------
# 1. READ INPUT DATA
# ------------------------------------------------------------
line = pd.read_excel(DATA_FILE, sheet_name="Production_Line")
arrival = pd.read_excel(DATA_FILE, sheet_name="Arrival")
setting = pd.read_excel(DATA_FILE, sheet_name="Simulation_Setting")

SIM_TIME = float(
    setting[
        (setting["Setting"] == "Simulation time") &
        (setting["Unit"] == "seconds")
    ]["Value"].iloc[0]
)

ARRIVAL_MEAN = float(arrival.loc[0, "Mean (seconds)"])
ARRIVAL_HALF = float(arrival.loc[0, "Half-width (seconds)"])

# ------------------------------------------------------------
# 2. CREATE SIMPY ENVIRONMENT AND RESOURCES
# ------------------------------------------------------------
env = simpy.Environment()

machines = {
    row["Process"]: simpy.Resource(
        env,
        capacity=int(row["Capacity"])
    )
    for _, row in line.iterrows()
}

busy_time = {p: 0.0 for p in line["Process"]}
queue_events = {p: [] for p in line["Process"]}
simulation_log = []
completed_parts = 0

# ------------------------------------------------------------
# 3. SAMPLE PROCESSING TIME FROM PAPER DISTRIBUTION
# ------------------------------------------------------------
def get_processing_time(row):
    mean = float(row["Mean (seconds)"])
    half = float(row["Half-width (seconds)"])

    if half == 0:
        return mean

    return random.uniform(mean - half, mean + half)

# ------------------------------------------------------------
# 4. ONE PART MOVES THROUGH ALL 12 PROCESSES
# ------------------------------------------------------------
def part(env, part_id):
    global completed_parts

    for _, row in line.iterrows():
        process = row["Process"]
        machine = machines[process]

        queue_enter = env.now

        # +1 means this part joins the queue.
        queue_events[process].append((env.now, +1))

        with machine.request() as request:
            yield request

            process_start = env.now

            # -1 means this part leaves the queue and starts processing.
            queue_events[process].append((env.now, -1))

            waiting_time = process_start - queue_enter
            ptime = get_processing_time(row)

            # Count only the busy time that actually falls inside the simulation horizon.
            actual_busy = min(ptime, max(0.0, SIM_TIME - env.now))
            busy_time[process] += actual_busy

            yield env.timeout(ptime)

            process_end = env.now

            simulation_log.append({
                "Part ID": part_id,
                "Process": process,
                "Queue Entry Time (sec)": queue_enter,
                "Processing Start Time (sec)": process_start,
                "Processing End Time (sec)": process_end,
                "Waiting Time (sec)": waiting_time,
                "Processing Time (sec)": ptime
            })

    completed_parts += 1

# ------------------------------------------------------------
# 5. GENERATE PARTS USING THE PAPER ARRIVAL DISTRIBUTION
# ------------------------------------------------------------
def source(env):
    n = 1

    while True:
        env.process(part(env, f"P{n:05d}"))

        interarrival = random.uniform(
            ARRIVAL_MEAN - ARRIVAL_HALF,
            ARRIVAL_MEAN + ARRIVAL_HALF
        )

        yield env.timeout(interarrival)
        n += 1

# ------------------------------------------------------------
# 6. RUN THE 80-HOUR SIMULATION
# ------------------------------------------------------------
env.process(source(env))
env.run(until=SIM_TIME)

# ------------------------------------------------------------
# 7. SAVE PART-LEVEL SIMULATION LOG
# ------------------------------------------------------------
log_df = pd.DataFrame(simulation_log)
log_df.to_excel("simulation_log.xlsx", index=False)

# ------------------------------------------------------------
# 8. CALCULATE TIME-WEIGHTED QUEUE LENGTH
# ------------------------------------------------------------
def queue_statistics(events, simulation_time):
    if not events:
        return 0.0, 0

    queue = 0
    last_time = 0.0
    queue_area = 0.0
    max_queue = 0

    # If two events occur at the same instant, queue entry (+1) is applied first.
    for time, change in sorted(events, key=lambda x: (x[0], -x[1])):
        if time > simulation_time:
            break

        queue_area += queue * (time - last_time)
        queue += change
        max_queue = max(max_queue, queue)
        last_time = time

    queue_area += queue * (simulation_time - last_time)

    average_queue = queue_area / simulation_time
    return average_queue, max_queue

# ------------------------------------------------------------
# 9. BOTTLENECK KPIs
# ------------------------------------------------------------
results = []

for _, row in line.iterrows():
    process = row["Process"]
    capacity = int(row["Capacity"])

    process_log = log_df[log_df["Process"] == process]

    utilization = 100 * busy_time[process] / (SIM_TIME * capacity)

    average_wait = (
        process_log["Waiting Time (sec)"].mean()
        if len(process_log) else 0.0
    )

    maximum_wait = (
        process_log["Waiting Time (sec)"].max()
        if len(process_log) else 0.0
    )

    average_queue, maximum_queue = queue_statistics(
        queue_events[process],
        SIM_TIME
    )

    results.append({
        "Process": process,
        "Capacity": capacity,
        "Utilization (%)": utilization,
        "Average Waiting Time (sec)": average_wait,
        "Maximum Waiting Time (sec)": maximum_wait,
        "Average Queue Length": average_queue,
        "Maximum Queue Length": maximum_queue,
        "Parts Processed": len(process_log)
    })

results_df = pd.DataFrame(results)

# ------------------------------------------------------------
# 10. IDENTIFY THE BOTTLENECK
# ------------------------------------------------------------
# The source paper identifies the critical operation using
# the MAXIMUM AVERAGE WAITING TIME.
# We follow the same rule for the first validation.
results_df = results_df.sort_values(
    "Average Waiting Time (sec)",
    ascending=False
).reset_index(drop=True)

results_df["Bottleneck?"] = "NO"
results_df.loc[0, "Bottleneck?"] = "YES"

results_df.to_excel(
    "bottleneck_results.xlsx",
    index=False
)

# ------------------------------------------------------------
# 11. COMPARE WITH THE PAPER REFERENCE
# ------------------------------------------------------------
paper = pd.read_excel(
    DATA_FILE,
    sheet_name="Paper_Reference_Results"
)

comparison = results_df.merge(
    paper,
    on="Process",
    how="left",
    suffixes=("_SimPy", "_Paper")
)

comparison.to_excel(
    "paper_vs_simpy.xlsx",
    index=False
)

# ------------------------------------------------------------
# 12. PRINT SUMMARY
# ------------------------------------------------------------
print("\n========================================")
print("BEARING PRODUCTION LINE DES")
print("========================================")

print(f"Simulation time: {SIM_TIME / 3600:.1f} hours")
print(f"Completed parts: {completed_parts}")

print("\nBottleneck analysis:")
print(
    results_df[
        [
            "Process",
            "Utilization (%)",
            "Average Waiting Time (sec)",
            "Average Queue Length",
            "Maximum Queue Length",
            "Parts Processed",
            "Bottleneck?"
        ]
    ].to_string(index=False)
)

print("\nMain bottleneck candidate:")
print(results_df.iloc[0]["Process"])

print("\nFiles created:")
print("simulation_log.xlsx")
print("bottleneck_results.xlsx")
print("paper_vs_simpy.xlsx")
