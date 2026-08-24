import random
import simpy
import pandas as pd

DATA_FILE = "test_1.xlsx"
SIM_TIME = 288000
random.seed(42)

line = pd.read_excel(DATA_FILE, sheet_name="Production_Line")
arrival = pd.read_excel(DATA_FILE, sheet_name="Arrival")
arrival_mean = float(arrival.loc[0, "Mean (seconds)"])
arrival_half = float(arrival.loc[0, "Half-width (seconds)"])

env = simpy.Environment()
machines = {r["Process"]: simpy.Resource(env, capacity=int(r["Capacity"]))
            for _, r in line.iterrows()}
busy = {p: 0.0 for p in line["Process"]}
log = []
completed = 0

def sample_time(row):
    mean = float(row["Mean (seconds)"])
    half = float(row["Half-width (seconds)"])
    return mean if half == 0 else random.uniform(mean-half, mean+half)

def part(env, part_id):
    global completed
    for _, row in line.iterrows():
        process = row["Process"]
        queue_enter = env.now
        with machines[process].request() as req:
            yield req
            start = env.now
            wait = start - queue_enter
            ptime = sample_time(row)
            busy[process] += ptime
            yield env.timeout(ptime)
            end = env.now
            log.append({
                "Part ID": part_id, "Process": process,
                "Queue Entry Time (sec)": queue_enter,
                "Processing Start Time (sec)": start,
                "Processing End Time (sec)": end,
                "Waiting Time (sec)": wait,
                "Processing Time (sec)": ptime})
    completed += 1

def source(env):
    n = 1
    while True:
        env.process(part(env, f"P{n:05d}"))
        ia = random.uniform(arrival_mean-arrival_half, arrival_mean+arrival_half)
        yield env.timeout(ia)
        n += 1

env.process(source(env))
env.run(until=SIM_TIME)

log_df = pd.DataFrame(log)
log_df.to_excel("simulation_log.xlsx", index=False)

results = []
for _, row in line.iterrows():
    p = row["Process"]
    cap = int(row["Capacity"])
    d = log_df[log_df["Process"] == p]
    util = 100 * busy[p] / (SIM_TIME * cap)
    avg_wait = d["Waiting Time (sec)"].mean() if len(d) else 0
    max_wait = d["Waiting Time (sec)"].max() if len(d) else 0
    results.append({"Process":p, "Capacity":cap, "Utilization (%)":util,
                    "Average Waiting Time (sec)":avg_wait,
                    "Maximum Waiting Time (sec)":max_wait,
                    "Parts Processed":len(d)})

res = pd.DataFrame(results)
u = res["Utilization (%)"].max()
w = res["Average Waiting Time (sec)"].max()
res["Bottleneck Score"] = 0.5*res["Utilization (%)"]/u + 0.5*res["Average Waiting Time (sec)"]/w
res = res.sort_values("Bottleneck Score", ascending=False)
res.to_excel("bottleneck_results.xlsx", index=False)

print("\nSimulation finished")
print("Completed parts:", completed)
print(res[["Process","Utilization (%)","Average Waiting Time (sec)","Parts Processed","Bottleneck Score"]].to_string(index=False))
print("\nMain bottleneck candidate:", res.iloc[0]["Process"])
print("\nCreated: simulation_log.xlsx and bottleneck_results.xlsx")
