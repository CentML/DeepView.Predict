import sqlite3
import pandas as pd
import glob
import functools
from tqdm import tqdm

from habitat.analysis.mlp.devices import get_device_features, get_all_devices


def get_dataset(path, features, device_features=None):
    if device_features is None:
        device_features = ['mem', 'mem_bw', 'num_sm', 'single']

    SELECT_QUERY_RUNTIME = """
      SELECT {features}, SUM(run_time_ms) AS run_time_ms
      FROM recordings
      GROUP BY {features}
    """
    
    SELECT_QUERY_KTIME = """
      SELECT {features}, SUM(ktime_ns) * 1e-6 AS run_time_ms
      FROM recordings
      GROUP BY {features}
    """
    # read datasets
    files = glob.glob(path + "/*.sqlite")

    # read individual sqlite files and categorize by device
    devices = dict()
    for f in files:
        device_name = f.split("/")[-1].split("-")[1]
        if "." in device_name: device_name = device_name[:device_name.index(".")]
        
        conn = sqlite3.connect(f)
        
        time_selected = ""
        # check if ktime_ns exist and take this field instead of run_time_ms:
        if "ktime_ns" in [fields[1] for fields in conn.cursor().execute("PRAGMA table_info(recordings)")]:
            SELECT_QUERY = SELECT_QUERY_KTIME
            time_selected = "ktime" 
        else:
            SELECT_QUERY = SELECT_QUERY_RUNTIME
            time_selected = "run_time_ms"
         
        query = SELECT_QUERY.format(features=",".join(features))

        df = pd.read_sql_query(query, conn)
        df = df.rename(columns={"run_time_ms": device_name})

        print("Loaded file %s (%d entries) using %s" % (f, len(df.index), time_selected))

        if device_name not in devices:
            devices[device_name] = []
        devices[device_name].append(df)

    for device in devices.keys():
        devices[device] = pd.concat(devices[device])
        print("Device %s contains %d entries" % (device, len(devices[device].index)))

    print()

    print("Generating dataset")
    # generate vectorized dataset (one entry for each device with device params)
    device_params = get_all_devices(device_features)

    x, y = [], []
    for device in devices.keys():
        df_device = devices[device]
        for row in tqdm(df_device.iterrows(), leave=False, desc=device, total=len(df_device.index)):
            row = row[1]

            x.append(list(row[:-1]) + device_params[device])
            y.append(row[-1])

    return x, y
