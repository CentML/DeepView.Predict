import sqlite3
import pandas as pd
import glob
import functools
from tqdm import tqdm

from habitat.analysis.mlp.devices import get_device_features, get_all_devices

def onehot(idx, count):
    enc = [0] * count
    enc[idx] = 1
    return enc

def get_devices(path):
    files = glob.glob(path + "/*.sqlite")
    devices = list()
    for f in files:
        device_name = f.split("/")[-1].split("-")[1]
        if "." in device_name: device_name = device_name[:device_name.index(".")]

        devices.append(device_name)

    return list(set(devices))

def get_dataset(path, features, device_features=None):
    print("get_dataset", path, features)
    if device_features is None:
        device_features = ['mem', 'mem_bw', 'num_sm', 'single']

    SELECT_QUERY = """
      SELECT {features}, SUM(run_time_ms) AS run_time_ms
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
        query = SELECT_QUERY.format(features=",".join(features))

        df = pd.read_sql_query(query, conn)
        df = df.rename(columns={"run_time_ms": device_name})

        print("Loaded file %s (%d entries)" % (f, len(df.index)))

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

    device_list = list(devices.keys())
    print("device_list", device_list)

    x, y = [], []
    for idx, device in enumerate(device_list):
        df_device = devices[device]
        device_encoding = onehot(idx, len(device_list))
        print("device_encoding", device_encoding)
        for row in tqdm(df_device.iterrows(), leave=False, desc=device, total=len(df_device.index)):
            row = row[1]

            x.append(device_encoding + list(row[:-1]) + device_params[device])
            y.append(row.iloc[-1])

    return device_list, x, y
