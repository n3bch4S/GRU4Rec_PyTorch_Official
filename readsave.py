import numpy as np
import pandas as pd
import joblib

read_params = [
    {
        "filepath_or_buffer": "dataset/yoochoose-clicks.dat",
        "usecols": ["SessionId", "Time", "ItemId"],
        "dtype": {
            "SessionId": np.int32,
            "ItemId": np.str_,
        },
        "parse_dates": ["Time"],
    },
    {
        "filepath_or_buffer": "dataset/yoochoose-test.dat",
        "usecols": ["SessionId", "Time", "ItemId"],
        "dtype": {
            "SessionId": np.int32,
            "ItemId": np.str_,
        },
        "parse_dates": ["Time"],
    },
]

write_params = [
    {"filename": "dataset/yoochoose-clicks.pickle"},
    {"filename": "dataset/yoochoose-test.pickle"},
]

for i in range(len(read_params)):
    read_param, write_param = read_params[i], write_params[i]

    print(f"read {read_param}...\n")
    df = pd.read_csv(**read_param)

    print(f"write {write_param}...\n")
    joblib.dump(df, **write_param)

    print(f"done\n\n\n")
