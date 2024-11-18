import numpy as np
import pandas as pd
import joblib

read_params = [
    {
        "filepath_or_buffer": "dataset/yoochoose-clicks-200k.dat",
        "usecols": ["SessionId", "Time", "ItemId"],
        "dtype": {
            "SessionId": np.int32,
            "ItemId": np.str_,
        },
        "parse_dates": ["Time"],
    },
    {
        "filepath_or_buffer": "dataset/yoochoose-test-200k.dat",
        "usecols": ["SessionId", "Time", "ItemId"],
        "dtype": {
            "SessionId": np.int32,
            "ItemId": np.str_,
        },
        "parse_dates": ["Time"],
    },
]

write_params = [
    {"filename": "dataset/yoochoose-clicks-200k.pickle"},
    {"filename": "dataset/yoochoose-test-200k.pickle"},
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Sessions of length 1 are filtered out (both in training and test sets)
    print("Preprocessing...")

    df = df.groupby("SessionId").filter(lambda x: len(x) > 1)
    df.info()

    print("Done Preprocessing")


def main() -> None:
    for i in range(len(read_params)):
        read_param, write_param = read_params[i], write_params[i]

        print(f"read {read_param}...\n")
        df = pd.read_csv(**read_param)

        preprocess(df)

        print(f"write {write_param}...\n")
        joblib.dump(df, **write_param)

        print(f"done\n\n")


if __name__ == "__main__":
    main()
