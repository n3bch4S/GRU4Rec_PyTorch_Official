import numpy as np
import pandas as pd
import joblib

read_params: list[dict] = [
    {
        "filepath_or_buffer": "raw_dataset/yoochoose-clicks-100k.dat",
        "usecols": ["SessionId", "Time", "ItemId"],
        "dtype": {
            "SessionId": np.int32,
            "ItemId": np.str_,
        },
        "parse_dates": ["Time"],
    },
    {
        "filepath_or_buffer": "raw_dataset/yoochoose-test-100k.dat",
        "usecols": ["SessionId", "Time", "ItemId"],
        "dtype": {
            "SessionId": np.int32,
            "ItemId": np.str_,
        },
        "parse_dates": ["Time"],
    },
]

pickle_params: list[dict] = [
    {"filename": "dataset/01_yoochoose_clicks_100k.pickle"},
    {"filename": "dataset/02_yoochoose_test_100k.pickle"},
]

csv_params: list[dict] = [
    {
        "path_or_buf": "dataset/03_yoochoose_clicks_100k.csv",
        "index": False,
    },
    {
        "path_or_buf": "dataset/04_yoochoose_test_100k.csv",
        "index": False,
    },
]

tsv_params: list[dict] = [
    {
        "path_or_buf": "dataset/05_yoochoose_clicks_100k.tsv",
        "sep": "\t",
        "index": False,
    },
    {
        "path_or_buf": "dataset/06_yoochoose_test_100k.tsv",
        "sep": "\t",
        "index": False,
    },
]


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Sessions of length 1 are filtered out (both in training and test sets)
    print("Preprocessing...")

    df = df.groupby("SessionId").filter(lambda x: len(x) > 1)
    df.info()

    print("Done preprocessing\n\n")
    return df


def remove_non_exist_item(
    train_df: pd.DataFrame, test_df: pd.DataFrame
) -> pd.DataFrame:
    # Removed clicks from the test set if the clicked item does not exist in the training set.
    print("Remove non exist item...")
    train_df = train_df["ItemId"].unique()
    test_df = test_df[test_df["ItemId"].isin(train_df)]
    test_df.info()

    print("Done removing\n\n")
    return test_df


def main() -> None:
    read_train, read_test = read_params[0], read_params[1]
    pickle_train, pickle_test = pickle_params[0], pickle_params[1]
    csv_train, csv_test = csv_params[0], csv_params[1]
    tsv_train, tsv_test = tsv_params[0], tsv_params[1]

    # Read train test
    print("Read train test...")
    train_df, test_df = pd.read_csv(**read_train), pd.read_csv(**read_test)
    print("Done reading\n\n")

    # Preprocess train test
    train_df, test_df = preprocess(train_df), preprocess(test_df)

    # Preprocess test
    test_df = remove_non_exist_item(train_df, test_df)

    # Save processed data
    print("Saving...")
    joblib.dump(train_df, **pickle_train)
    joblib.dump(test_df, **pickle_test)

    train_df.to_csv(**csv_train)
    test_df.to_csv(**csv_test)

    train_df.to_csv(**tsv_train)
    test_df.to_csv(**tsv_test)
    print("Done saving")


if __name__ == "__main__":
    main()
