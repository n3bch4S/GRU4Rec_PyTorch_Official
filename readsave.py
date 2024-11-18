import numpy as np
import pandas as pd
import joblib

read_params: list[dict] = [
    {
        "filepath_or_buffer": "dataset/yoochoose-clicks-100k.dat",
        "usecols": ["SessionId", "Time", "ItemId"],
        "dtype": {
            "SessionId": np.int32,
            "ItemId": np.str_,
        },
        "parse_dates": ["Time"],
    },
    {
        "filepath_or_buffer": "dataset/yoochoose-test-100k.dat",
        "usecols": ["SessionId", "Time", "ItemId"],
        "dtype": {
            "SessionId": np.int32,
            "ItemId": np.str_,
        },
        "parse_dates": ["Time"],
    },
]

write_params: list[dict] = [
    {"filename": "dataset/yoochoose-clicks-100k.pickle"},
    {"filename": "dataset/yoochoose-test-100k.pickle"},
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
    write_train, write_test = write_params[0], write_params[1]

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
    joblib.dump(train_df, **write_train)
    joblib.dump(test_df, **write_test)
    print("Done saving")


if __name__ == "__main__":
    main()
