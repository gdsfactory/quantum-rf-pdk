"""Merge two CSV files horizontally using Polars and save the result to a new CSV file."""

import argparse

import polars as pl


def main(
    file1: str = "data1.csv",
    file2: str = "data2.csv",
    output_file: str = "merged_data.csv",
) -> None:
    df1 = pl.read_csv(file1, separator=";")
    df2 = pl.read_csv(file2, separator=";")
    # First ensure the values are the same for duplicate columns, then remove them from df2
    common_columns = set(df1.columns).intersection(df2.columns)
    for col in common_columns:
        if not df1[col].equals(df2[col]):
            raise ValueError(f"Mismatch found in column: {col}")

    df2 = df2.drop(common_columns)

    merged_df = pl.concat([df1, df2], how="horizontal")

    merged_df.write_csv(output_file)
    print(f"Merged DataFrame saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two CSV files horizontally.")
    parser.add_argument("file1", type=str, help="Path to the first CSV file")
    parser.add_argument("file2", type=str, help="Path to the second CSV file")
    parser.add_argument(
        "output_file", type=str, help="Path to save the merged CSV file"
    )
    args = parser.parse_args()
    main(args.file1, args.file2, args.output_file)
