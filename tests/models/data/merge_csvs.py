"""Merge any number of CSV files horizontally using Polars and save the result to a new CSV file."""

import argparse

import polars as pl


def main(
    files: list[str],
    output_file: str = "merged_data.csv",
) -> None:
    """Merge any number of CSV files horizontally and save the result to a new CSV file.

    This is used for post-processing exported data from Qucs-S simulations.
    """
    if not files:
        raise ValueError("At least one CSV file must be provided")

    # Read the first CSV file as the base
    merged_df = pl.read_csv(files[0], separator=";")

    # Process and merge each subsequent CSV file
    for file in files[1:]:
        df = pl.read_csv(file, separator=";")

        # First ensure the values are the same for duplicate columns, then remove them from df
        common_columns = set(merged_df.columns).intersection(df.columns)
        for col in common_columns:
            if not merged_df[col].equals(df[col]):
                raise ValueError(f"Mismatch found in column: {col} in file: {file}")

        df = df.drop(common_columns)

        merged_df = pl.concat([merged_df, df], how="horizontal")

    merged_df.write_csv(output_file)
    print(f"Merged DataFrame saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge any number of CSV files horizontally."
    )
    parser.add_argument(
        "files", type=str, nargs="+", help="Paths to CSV files to merge"
    )
    parser.add_argument(
        "output_file", type=str, help="Path to save the merged CSV file"
    )
    args = parser.parse_args()
    main(args.files, args.output_file)
