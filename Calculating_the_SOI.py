import pandas as pd
import numpy as np
import math
#The A~C indices can be customized and replaced as needed.Updated September 2025
# Calculate the SOI index
def calculate_soi_indicators(row):
    """
   Calculate four SOI metrics
    """
    # Extract values A, B, H, and C from the row.
    A = row['A']
    B = row['B']
    H = row['H']
    C = row['C']

    # Calculate the SOI index
    SOI_sum = A + B + H + C
    SOI_multiply = A * B * H * C

    # Logarithmic calculations require ensuring all values are positive and taking the absolute value of each ln(x).
    try:
        # Sum the absolute values of each ln(x)
        SOI_log = abs(math.log(A)) + abs(math.log(B)) + abs(math.log(H)) + abs(math.log(C))
    except ValueError:
        SOI_log = np.nan  # If a negative value or zero occurs, set it to NaN.

    SOI_exp = math.exp(A) + math.exp(B) + math.exp(H) + math.exp(C)

    return pd.Series({
        'SOI_sum': SOI_sum,
        'SOI_multiply': SOI_multiply,
        'SOI_log': SOI_log,
        'SOI_exp': SOI_exp
    })


# Main
def main():
    # Specify the CSV file path
    input_file_path = ''  
    output_file_path = ''  

    try:
        # Read CSV file
        df = pd.read_csv(input_file_path)

        # Print column names for verification
        print("Column names in CSV files:", df.columns.tolist())

        # Check whether it contains the required columns
        required_columns = ['TreeID', 'A', 'B', 'H', 'C']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Error: The following columns are missing in the CSV file: {missing_columns}")
            return

        # Calculating the SOI Index
        soi_columns = df.apply(calculate_soi_indicators, axis=1)

        # Merge the SOI index into the raw data
        df_with_soi = pd.concat([df, soi_columns], axis=1)

        # Save to a new CSV file
        df_with_soi.to_csv(output_file_path, index=False)

        print(f"Calculation complete！The results have been saved to: {output_file_path}")
        print(f"added columns: {soi_columns.columns.tolist()}")

        # Display the first few lines of results as a preview.
        print("\nTop 5 Results Preview:")
        print(df_with_soi.head())

    except FileNotFoundError:
        print(f"Error: File not found {input_file_path}")
    except Exception as e:
        print(f"Error occurred during processing: {str(e)}")


if __name__ == "__main__":
    main()