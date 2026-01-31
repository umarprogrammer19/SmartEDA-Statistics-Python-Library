import pandas as pd
import sys
from src.eda.full_eda import full_eda


def main():
    print("\n[EDA] SMART EDA LIBRARY - AUTO EDA STARTED\n")

    # -----------------------------
    # 1. Load your dataset here
    # -----------------------------
    try:
        df = pd.read_csv("data.csv")
        print("[OK] Dataset loaded successfully.")
    except FileNotFoundError:
        print("[ERROR] data.csv not found.")
        print("Please place your dataset in the same folder as main.py")
        return

    # -----------------------------
    # 2. Ask user for target column
    # -----------------------------
    print("\nColumns in your dataset:")
    print(list(df.columns))

    # Get target column from command line argument or user input
    if len(sys.argv) > 1:
        target = sys.argv[1]
        print(f"\nUsing target column from command line: {target}")
    else:
        target = input("\nEnter target column name: ").strip()

    if target not in df.columns:
        print(f"[ERROR] '{target}' does not exist in dataset.")
        return

    # -----------------------------
    # 3. Run automated EDA
    # -----------------------------
    print("\n[INFO] Running full EDA... Please wait...\n")

    clean_df, insights = full_eda(df, target=target)

    # -----------------------------
    # 4. Print Insights
    # -----------------------------
    print("\n[SUMMARY] EDA Insights Summary:\n")

    for key, value in insights.items():
        print(f"{key.upper()}:")
        print(value)
        print("-" * 50)

    # -----------------------------
    # 5. Save cleaned dataset
    # -----------------------------
    clean_df.to_csv("cleaned_output.csv", index=False)
    print("\n[SAVE] Cleaned dataset saved as: cleaned_output.csv")

    print("\n[COMPLETE] EDA Complete.\n")


if __name__ == "__main__":
    main()
