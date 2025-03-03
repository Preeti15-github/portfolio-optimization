import os
import pandas as pd

# ✅ CSV File Path
csv_file_path = "C:\\Users\\Preeti\\Documents\\portfolio_results.csv"  # Change path if needed

# ✅ Sample DataFrame (Agar CSV File Nahi Hai To Sample Data Banayenge)
if not os.path.exists(csv_file_path):
    sample_data = {"Stock": ["AAPL", "MSFT", "GOOGL"], "Return": [0.12, 0.15, 0.18], "Risk": [0.05, 0.07, 0.09]}
    df = pd.DataFrame(sample_data)
    df.to_csv(csv_file_path, index=False)
    print("✅ Sample CSV file created:", csv_file_path)
else:
    df = pd.read_csv(csv_file_path)

# ✅ Show First 5 Rows
print(df.head())
