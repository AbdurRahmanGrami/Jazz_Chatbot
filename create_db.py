import pandas as pd
import sqlite3

# Load the CSV file
csv_file = "dummy_db.csv"  # Make sure this file is in the same directory as the script
df = pd.read_csv(csv_file)

# Remove any unnamed index column if present
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Create SQLite database and insert data
db_file = "jazz_telco.db"
conn = sqlite3.connect(db_file)
df.to_sql("users", conn, if_exists="replace", index=False)
conn.close()

print(f"Database '{db_file}' created successfully with table 'users'.")
