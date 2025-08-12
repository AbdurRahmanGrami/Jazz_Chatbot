import sqlite3
from faker import Faker
import random
import pandas as pd

# Initialize Faker with Pakistani locale
fake = Faker('en_PK')

# Connect to the SQLite database
conn = sqlite3.connect('jazz_telco.db')
cursor = conn.cursor()

# Create the 'users' table if it doesn't exist
cursor.execute('''
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    phone_number TEXT NOT NULL,
    active_plan TEXT NOT NULL,
    balance REAL,
    usage_stats TEXT,
    email TEXT,
    cnic TEXT,
    region TEXT,
    city TEXT,
    signup_date TEXT,
    last_recharge_date TEXT,
    recharge_amount REAL,
    data_plan TEXT,
    call_minutes_used INTEGER,
    sms_sent INTEGER
)
''')

# Clear existing data and reset auto-increment
cursor.execute("DELETE FROM users")
cursor.execute("DELETE FROM sqlite_sequence WHERE name='users'")

# Define options
plans = ['Prepaid', 'Postpaid']
regions = ['Punjab', 'Sindh', 'Khyber Pakhtunkhwa', 'Balochistan']
data_plans = ['1GB', '5GB', '10GB', 'Unlimited']

# Insert 100 dummy records
for _ in range(100):
    name = fake.name()
    phone_number = fake.phone_number()
    active_plan = random.choice(plans)
    balance = round(random.uniform(0, 10000), 2)
    usage_stats = f"Calls: {random.randint(0, 1000)}, Data: {random.randint(0, 100)}GB, SMS: {random.randint(0, 500)}"
    email = fake.email()
    cnic = fake.unique.random_number(digits=13, fix_len=True)
    region = random.choice(regions)
    city = fake.city()
    signup_date = fake.date_this_decade()
    last_recharge_date = fake.date_between(start_date=signup_date, end_date='today')
    recharge_amount = round(random.uniform(50, 5000), 2)
    data_plan = random.choice(data_plans)
    call_minutes_used = random.randint(0, 10000)
    sms_sent = random.randint(0, 1000)

    cursor.execute('''
        INSERT INTO users (
            name, phone_number, active_plan, balance, usage_stats,
            email, cnic, region, city, signup_date, last_recharge_date,
            recharge_amount, data_plan, call_minutes_used, sms_sent
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        name, phone_number, active_plan, balance, usage_stats,
        email, cnic, region, city, signup_date, last_recharge_date,
        recharge_amount, data_plan, call_minutes_used, sms_sent
    ))

# Commit changes
conn.commit()

# Export to CSV
df = pd.read_sql_query("SELECT * FROM users", conn)
df.to_csv("dummy_db.csv", index=False)

# Close connection
conn.close()

print("Database reset, repopulated, and exported to 'users_export.csv'.")
