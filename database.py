import sqlite3

conn = sqlite3.connect("skin_disease.db", check_same_thread=False)
cursor = conn.cursor()

# Create users table
cursor.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT
)
""")

# Create predictions table
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT,
    disease TEXT,
    confidence REAL,
    risk TEXT,
    suggestion TEXT,
    prediction_time TEXT
)
""")

conn.commit()

# Show all patient records
cursor.execute("SELECT * FROM predictions")
records = cursor.fetchall()

print("Patient Records:")
print(records)