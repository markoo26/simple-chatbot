import sqlite3
import random
from datetime import datetime, timedelta

# Database file name
DB_NAME = "parking_system.db"

# Create connection and cursor
conn = sqlite3.connect(DB_NAME)
cursor = conn.cursor()

# Drop tables if they exist (for clean slate)
cursor.execute("DROP TABLE IF EXISTS bookings")
cursor.execute("DROP TABLE IF EXISTS prices")

# Create prices table
cursor.execute("""
CREATE TABLE prices (
    parking_id INTEGER PRIMARY KEY,
    eur_price_per_day REAL NOT NULL
)
""")

# Create bookings table
cursor.execute("""
CREATE TABLE bookings (
    booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
    parking_id INTEGER NOT NULL,
    booking_start_date TEXT NOT NULL,
    booking_end_date TEXT NOT NULL,
    total_price REAL NOT NULL,
    FOREIGN KEY (parking_id) REFERENCES prices(parking_id)
)
""")

print("Creating prices table with 100 parking spots...")

# Insert 100 parking spots with varying prices
# Prices range from 5 to 50 EUR per day
for parking_id in range(1, 101):
    # Create some price variety: budget (5-15), standard (15-30), premium (30-50)
    if parking_id <= 40:  # Budget spots
        price = round(random.uniform(5, 15), 2)
    elif parking_id <= 80:  # Standard spots
        price = round(random.uniform(15, 30), 2)
    else:  # Premium spots
        price = round(random.uniform(30, 50), 2)
    
    cursor.execute("INSERT INTO prices (parking_id, eur_price_per_day) VALUES (?, ?)",
                   (parking_id, price))

print("Prices table populated with 100 records.")

print("\nCreating bookings table with realistic booking data...")

# Generate bookings
# Let's create bookings spanning the last 6 months and next 3 months
start_date = datetime.now() - timedelta(days=180)
end_date = datetime.now() + timedelta(days=90)

bookings_created = 0

# Create multiple bookings for random parking spots
for _ in range(300):  # Generate 300 bookings
    # Random parking spot
    parking_id = random.randint(1, 100)
    
    # Get the price for this parking spot
    cursor.execute("SELECT eur_price_per_day FROM prices WHERE parking_id = ?", (parking_id,))
    price_per_day = cursor.fetchone()[0]
    
    # Random booking start date within our range
    random_days = random.randint(0, 270)
    booking_start = start_date + timedelta(days=random_days)
    
    # Random booking duration (1 to 14 days)
    duration_days = random.randint(1, 14)
    booking_end = booking_start + timedelta(days=duration_days)
    
    # Calculate total price (ceiling division for partial days)
    # For demo purposes, duration in days is simply the difference
    total_days = (booking_end - booking_start).days
    if total_days == 0:
        total_days = 1  # Minimum 1 day charge
    
    total_price = round(total_days * price_per_day, 2)
    
    # Format dates as strings (YYYY-MM-DD)
    start_str = booking_start.strftime("%Y-%m-%d")
    end_str = booking_end.strftime("%Y-%m-%d")
    
    # Insert booking
    cursor.execute("""
        INSERT INTO bookings (parking_id, booking_start_date, booking_end_date, total_price)
        VALUES (?, ?, ?, ?)
    """, (parking_id, start_str, end_str, total_price))
    
    bookings_created += 1

print(f"Bookings table populated with {bookings_created} records.")

# Commit changes
conn.commit()

# Display some sample data
print("\n" + "="*60)
print("SAMPLE DATA FROM PRICES TABLE:")
print("="*60)
cursor.execute("SELECT * FROM prices LIMIT 10")
prices_sample = cursor.fetchall()
print(f"{'Parking ID':<12} {'EUR/Day':<10}")
print("-"*60)
for row in prices_sample:
    print(f"{row[0]:<12} {row[1]:<10.2f}")

print("\n" + "="*60)
print("SAMPLE DATA FROM BOOKINGS TABLE:")
print("="*60)
cursor.execute("""
    SELECT b.booking_id, b.parking_id, b.booking_start_date, 
           b.booking_end_date, b.total_price, p.eur_price_per_day
    FROM bookings b
    JOIN prices p ON b.parking_id = p.parking_id
    LIMIT 10
""")
bookings_sample = cursor.fetchall()
print(f"{'Booking ID':<12} {'Parking ID':<12} {'Start Date':<12} {'End Date':<12} {'Total':<10} {'Daily Rate':<10}")
print("-"*60)
for row in bookings_sample:
    print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12} €{row[4]:<9.2f} €{row[5]:<9.2f}")

# Show some statistics
print("\n" + "="*60)
print("DATABASE STATISTICS:")
print("="*60)

cursor.execute("SELECT COUNT(*) FROM prices")
price_count = cursor.fetchone()[0]
print(f"Total parking spots: {price_count}")

cursor.execute("SELECT MIN(eur_price_per_day), MAX(eur_price_per_day), AVG(eur_price_per_day) FROM prices")
min_price, max_price, avg_price = cursor.fetchone()
print(f"Price range: €{min_price:.2f} - €{max_price:.2f} (avg: €{avg_price:.2f})")

cursor.execute("SELECT COUNT(*) FROM bookings")
booking_count = cursor.fetchone()[0]
print(f"Total bookings: {booking_count}")

cursor.execute("SELECT MIN(total_price), MAX(total_price), AVG(total_price) FROM bookings")
min_total, max_total, avg_total = cursor.fetchone()
print(f"Booking totals: €{min_total:.2f} - €{max_total:.2f} (avg: €{avg_total:.2f})")

print("\n" + "="*60)
print(f"Database '{DB_NAME}' created successfully!")
print("="*60)

# Close connection
conn.close()
