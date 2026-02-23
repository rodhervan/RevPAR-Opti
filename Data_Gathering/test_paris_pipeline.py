import os
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from amadeus import Client, ResponseError

# Load environment variables (searches current and parent dirs)
load_dotenv(find_dotenv())

DB_NAME = os.path.join(os.path.dirname(__file__), "hotels.db")

def get_amadeus_client():
    client_id = os.getenv('AMADEUS_CLIENT_ID')
    client_secret = os.getenv('AMADEUS_CLIENT_SECRET')
    if not client_id or not client_secret:
        # Fallback manual check for debugging
        print("DEBUG: Current CWD:", os.getcwd())
        print("DEBUG: Env locations checked:", find_dotenv())
        raise ValueError("API credentials missing. Please ensure .env file exists with AMADEUS_CLIENT_ID and AMADEUS_CLIENT_SECRET.")
    return Client(client_id=client_id, client_secret=client_secret)

def init_db():
    print(f"[DB] Initializing {DB_NAME}...")
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # 1. Hotels Table (The Master List)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS hotels (
            id TEXT PRIMARY KEY,
            name TEXT,
            city_code TEXT,
            latitude REAL,
            longitude REAL,
            fetched_at DATETIME
        )
    ''')
    
    # 2. Prices Table (The Harvested Data)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            hotel_id TEXT,
            check_in DATE,
            check_out DATE,
            amount REAL,
            currency TEXT,
            room_category TEXT,
            description TEXT,
            recorded_at DATETIME,
            FOREIGN KEY(hotel_id) REFERENCES hotels(id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("[DB] Tables ready.")

def step1_harvest_hotel_ids(city_code):
    print(f"\n[Step 1] Harvesting Hotel IDs for {city_code}...")
    try:
        amadeus = get_amadeus_client()
        
        response = amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
        
        if not response.data:
            print("  No hotels found.")
            return

        hotels = response.data
        print(f"  Found {len(hotels)} hotels from API.")
        
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        
        count_new = 0
        for h in hotels:
            hotel_id = h.get('hotelId')
            name = h.get('name')
            geo = h.get('geoCode', {})
            lat = geo.get('latitude')
            lon = geo.get('longitude')
            
            try:
                cursor.execute('''
                    INSERT OR IGNORE INTO hotels (id, name, city_code, latitude, longitude, fetched_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (hotel_id, name, city_code, lat, lon, datetime.now()))
                if cursor.rowcount > 0:
                    count_new += 1
            except sqlite3.Error as e:
                print(f"  DB Error: {e}")
                
        conn.commit()
        conn.close()
        print(f"  Saved {count_new} NEW hotels to DB.")
        
    except ResponseError as error:
        print(f"  Amadeus Error: {error}")

def step2_harvest_prices_batch(limit=1):
    print(f"\n[Step 2] Harvesting Prices (Batch Limit: {limit} hotels)...")
    
    # 1. Get IDs from DB
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM hotels ORDER BY fetched_at ASC LIMIT ?", (limit,))
    rows = cursor.fetchall()
    hotel_ids = [row[0] for row in rows]
    conn.close()
    
    if not hotel_ids:
        print("  No hotels in DB to price. Run Step 1 first.")
        return

    print(f"  Selected {len(hotel_ids)} hotels for pricing check: {hotel_ids}")
    
    # 2. Call API (Batch)
    amadeus = get_amadeus_client()
    ids_str = ",".join(hotel_ids)
    
    # Function to process a single hotel or a batch
    def fetch_ids(ids_to_fetch, is_batch=True):
        ids_str = ",".join(ids_to_fetch)
        try:
            print(f"  Calling 'hotel_offers_search' for {len(ids_to_fetch)} hotels (Batch={is_batch})...")
            response = amadeus.shopping.hotel_offers_search.get(
                hotelIds=ids_str,
                adults='1',
                checkInDate='2026-06-01', # Future dates for better success
                checkOutDate='2026-06-05'
            )
            return response.data if response.data else []
        except ResponseError as e:
            if is_batch: 
                print(f"  Batch failed ({e.response.status_code}). Retrying individually...")
                return None # Signal batch failure
            else:
                # Individual failure is fine, just log it lightly
                # print(f"  Hotel {ids_to_fetch[0]} failed: {e}") 
                return []

    # Try Batch ONLY
    offers = fetch_ids(hotel_ids, is_batch=True)
    
    if offers is None:
        print("  Batch failed. Skipping to save cost (Fallback disabled).")
        return

    if not offers:
        print("  No offers returned.")
        return

    # 3. Save to DB
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    count_prices = 0
    for offer in offers:
        hotel_id = offer.get('hotel', {}).get('hotelId')
        
        for item in offer.get('offers', []):
            price_obj = item.get('price', {})
            room_obj = item.get('room', {})
            
            check_in = item.get('checkInDate')
            check_out = item.get('checkOutDate')
            amount = price_obj.get('total')
            currency = price_obj.get('currency')
            category = room_obj.get('typeEstimated', {}).get('category')
            desc = room_obj.get('description', {}).get('text')
            
            cursor.execute('''
                INSERT INTO prices (hotel_id, check_in, check_out, amount, currency, room_category, description, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (hotel_id, check_in, check_out, amount, currency, category, desc, datetime.now()))
            count_prices += 1
            
    conn.commit()
    conn.close()
    print(f"  SUCCESS: Saved {count_prices} price records to 'prices' table.")


def main():
    # A. Init
    init_db()
    
    # B. Harvest IDs (The "Net")
    step1_harvest_hotel_ids('PAR')
    
    # C. Harvest Prices (The "Harvester") - Just 1 batch
    step2_harvest_prices_batch(limit=100)

if __name__ == "__main__":
    main()
