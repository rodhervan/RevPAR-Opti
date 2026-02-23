import os
import sqlite3
import time
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from amadeus import Client, ResponseError

# Load environment variables
load_dotenv(find_dotenv())

DB_NAME = os.path.join(os.path.dirname(__file__), "hotels.db")

# IDs extracted from data/hotel_prices.csv
KNOWN_GOOD_IDS = [
    'BWPAR160',  # Best Western Gaillon Opera
    'BWPAR789',  # THE BW PREMIER FAUBOURG 88
    'FGPARPAL',  # HOTEL PRINCE ALBERT LOUVRE
    'HNPARKGU',  # Test Property
    'XKPAR120',  # Diamond Hotel
    'INNYCC96',  # Hotel Indigo Lower East Side
    'CYTYOCYC',  # Courtyard by Marriott Tokyo Ginza Hotel
    'ICTYOICC'   # INTERCONTINENTAL ANA TOKYO
]

def get_amadeus_client():
    client_id = os.getenv('AMADEUS_CLIENT_ID')
    client_secret = os.getenv('AMADEUS_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise ValueError("API credentials missing.")
    return Client(client_id=client_id, client_secret=client_secret)

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    # Ensure prices table exists
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
            recorded_at DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def main():
    print("=== Verifying Batch Pricing with Known Good IDs ===")
    init_db()
    
    amadeus = get_amadeus_client()
    
    # 1. Prepare Batch
    ids_str = ",".join(KNOWN_GOOD_IDS)
    print(f"Targeting {len(KNOWN_GOOD_IDS)} Hotels: {ids_str}")
    
    try:
        print("\n[Action] Sending SINGLE Batch Request...")
        start_time = time.time()
        
        response = amadeus.shopping.hotel_offers_search.get(
            hotelIds=ids_str,
            adults='1',
            checkInDate='2026-06-01',
            checkOutDate='2026-06-05'
        )
        
        elapsed = time.time() - start_time
        print(f"[Success] API responded in {elapsed:.2f}s")
        
        offers = response.data if response.data else []
        print(f"[Result] Received offers for {len(offers)} hotels.")
        
        if not offers:
            print("No offers returned.")
            return

        # 2. Verify we got multiple hotels back
        returned_ids = set()
        for offer in offers:
            h_id = offer.get('hotel', {}).get('hotelId')
            returned_ids.add(h_id)
            
        print(f"Hotels with offers: {returned_ids}")
        
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
        print(f"\n[DB] Saved {count_prices} price records to database.")
        print("Batch verification PASSED.")

    except ResponseError as error:
        print(f"\n[Failed] API Error: {error}")
        try:
            print(f"Details: {error.response.body}")
        except:
            pass

if __name__ == "__main__":
    main()
