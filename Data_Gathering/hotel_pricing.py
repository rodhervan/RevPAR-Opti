import os
import csv
import time
from datetime import datetime
from dotenv import load_dotenv
from amadeus import Client, ResponseError

# Load environment variables
load_dotenv()

def get_amadeus_client():
    client_id = os.getenv('AMADEUS_CLIENT_ID')
    client_secret = os.getenv('AMADEUS_CLIENT_SECRET')
    if not client_id or not client_secret:
        raise ValueError("API credentials missing in .env file")
    return Client(client_id=client_id, client_secret=client_secret)

def find_hotels_by_city(amadeus, city_code, limit=20):
    """
    Finds hotels in a city.
    Returns a list of hotel IDs.
    """
    print(f"\n[Search] Finding hotels in {city_code}...")
    try:
        response = amadeus.reference_data.locations.hotels.by_city.get(
            cityCode=city_code
        )
        if response.data:
            hotels = response.data
            print(f"  Found {len(hotels)} hotels in {city_code}.")
            # return the top 'limit' hotels to manage API quota
            return hotels[:limit]
        return []
    except ResponseError as error:
        print(f"  Error searching hotels in {city_code}: {error}")
        return []

def get_hotel_offers(amadeus, hotel_ids_list, check_in=None, check_out=None):
    """
    Fetches offers for a list of hotel IDs.
    Returns a list of offer objects.
    """
    if not hotel_ids_list:
        return []
    
    # Amadeus often prefers batches. Let's do a simple batch if needed, 
    # but for this script we assume list is small enough (< 50 usually safe).
    hotel_ids_str = ",".join(hotel_ids_list)
    print(f"[Pricing] Fetching offers for {len(hotel_ids_list)} hotels...")
    
    try:
        # Build arguments dictionary dynamically
        kwargs = {
            'hotelIds': hotel_ids_str,
            'adults': '1'
        }
        if check_in:
            kwargs['checkInDate'] = check_in
        if check_out:
            kwargs['checkOutDate'] = check_out
            
        response = amadeus.shopping.hotel_offers_search.get(**kwargs)
        
        if response.data:
            return response.data
        return []
    except ResponseError as error:
        print(f"  Error fetching offers: {error}")
        return []

def main():
    try:
        amadeus = get_amadeus_client()
    except ValueError as e:
        print(e)
        return

    cities = ['PAR', 'NYC', 'TYO']
    
    # --- Date Configuration ---
    # Set dates in 'YYYY-MM-DD' format. 
    # Example: target_check_in = '2024-12-01'
    # Leave as None to use API defaults (usually current date)
    target_check_in = None 
    target_check_out = None
    target_check_in = '2026-06-01'
    target_check_out = '2026-06-05'
    
    all_offers_data = []

    for city in cities:
        # 1. Find hotels
        matching_hotels = find_hotels_by_city(amadeus, city, limit=10) # Limiting to 10 for speed
        
        if not matching_hotels:
            continue

        # Extract IDs
        hotel_ids = [h['hotelId'] for h in matching_hotels]
        
        # 2. Get Offers
        # Note: In a real app, you might loop through batches of IDs if list is long
        offers = get_hotel_offers(amadeus, hotel_ids, check_in=target_check_in, check_out=target_check_out)
        
        print(f"  Retrieved {len(offers)} offers for {city}.")

        # 3. Process Data
        for offer in offers:
            hotel_name = offer.get('hotel', {}).get('name', 'Unknown')
            hotel_id = offer.get('hotel', {}).get('hotelId', 'Unknown')
            
            for item in offer.get('offers', []):
                # Extract simplified details
                price_obj = item.get('price', {})
                room_obj = item.get('room', {})
                desc_obj = room_obj.get('description', {})
                
                row = {
                    'City': city,
                    'HotelID': hotel_id,
                    'HotelName': hotel_name,
                    'CheckIn': item.get('checkInDate'),
                    'CheckOut': item.get('checkOutDate'),
                    'Price': price_obj.get('total'),
                    'Currency': price_obj.get('currency'),
                    'RoomCategory': room_obj.get('typeEstimated', {}).get('category'),
                    'Description': desc_obj.get('text', '')[:100] # Limit desc length
                }
                all_offers_data.append(row)
        
        # Be nice to the API
        time.sleep(1)

    # 4. Save to CSV
    if all_offers_data:
        csv_file = os.path.join("d:\\RODRIGO\\Tesis IIND\\data", "hotel_prices.csv")
        headers = all_offers_data[0].keys()
        
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(all_offers_data)
            print(f"\nSuccess! Saved {len(all_offers_data)} rows to {csv_file}")
        except Exception as e:
            print(f"Error saving CSV: {e}")
    else:
        print("\nNo offers found for any city.")

if __name__ == "__main__":
    main()
