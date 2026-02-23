import os
import time
from dotenv import load_dotenv, find_dotenv
from amadeus import Client, ResponseError

# Load environment variables
load_dotenv(find_dotenv())

def main():
    try:
        amadeus = Client(
            client_id=os.getenv('AMADEUS_CLIENT_ID'),
            client_secret=os.getenv('AMADEUS_CLIENT_SECRET')
        )
        
        # Known working ID from CSV
        target_id = 'XKPAR120' 
        print(f"Testing pricing for known good ID: {target_id}")
        
        response = amadeus.shopping.hotel_offers_search.get(
            hotelIds=target_id,
            adults='1',
            checkInDate='2026-06-01',
            checkOutDate='2026-06-05'
        )
        
        if response.data:
            print(f"Success! Found {len(response.data)} offers.")
            print(response.data[0])
        else:
            print("No offers found, but request succeeded.")
            
    except ResponseError as error:
        print(f"Amadeus Error: {error}")
        try:
            print(f"Details: {error.response.body}")
        except:
            pass

if __name__ == "__main__":
    main()
