import os
import sys
from dotenv import load_dotenv
from amadeus import Client, ResponseError, Location

# Load environment variables
load_dotenv()

def main():
    if len(sys.argv) < 2:
        print("Usage: python find_city_code.py \"City Name\"")
        print("Example: python find_city_code.py \"London\"")
        return

    keyword = sys.argv[1]

    # Initialize Client
    try:
        amadeus = Client(
            client_id=os.getenv('AMADEUS_CLIENT_ID'),
            client_secret=os.getenv('AMADEUS_CLIENT_SECRET')
        )
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    print(f"Searching for city code for: '{keyword}'...")

    try:
        response = amadeus.reference_data.locations.get(
            keyword=keyword,
            subType=Location.CITY
        )

        if response.data:
            print(f"\nFound {len(response.data)} matches:")
            print(f"{'City Name':<30} | {'IATA Code':<10} | {'Country'}")
            print("-" * 55)
            for location in response.data:
                name = location.get('name')
                code = location.get('iataCode')
                country = location.get('address', {}).get('countryName')
                print(f"{name:<30} | {code:<10} | {country}")
        else:
            print("No locations found.")

    except ResponseError as error:
        print(f"API Error: {error}")

if __name__ == "__main__":
    main()
