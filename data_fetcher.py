import pandas as pd
import requests
import os
import time

API_KEY = "sk.eyJ1IjoieW55ZXRuYW1lIiwiYSI6ImNtamd4bHhreTFjYmozZ3Noem15Z2w0cDkifQ.murMJjGnbNZ1ANgoJ4JjdA"
IMAGE_DIR = "../satellite_ima/images"

CSV_FILE = "C:\\Users\\Ayyan Aftab\\OneDrive\\Documents\\train_curbappealnet.csv"

ZOOM_LEVEL = 17
IMAGE_SIZE = "700x700"

def fetch_image(lat, long, house_id):  
    """
    Fetches a satellite image for a specific lat/long from Mapbox and saves it.
    """
    # Validate coordinates
    try:                                  
        lat = float(lat)
        long = float(long)
    except (ValueError, TypeError):  
        print(f"Skipped {house_id}: Invalid coordinates (lat={lat}, lon={long})")
        return
    
    # Check valid lat/long ranges
    if not (-90 <= lat <= 90) or not (-180 <= long <= 180):
        print(f"Skipped {house_id}: Out of range coordinates (lat={lat}, lon={long})")
        return
    
    filename = f"{IMAGE_DIR}/{house_id}.jpg"
    
    if os.path.exists(filename):
        print(f"Skipping {house_id}, already exists.")
        return

    # Format: /styles/v1/{username}/{style_id}/static/{lon},{lat},{zoom},{bearing},{pitch}/{width}x{height}@2x?access_token={key}
    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{long},{lat},{ZOOM_LEVEL},0,0/{IMAGE_SIZE}@2x?access_token={API_KEY}"

    try:
        response = requests.get(url, stream=True, timeout=10)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            print(f"Downloaded: {house_id}")
        else:
            print(f"Failed {house_id}: Status {response.status_code}")
            if response.status_code == 403:
                print(f"API Key issue. Response: {response.text[:200]}")
            
    except Exception as e:
        print(f"Error fetching {house_id}: {e}")

def main():
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)

    if not os.path.exists(CSV_FILE):
        print(f"Error: File not found at {CSV_FILE}")
        return

    print("Loading CSV dataset")
    
    df = pd.read_csv(CSV_FILE)
    
    print(f"Columns found: {df.columns.tolist()}")

    print(f"Found {len(df)} properties. Starting download")
    
    for index, row in df.iterrows():
        try:
      
            lat = row['lat']
            lon = row['long'] 
            house_id = row['id']
            
            fetch_image(lat, lon, house_id)
            time.sleep(0.01)  # increased delay to avoid rate limiting
            
        except KeyError as e:
            print(f"Column name error: {e}. Check your CSV headers.")
            break
        except Exception as e:
            print(f"Row {index} error: {e}")
            continue 

if __name__ == "__main__":
    main()