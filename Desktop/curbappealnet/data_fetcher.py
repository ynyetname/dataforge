import pandas as pd
import requests
import os
import time

api_key = "sk.eyJ1IjoieW55ZXRuYW1lIiwiYSI6ImNtamd4bHhreTFjYmozZ3Noem15Z2w0cDkifQ.murMJjGnbNZ1ANgoJ4JjdA"
train_image_dir = "C:\\dataset_curbappealnet\\satellite_ima\\train_images"
test_image_dir = "C:\\dataset_curbappealnet\\satellite_ima\\test_images"

train_curbappeal_csv = "C:\\dataset_curbappealnet\\train_curbappealnet.csv"
test_curbappeal_csv = "C:\\dataset_curbappealnet\\test_curbappealnet.csv"

zoom_level = 17
image_size = "700x700"

def fetch_image(lat, long, house_id, image_dir):  
    try:                                  
        lat = float(lat)
        long = float(long)
    except (ValueError, TypeError):  
        print(f"Skipped {house_id}: Invalid coordinates (lat={lat}, lon={long})")
        return
    
    if not (-90 <= lat <= 90) or not (-180 <= long <= 180):
        print(f"Skipped {house_id}: Out of range coordinates (lat={lat}, lon={long})")
        return
    
    filename = f"{image_dir}/{house_id}.jpg"
    
    if os.path.exists(filename):
        print(f"Skipping {house_id}, already exists.")
        return

    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{long},{lat},{zoom_level},0,0/{image_size}@2x?access_token={api_key}"

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

def process_dataset(csv_path, image_dir, dataset_name):
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    if not os.path.exists(csv_path):
        print(f"Error: File not found at {csv_path}")
        return

    print(f"Loading {dataset_name} CSV dataset")
    df = pd.read_csv(csv_path)
    print(f"Columns found: {df.columns.tolist()}")
    print(f"Found {len(df)} properties. Starting download")
    
    for index, row in df.iterrows():
        try:
            lat = row['lat']
            lon = row['long'] 
            house_id = row['id']
            
            fetch_image(lat, lon, house_id, image_dir)
            time.sleep(0.01)
            
        except KeyError as e:
            print(f"Column name error: {e}. Check your CSV headers.")
            break
        except Exception as e:
            print(f"Row {index} error: {e}")
            continue

def main():
    
    process_dataset(train_curbappeal_csv, train_image_dir, "train")
    
    process_dataset(test_curbappeal_csv, test_image_dir, "test")

if __name__ == "__main__":
    main()
