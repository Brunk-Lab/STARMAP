import datetime
import json
import requests
import os

# Set variables
BASE_URL = 'https://api.figshare.com/v2'
ITEM_ID = 29190726   # <-- X-Atlas ID from figshare
OUTPUT_DIR = "../data/downloads"  # local folder for saving files
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Retrieve metadata
r = requests.get(f"{BASE_URL}/articles/{ITEM_ID}")
if r.status_code != 200:
    print("Something is wrong:", r.content)
    exit()

metadata = r.json()

# Look at metadata
print("Article title:", metadata["title"])
print("Number of files:", len(metadata.get("files", [])))

# Download all files
for f in metadata.get("files", []):
    file_name = f["name"]
    file_url = f["download_url"]

    print(f"Downloading {file_name} ...")
    resp = requests.get(file_url, stream=True)
    if resp.status_code == 200:
        outpath = os.path.join(OUTPUT_DIR, file_name)
        with open(outpath, "wb") as out:
            for chunk in resp.iter_content(chunk_size=8192):
                out.write(chunk)
        print(f"Saved to {outpath}")
    else:
        print(f"Failed to download {file_name}: {resp.status_code}")
