import os
import sys
import urllib.request
import bz2

URL = "https://raw.githubusercontent.com/davisking/dlib-models/master/shape_predictor_68_face_landmarks.dat.bz2"
OUTPUT_FILE = "shape_predictor_68_face_landmarks.dat"

def download_model():
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 90_000_000:
        print(f"'{OUTPUT_FILE}' already exists ({os.path.getsize(OUTPUT_FILE)} bytes).")
        return

    print(f"Downloading facial landmark predictor from:\n  {URL}...")
    req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
    decompressor = bz2.BZ2Decompressor()

    with urllib.request.urlopen(req) as response, open(OUTPUT_FILE, "wb") as out_file:
        total = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            decompressed = decompressor.decompress(chunk)
            if decompressed:
                out_file.write(decompressed)
            total += len(chunk)
            print(f"\rDownloaded {total // (1024 * 1024)} MB compressed...", end="", flush=True)

    print(f"\nSuccessfully downloaded and saved {OUTPUT_FILE} ({os.path.getsize(OUTPUT_FILE)} bytes).")

if __name__ == "__main__":
    download_model()
