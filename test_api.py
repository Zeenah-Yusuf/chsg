import requests
import os

BASE_URL = "http://127.0.0.1:8000"

# Get the folder where this script lives
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def test_ping():
    r = requests.get(f"{BASE_URL}/ping")
    print("\n=== Ping ===")
    print("Status:", r.status_code)
    print("Response:", r.json() if r.ok else r.text)

def test_ingest_text():
    payload = {"text": "cholera outbreak near river", "lat": 6.49, "lon": 3.38}
    r = requests.post(f"{BASE_URL}/ingest/text", json=payload)
    print("\n=== Ingest Text ===")
    print("Status:", r.status_code)
    print("Response:", r.json() if r.ok else r.text)

def test_predict_combined():
    payload = {
        "Household Water Source": "Borehole",
        "Location of households:Latitude": 6.49,
        "Location of households:Longitude": 3.38,
        "UnsafeWater": 1,
    }
    r = requests.post(f"{BASE_URL}/predict/combined", json=payload)
    print("\n=== Predict Combined ===")
    print("Status:", r.status_code)
    print("Response:", r.json() if r.ok else r.text)

def test_latest_risk():
    r = requests.get(f"{BASE_URL}/risk/latest")
    print("\n=== Latest Risk ===")
    print("Status:", r.status_code)
    print("Response:", r.json() if r.ok else r.text)

def test_ingest_voice():
    file_path = os.path.join(SCRIPT_DIR, "sample.wav")
    print("\n=== Ingest Voice ===")
    if not os.path.exists(file_path):
        print("Skipped: sample.wav not found")
        return
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"lat": 6.49, "lon": 3.38, "language": "yo"}
        r = requests.post(f"{BASE_URL}/ingest/voice", files=files, data=data)
        print("Status:", r.status_code)
        print("Response:", r.json() if r.ok else r.text)

def test_ingest_image():
    file_path = os.path.join(SCRIPT_DIR, "sample.jpeg")
    print("\n=== Ingest Image ===")
    if not os.path.exists(file_path):
        print("Skipped: sample.jpeg not found")
        return
    with open(file_path, "rb") as f:
        files = {"file": f}
        data = {"lat": 6.49, "lon": 3.38}
        r = requests.post(f"{BASE_URL}/ingest/image", files=files, data=data)
        print("Status:", r.status_code)
        print("Response:", r.json() if r.ok else r.text)

if __name__ == "__main__":
    test_ping()
    test_ingest_text()
    test_predict_combined()
    test_latest_risk()
    test_ingest_voice()
    test_ingest_image()
