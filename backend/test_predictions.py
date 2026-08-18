import os
import requests

# 1. Configuration
BASE_URL = "http://127.0.0.1:8000"
USERNAME = "test_user"
PASSWORD = "Test@1234"

# Use high-quality pre-packaged sample images in the repository (no internet/429 issues!)
LOCAL_IMAGES_DIR = "d:/Skin Disease clone/frontend/img/diseases"
LOCAL_IMAGES = {
    "Actinic Keratoses": "Actinic Keratoses.jpg",
    "Basal Cell Carcinoma": "Basal Cell Carcinoma.jpg",
    "Benign Keratosis": "Benign Keratosis.jpg",
    "Dermatofibroma": "Dermatofibroma.jpg",
    "Melanoma": "Melanoma.jpg",
    "Nevi": "Nevi.jpg",
    "Vascular": "Vascular Lesions.jpg"
}

print("1. Logging in to obtain JWT token...")
try:
    login_resp = requests.post(
        f"{BASE_URL}/login",
        data={"username": USERNAME, "password": PASSWORD}
    )
    if login_resp.status_code == 200:
        token = login_resp.json()["access_token"]
        print("Successfully obtained token!")
    else:
        print(f"Failed to log in: {login_resp.status_code} - {login_resp.text}")
        exit(1)
except Exception as e:
    print(f"Error logging in: {e}")
    exit(1)

auth_headers = {
    "Authorization": f"Bearer {token}"
}

results = []

print("\n2. Testing local pre-packaged skin disease images...")
for label, filename in LOCAL_IMAGES.items():
    local_path = os.path.join(LOCAL_IMAGES_DIR, filename)
    print(f"\n--- Testing: {label} ---")
    print(f"Local Path: {local_path}")
    
    if not os.path.exists(local_path):
        print(f"File not found: {local_path}. Skipping.")
        continue
        
    print(f"Uploading to /predict endpoint...")
    try:
        with open(local_path, "rb") as f:
            files = {"file": (filename, f, "image/jpeg")}
            pred_resp = requests.post(f"{BASE_URL}/predict", headers=auth_headers, files=files)
            
            if pred_resp.status_code == 200:
                data = pred_resp.json()
                pred_class = data.get("prediction")
                confidence = data.get("confidence")
                print(f"API Response:")
                print(f"  Prediction: {pred_class}")
                print(f"  Confidence: {confidence}%")
                
                # Check correctness
                correct = "Unknown"
                if label == "Melanoma" and "Melanoma" in pred_class:
                    correct = "Yes (Correct)"
                elif label == "Basal Cell Carcinoma" and "Basal Cell Carcinoma" in pred_class:
                    correct = "Yes (Correct)"
                elif label == "Nevi" and "Nevi" in pred_class:
                    correct = "Yes (Correct)"
                elif label == "Vascular" and "Vascular" in pred_class:
                    correct = "Yes (Correct)"
                elif label == "Benign Keratosis" and "Benign Keratosis" in pred_class:
                    correct = "Yes (Correct)"
                elif label == "Dermatofibroma" and "Dermatofibroma" in pred_class:
                    correct = "Yes (Correct)"
                elif label == "Actinic Keratoses" and "Actinic Keratoses" in pred_class:
                    correct = "Yes (Correct)"
                else:
                    correct = "No (Incorrect)"
                    
                print(f"  Correctness: {correct}")
                results.append({
                    "Label": label,
                    "Prediction": pred_class,
                    "Confidence": confidence,
                    "Correct": correct
                })
            else:
                print(f"Prediction failed: {pred_resp.status_code} - {pred_resp.text}")
    except Exception as e:
        print(f"Prediction error: {e}")

print("\n" + "=" * 100)
print("=== SUMMARY OF REAL-TIME LOCAL MODEL PREDICTIONS ===")
print("=" * 100)
print(f"{'Actual Label':<25} | {'Predicted Label':<40} | {'Confidence':<10} | {'Correct?':<15}")
print("-" * 100)
for res in results:
    print(f"{res['Label']:<25} | {res['Prediction']:<40} | {res['Confidence']:<10}% | {res['Correct']:<15}")
print("=" * 100)
