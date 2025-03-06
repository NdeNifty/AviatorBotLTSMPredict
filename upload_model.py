import requests
import os
import argparse

def upload_model(model_path, endpoint_url):
    """
    Upload a model file to the specified endpoint.
    
    Args:
        model_path (str): Local path to the model file
        endpoint_url (str): URL of the /upload-model endpoint
    
    Returns:
        dict: Response from the server as a JSON object
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    files = {'file': open(model_path, 'rb')}
    
    try:
        response = requests.post(endpoint_url, files=files)
        response.raise_for_status()
        result = response.json()
        print("Upload successful!")
        print(f"Response: {result}")
        return result
    
    except requests.exceptions.RequestException as e:
        print(f"Upload failed: {e}")
        if response is not None and response.text:
            print(f"Server response: {response.text}")
        return None
    
    finally:
        files['file'].close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload a model file to a Render endpoint.")
    parser.add_argument("--model-path", default="model/best_lstm_model.pth", help="Path to the model file")
    parser.add_argument("--endpoint-url", default="https://aviatorbotltsmpredict.onrender.com/upload-model", help="URL of the upload endpoint")
    
    args = parser.parse_args()
    
    upload_model(args.model_path, args.endpoint_url)