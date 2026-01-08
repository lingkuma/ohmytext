import base64
import requests
import json

def image_to_base64(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def test_ocr_api():
    api_url = 'http://127.0.0.1:2333/api/ocr'
    image_path = './test/test1.png'
    
    try:
        base64_image = image_to_base64(image_path)
        
        payload = {
            'image': base64_image
        }
        
        headers = {
            'Content-Type': 'application/json'
        }
        
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)
        
        print('Status Code:', response.status_code)
        print('Response:', json.dumps(response.json(), indent=2, ensure_ascii=False))
        
    except FileNotFoundError:
        print(f'Error: Image file not found at {image_path}')
    except requests.exceptions.RequestException as e:
        print(f'Request Error: {e}')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    test_ocr_api()
