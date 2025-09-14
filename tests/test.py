import os
import requests

# Load environment variables (optional, or replace directly)
INFOBIP_API_URL = os.getenv("INFOBIP_SEND_API", "https://ppj2nm.api.infobip.com/whatsapp/1/message/text")
INFOBIP_API_KEY = os.getenv("INFOBIP_API_KEY", "your_infobip_api_key_here")

def send_whatsapp_message(sender_number: str, recipient_number: str, message: str):
    headers = {
        "Authorization": f"App {INFOBIP_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload = {
        "from": sender_number,       # Your Infobip WhatsApp sender number
        "to": recipient_number,      # Recipient WhatsApp number (with country code)
        "content": {
            "text": message
        }
    }

    try:
        response = requests.post(INFOBIP_API_URL, headers=headers, json=payload)
        response.raise_for_status()
        print("✅ Message sent successfully:", response.json())
    except requests.exceptions.HTTPError as http_err:
        print(f"❌ HTTP error: {http_err} - {response.text}")
    except Exception as err:
        print(f"❌ Other error: {err}")


if __name__ == "__main__":
    # Example usage
    sender = "441134960000"       # Replace with your Infobip WhatsApp sender ID
    recipient = "2547XXXXXXXX"    # Replace with recipient WhatsApp number
    text_message = "Hello from Infobip WhatsApp API 🚀"

    send_whatsapp_message(sender, recipient, text_message)
