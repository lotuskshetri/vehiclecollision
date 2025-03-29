# twilio_notifier.py

from twilio.rest import Client

# Twilio credentials
TWILIO_ACCOUNT_SID = "xxxxxxxxxxxxxxxxxxxx"  # Replace with your Twilio Account SID
TWILIO_AUTH_TOKEN = "xxxxxxxxxxxxxxxxxxxxx"    # Replace with your Twilio Auth Token
TWILIO_WHATSAPP_NUMBER = "whatsapp:+xxxxxxxxxx"  # Twilio WhatsApp sandbox number
YOUR_WHATSAPP_NUMBER = "whatsapp:+xxxxxxxx"    # Replace with your WhatsApp number (include country code)

def send_whatsapp_notification(video_filename):
    """
    Sends a WhatsApp notification using Twilio.
    :param video_filename: The name of the video where the collision was detected.
    """
    try:
        # Initialize Twilio client
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

        # Compose the message
        message_body = f"⚠️ Collision detected in video `{video_filename}`!"

        # Send the WhatsApp message
        message = client.messages.create(
            body=message_body,
            from_=TWILIO_WHATSAPP_NUMBER,
            to=YOUR_WHATSAPP_NUMBER
        )

        print(f"WhatsApp notification sent for video `{video_filename}`. Message SID: {message.sid}")
    except Exception as e:
        print(f"Error sending WhatsApp notification: {e}")