# api/alerts.py
import os
# prefer python-dotenv if available; import it dynamically to avoid static analyzer errors
import importlib.util
import importlib

def load_dotenv(dotenv_path=".env"):
    spec = importlib.util.find_spec("dotenv")
    if spec is not None:
        try:
            module = importlib.import_module("dotenv")
            if hasattr(module, "load_dotenv"):
                # delegate to python-dotenv's loader when available
                module.load_dotenv(dotenv_path)
                return
        except Exception:
            # fall through to lightweight fallback on any error
            pass

    # lightweight fallback: parse a .env file into os.environ if present
    try:
        with open(dotenv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k not in os.environ:
                        os.environ[k] = v
    except FileNotFoundError:
        pass

load_dotenv()

USE_STUB = os.getenv("USE_ALERT_STUB", "true").lower() == "true"

if not USE_STUB:
    from twilio.rest import Client
    ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    WHATSAPP_FROM = os.getenv("TWILIO_WHATSAPP_FROM")
    client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_alert(recipient: str, message: str):
    """
    Send WhatsApp alert via Twilio if configured, else print stub.
    """
    if USE_STUB:
        print(f"[STUB ALERT] To: {recipient} | Message: {message}")
        return {"status": "stub", "recipient": recipient, "message": message}
    else:
        response = client.messages.create(
            body=message,
            from_=os.getenv("TWILIO_WHATSAPP_FROM"),
            to=recipient
        )
        return {"status": "sent", "sid": response.sid}
