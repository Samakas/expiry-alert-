import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import pytz
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from PIL import Image
import pytesseract
from pdf2image import convert_from_bytes
import tempfile
import re
from dotenv import load_dotenv
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import schedule
import time
import threading
import json
from pymongo import MongoClient
import bcrypt
import uuid
from streamlit_cookies_manager import CookieManager
import platform, shutil

# ---------------------------
# Environment & Config Loader
# ---------------------------

# Load .env locally
load_dotenv()

# Load Streamlit Cloud/Render secrets if available
try:
    if hasattr(st, "secrets") and st.secrets:
        for k, v in st.secrets.items():
            os.environ[str(k)] = str(v)
except Exception:
    pass

# Streamlit Page Config
st.set_page_config(
    page_title="Grocery Expiry Alert System",
    page_icon="🛒",
    layout="wide"
)

# ---------------------------
# Tesseract Path (Cross-Platform)
# ---------------------------
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
else:
    pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")

# ---------------------------
# MongoDB Init
# ---------------------------
def init_mongodb():
    try:
        mongodb_uri = os.getenv('MONGODB_URI', 'mongodb://127.0.0.1:27017/expiry-alert')
        client = MongoClient(mongodb_uri)
        db = client['grocery_app']
        return db
    except Exception as e:
        st.error(f"Error connecting to MongoDB: {e}")
        return None

# ---------------------------
# Cookies
# ---------------------------
cookies = CookieManager()

def init_auth():
    if not cookies.ready():
        st.error("Cookies not ready. Please refresh the page.")
        return False
    return True

# ---------------------------
# User Authentication
# ---------------------------
def create_user(db, email, password, name):
    users_collection = db['users']
    if users_collection.find_one({'email': email}):
        return False, "User already exists"
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    user = {
        '_id': str(uuid.uuid4()),
        'email': email,
        'password': hashed_password,
        'name': name,
        'created_at': datetime.now(),
        'email_verified': False
    }
    users_collection.insert_one(user)
    send_welcome_email(email, name)
    return True, "User created successfully"

def verify_user(db, email, password):
    users_collection = db['users']
    user = users_collection.find_one({'email': email})
    if user and bcrypt.checkpw(password.encode('utf-8'), user['password']):
        return True, user
    return False, None

def send_welcome_email(email, name):
    subject = "Welcome to Grocery Expiry Alert System"
    body = f"""
    <html>
    <body>
        <h2>Welcome, {name}!</h2>
        <p>Thank you for registering with the Grocery Expiry Alert System.</p>
        <p>You will now receive alerts about your grocery items before they expire.</p>
        <br>
        <p>Best regards,<br>Grocery Expiry Alert System Team</p>
    </body>
    </html>
    """
    send_email(email, subject, body)

# ---------------------------
# SQLite per-user DB
# ---------------------------
def init_db(user_id):
    db_path = f'grocery_{user_id}.db'
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS items
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 name TEXT NOT NULL,
                 purchase_date DATE NOT NULL,
                 expiry_date DATE NOT NULL,
                 notified INTEGER DEFAULT 0)''')
    conn.commit()
    conn.close()
    return db_path

# ---------------------------
# ML Expiry Predictor
# ---------------------------
class ExpiryPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.features = ['item_name', 'storage_type', 'purchase_month']

    def train_model(self, data_path='data/shelf_life.csv'):
        try:
            df = pd.read_csv(data_path)
            df['purchase_month'] = pd.to_datetime(df['purchase_date']).dt.month
            df['storage_type'] = df['storage_type'].fillna('pantry')
            for feature in ['item_name', 'storage_type']:
                le = LabelEncoder()
                df[feature] = le.fit_transform(df[feature])
                self.label_encoders[feature] = le
            X = df[['item_name', 'storage_type', 'purchase_month']]
            y = df['shelf_life_days']
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X, y)
            with open('expiry_model.pkl', 'wb') as f:
                pickle.dump({'model': self.model, 'label_encoders': self.label_encoders}, f)
            return True
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False

    def load_model(self):
        try:
            if os.path.exists('expiry_model.pkl'):
                with open('expiry_model.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.label_encoders = data['label_encoders']
                return True
            return False
        except:
            return False

    def predict_expiry(self, item_name, purchase_date, storage_type='pantry'):
        if not self.model:
            if not self.load_model():
                default_shelf_life = {
                    'milk': 7, 'bread': 5, 'eggs': 21, 'yogurt': 14, 'cheese': 30,
                    'chicken': 3, 'beef': 5, 'fish': 2, 'apples': 30, 'bananas': 7,
                    'oranges': 21, 'tomatoes': 7, 'lettuce': 5, 'carrots': 21,
                    'potatoes': 90, 'onions': 60, 'broccoli': 7, 'spinach': 5
                }
                item_lower = item_name.lower()
                for item, days in default_shelf_life.items():
                    if item in item_lower:
                        return purchase_date + timedelta(days=days)
                return purchase_date + timedelta(days=7)

        try:
            purchase_month = purchase_date.month
            item_encoded = self.label_encoders['item_name'].transform([item_name])[0]
            storage_encoded = self.label_encoders['storage_type'].transform([storage_type])[0]
            features = np.array([[item_encoded, storage_encoded, purchase_month]])
            shelf_life_days = self.model.predict(features)[0]
            return purchase_date + timedelta(days=shelf_life_days)
        except:
            return purchase_date + timedelta(days=7)

predictor = ExpiryPredictor()

# ---------------------------
# OCR & PDF Parsing
# ---------------------------
def extract_text_from_image(image):
    try:
        return pytesseract.image_to_string(image)
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def extract_text_from_pdf(pdf_file):
    try:
        images = convert_from_bytes(pdf_file.read())
        return "\n".join([extract_text_from_image(img) for img in images])
    except Exception as e:
        st.error(f"PDF Processing Error: {e}")
        return ""

def parse_receipt_text(text):
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',
        r'\d{1,2}-\d{1,2}-\d{2,4}',
        r'\d{1,2}\s+(Jan|Feb|Mar|Apr|...|Dec)\s+\d{2,4}',
        r'(Jan|Feb|Mar|Apr|...|Dec)\s+\d{1,2},\s+\d{2,4}'
    ]
    dates, purchase_date = [], None
    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        dates.extend(matches)
    for date_str in dates:
        try:
            purchase_date = datetime.strptime(date_str, '%m/%d/%Y')
            break
        except:
            try:
                purchase_date = datetime.strptime(date_str, '%m-%d-%Y')
                break
            except:
                continue
    if not purchase_date:
        purchase_date = datetime.now()

    items, common_items = [], [
        'milk','bread','eggs','yogurt','cheese','chicken','beef','fish',
        'apples','bananas','oranges','tomatoes','lettuce','carrots','potatoes',
        'onions','broccoli','spinach','rice','pasta','cereal','juice','soda'
    ]
    for line in text.split('\n'):
        for item in common_items:
            if item in line.lower() and len(line.strip()) < 50:
                items.append(item.capitalize()); break
    return purchase_date, list(set(items))

# ---------------------------
# Email Functions
# ---------------------------
def send_email(to_email, subject, body):
    try:
        from_email = os.getenv('EMAIL_FROM')
        password = os.getenv('EMAIL_PASSWORD')
        smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.getenv('SMTP_PORT', 465))
        if not from_email or not password:
            st.error("Email credentials not configured. Please check secrets.")
            return False
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = from_email, to_email, subject
        msg.attach(MIMEText(body, 'html'))
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(from_email, password)
        server.sendmail(from_email, to_email, msg.as_string())
        server.quit()
        return True
    except Exception as e:
        st.error(f"Error sending email: {e}")
        return False

# ---------------------------
# Scheduler (Guarded)
# ---------------------------
def run_scheduler(db):
    users_collection = db['users']
    users = users_collection.find({})
    for user in users:
        schedule.every().day.at("09:00").do(check_and_notify, user_id=user['_id'], user_email=user['email'])
    while True:
        schedule.run_pending()
        time.sleep(60)

def start_scheduler(db):
    if st.session_state.get("scheduler_started"):
        return
    st.session_state["scheduler_started"] = True
    threading.Thread(target=run_scheduler, args=(db,), daemon=True).start()

# ---------------------------
# Main app functions...
# ---------------------------
# (⚠️ keep your existing main_app(), auth_page(), check_and_notify() etc. unchanged)

# ---------------------------
# Entry Point
# ---------------------------
def main():
    if not init_auth():
        return
    db = init_mongodb()
    if not db:
        st.error("Failed to connect to database.")
        return
    user_id = cookies.get('user_id')
    user_email = cookies.get('user_email')
    user_name = cookies.get('user_name')
    if user_id and user_email and user_name:
        main_app(db, user_id, user_email, user_name)
    else:
        auth_page(db)

if __name__ == "__main__":
    main()
