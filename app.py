from flask import Flask, request, jsonify, render_template,redirect, url_for, session,send_file
import pickle
import numpy as np
from flask_cors import CORS
import sqlite3
from datetime import datetime , timedelta
import os
import threading
import uuid

app = Flask(__name__)
CORS(app,origins=["https://sisso.dgon.onrender.com"], supports_credentials=True)


@app.route('/loading')
def loading_screen():
    return render_template('loading.html')
    
app.secret_key = 'your_secret_key_here'

_mqtt_client = None
_mqtt_thread = None
_mqtt_running = False
_latest_iot = None

# Simple admin credentials (change in production)
ADMIN_USER = 'admin'
ADMIN_PASS = 'adminpass'
ADMIN_TOKEN = 'admintoken123'  # simple static token for demo



@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        mobile = request.form['mobile']
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id FROM users WHERE mobile = ?', (mobile,))
        user = cur.fetchone()
        conn.close()
        if user:
            session['user_id'] = user['id']
            return redirect(url_for('predict_page'))
        else:
            return render_template('login.html', error='Mobile not found')
    return render_template('login.html')
    
@app.route('/predict', methods=['POST'])
def predict():
    if not session.get('user_id'):
        return redirect(url_for('login_page'))

    try:
        N = float(request.form['N'])
        P = float(request.form['P'])
        K = float(request.form['K'])
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]

        if le:
            crop = le.inverse_transform([prediction])[0]
        else:
            crop = crop_dict.get(prediction, "Unknown")

        return render_template('result.html', crop=crop)

    except Exception as e:
        # If form is empty or error occurs, show input page again
        return render_template('smart_crop_recommendation_multilang.html', error=str(e))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('main_page'))






def admin_required(fn):
    def wrapper(*args, **kwargs):
        token = request.headers.get('X-Admin-Token')
        expiry = active_tokens.get(token)
        if token != ADMIN_TOKEN or not expiry or datetime.utcnow() > expiry:
            return jsonify({'error': 'unauthorized or session expired'}), 401
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


DB_PATH = os.path.join(os.path.dirname(__file__), 'users.db')

def get_db_connection():
    # add a timeout to wait for locks to clear (helps with "database is locked" errors)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    # initialize DB and enable WAL to reduce write locks
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute('PRAGMA journal_mode=WAL')
    except Exception:
        pass
    # users table stores registered users
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            mobile TEXT NOT NULL UNIQUE,
            address TEXT,
            created_at TEXT NOT NULL
        )
    ''')
    # predictions table stores inputs and results, optionally linked to a user
    cur.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            N REAL,
            P REAL,
            K REAL,
            temperature REAL,
            humidity REAL,
            ph REAL,
            rainfall REAL,
            predicted_crop TEXT,
            input_source TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    # If DB existed previously, ensure column `input_source` exists
    try:
        cur.execute("PRAGMA table_info(predictions)")
        cols = [r[1] for r in cur.fetchall()]
        if 'input_source' not in cols:
            cur.execute("ALTER TABLE predictions ADD COLUMN input_source TEXT")
    except Exception:
        pass
    conn.commit()
    conn.close()

# Load model and scaler (use file-relative paths so working dir doesn't matter)
MODEL_PATH = os.path.join(os.path.dirname(__file__),'model', 'model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__),'model', 'minmaxscaler.pkl')
model = None
scaler = None
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Warning: could not load model from {MODEL_PATH}: {e}")
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print(f"Warning: could not load scaler from {SCALER_PATH}: {e}")

LE_PATH = os.path.join(os.path.dirname(__file__),'model' ,'label_encoder.pkl')
le = None
try:
    with open(LE_PATH, 'rb') as f:
        le = pickle.load(f)
except Exception as e:
    print(f"Warning: could not load label encoder: {e}")

from tensorflow.keras.models import load_model
import cv2

CNN_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'cnn_model.h5')
ENV_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'env_model.pkl')

cnn_model = None
env_model = None
try:
    cnn_model = load_model(CNN_MODEL_PATH)
except Exception as e:
    print(f"Could not load CNN model: {e}")

try:
    with open(ENV_MODEL_PATH, 'rb') as f:
        env_model = pickle.load(f)
except Exception as e:
    print(f"Could not load environmental model: {e}")

# Crop dictionary (update as per your notebook)
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
    8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
    14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
    19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
}

@app.route('/api/predict_disease', methods=['POST'])
def predict_disease():
    crop_type = request.form.get('crop_type')
    location = request.form.get('location')
    image = request.files.get('leaf_image')

    if not crop_type or not location or not image:
        return jsonify({'error': 'Missing crop_type, location, or image'}), 400

    try:
        # Image preprocessing
        img_np = np.frombuffer(image.read(), np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        img = cv2.resize(img, (128, 128)) / 255.0
        img = np.expand_dims(img, axis=0)

        # CNN prediction
        cnn_pred = cnn_model.predict(img)[0]
        disease_label = np.argmax(cnn_pred)
        image_confidence = float(np.max(cnn_pred))

        # Environmental prediction (mocked for now)
        env_features = np.array([[30, 80, 120]])  # Replace with actual weather data
        env_risk_score = env_model.predict_proba(env_features)[0][1]  # Assuming binary classifier

        # Fusion
        final_score = 0.6 * image_confidence + 0.4 * env_risk_score
        risk_level = "High" if final_score > 0.7 else "Moderate" if final_score > 0.4 else "Low"

        # Suggestion (mocked)
        suggestion = "Use organic fungicide and monitor humidity."

        return jsonify({
            "disease": f"Disease {disease_label}",
            "confidence": round(final_score, 2),
            "risk_level": risk_level,
            "suggestion": suggestion
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts')
def get_alerts():
    mobile = request.args.get('mobile')
    if not mobile:
        return jsonify({'error': 'mobile required'}), 400

    # Mocked alert logic
    alerts = [
        {"disease": "Leaf Blight", "location": "Karchana", "level": "High"},
        {"disease": "Rust", "location": "Karchana", "level": "Moderate"}
    ]
    return jsonify({'alerts': alerts})


@app.route('/')
def main_page():
    return render_template('smart_crop_system_main.html')

@app.route('/predict-page')
def predict_page():
    return render_template('smart_crop_recommendation_multilang.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    name = data.get('name')
    mobile = data.get('mobile')
    address = data.get('address', '')
    if not name or not mobile:
        return jsonify({'error': 'name and mobile are required'}), 400
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('INSERT INTO users (name, mobile, address, created_at) VALUES (?,?,?,?)',
                    (name, mobile, address, datetime.utcnow().isoformat()))
        conn.commit()
        user_id = cur.lastrowid
        conn.close()
        return jsonify({'status': 'registered', 'user_id': user_id})
    except sqlite3.IntegrityError:
        return jsonify({'error': 'mobile already registered'}), 409
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    mobile = data.get('mobile')
    if not mobile:
        return jsonify({'error': 'mobile is required'}), 400
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, name, mobile, address, created_at FROM users WHERE mobile = ?', (mobile,))
    row = cur.fetchone()
    conn.close()
    if row:
        return jsonify(dict(row))
    return jsonify({'error': 'not found'}), 404


@app.route('/admin/dashboard')
def admin_dashboard():
    return render_template('admin.html')

@app.route('/admin/login')
def admin_login_page():
    return render_template('login.html')


active_tokens = {}  # token: expiry_time

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    data = request.get_json()
    user = data.get('username')
    pwd = data.get('password')
    if user == ADMIN_USER and pwd == ADMIN_PASS:
        expiry = datetime.utcnow() + timedelta(seconds=60)
        active_tokens[ADMIN_TOKEN] = expiry
        return jsonify({'token': ADMIN_TOKEN})
    return jsonify({'error': 'invalid credentials'}), 401

   

@app.route('/api/admin/users')
@admin_required
def admin_list_users():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id, name, mobile, address, created_at FROM users')
        rows = cur.fetchall()
        conn.close()
        return jsonify({'users': [dict(r) for r in rows]})
    except Exception as e:
        print("Error loading users:", e)
        return jsonify({'error': 'server error'}), 500



@app.route('/api/admin/user/<int:user_id>/delete', methods=['POST'])
@admin_required
def admin_delete_user(user_id):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('DELETE FROM users WHERE id = ?', (user_id,))
        cur.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
        conn.commit()
        conn.close()
        return jsonify({'status': 'deleted'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/iot/start', methods=['POST'])
@admin_required
def iot_start():
    global _mqtt_client, _mqtt_thread, _mqtt_running
    data = request.get_json() or {}
    broker = data.get('broker')
    topic = data.get('topic')
    if not broker or not topic:
        return jsonify({'error': 'broker and topic required'}), 400
    try:
        import paho.mqtt.client as mqtt
    except Exception:
        return jsonify({'error': 'paho-mqtt package not installed; run pip install paho-mqtt'}), 500
    if _mqtt_running:
        return jsonify({'status': 'already running'})
    client = mqtt.Client()
    client.on_message = _mqtt_on_message
    try:
        client.connect(broker)
        client.subscribe(topic)
        _mqtt_client = client
        _mqtt_running = True
        # run loop in a background thread
        _mqtt_thread = threading.Thread(target=client.loop_forever, daemon=True)
        _mqtt_thread.start()
        return jsonify({'status': 'started'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/iot/stop', methods=['POST'])
@admin_required
def iot_stop():
    global _mqtt_client, _mqtt_running
    try:
        if _mqtt_client:
            _mqtt_client.disconnect()
        _mqtt_running = False
        return jsonify({'status': 'stopped'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/iot/latest', methods=['GET'])
@admin_required
def iot_latest():
    if _latest_iot:
        return jsonify(_latest_iot)
    return jsonify({'error': 'no data yet'}), 404

@app.route('/api/admin/feedback')
@admin_required
def admin_feedback():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            SELECT f.actual_crop, f.created_at, u.mobile, p.input_source
            FROM feedback f
            LEFT JOIN users u ON f.user_id = u.id
            LEFT JOIN predictions p ON f.prediction_id = p.id
            ORDER BY f.created_at DESC
        ''')
        rows = cur.fetchall()
        conn.close()
        return jsonify({'feedback': [dict(r) for r in rows]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    try:
        N = float(data['N'])
        P = float(data['P'])
        K = float(data['K'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        print(f"Received input: N={N}, P={P}, K={K}, temperature={temperature}, humidity={humidity}, ph={ph}, rainfall={rainfall}")
        if scaler is None or model is None:
            return jsonify({'error': 'model or scaler not loaded on server'}), 500
        transformed = scaler.transform(features)
        # try to obtain probabilities if supported
        probs = None
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(transformed).tolist()
        except Exception:
            probs = None
        prediction = model.predict(transformed)[0]
        if le:
            crop = le.inverse_transform([prediction])[0]
        else:
            crop = crop_dict.get(prediction, "Unknown")

        print(f"Predicted crop: {crop}")

        # If mobile provided, link prediction to user
        mobile = data.get('mobile')
        user_id = None
        if mobile:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT id FROM users WHERE mobile = ?', (mobile,))
            r = cur.fetchone()
            if r:
                user_id = r['id']
            conn.close()

        # store prediction
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('''INSERT INTO predictions (user_id, N, P, K, temperature, humidity, ph, rainfall, predicted_crop, input_source, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
            (user_id, N, P, K, temperature, humidity, ph, rainfall, crop, 'api', datetime.utcnow().isoformat()))
            conn.commit()
            prediction_id = cur.lastrowid
            conn.close()
        except Exception as e:
            print(f"Warning: could not save prediction: {e}")

      
        # If client asked for debug info, return transformed vector and raw outputs
        if data.get('debug'):
            return jsonify({
                'crop': crop,
                'prediction_raw': int(prediction) if np.issubdtype(type(prediction), np.integer) else prediction,
                'transformed': transformed.tolist(),
                'probs': probs,
                'prediction_id': prediction_id if 'prediction_id' in locals() else None
            })
        return jsonify({'crop': crop, 'prediction_id': prediction_id if 'prediction_id' in locals() else None})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 400


def _mqtt_on_message(client, userdata, msg):
    global _latest_iot
    try:
        import json
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        _latest_iot = {'topic': msg.topic, 'data': data, 'timestamp': datetime.utcnow().isoformat()}
    except Exception:
        pass



@app.route('/api/feedback', methods=['POST'])
def feedback():
    """Record user feedback: actual crop name for a previous prediction.
    Expected JSON: { mobile: '...', prediction_id: <optional>, actual_crop: 'Mango' }
    """
    data = request.get_json() or {}
    mobile = data.get('mobile')
    actual_crop = data.get('actual_crop')
    prediction_id = data.get('prediction_id')
    if not mobile or not actual_crop:
        return jsonify({'error': 'mobile and actual_crop are required'}), 400
    # resolve user id
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id FROM users WHERE mobile = ?', (mobile,))
    row = cur.fetchone()
    if not row:
        conn.close()
        return jsonify({'error': 'user not found'}), 404
    user_id = row['id']
    try:
        # # store feedback as a special row in predictions with predicted_crop set to actual and a flag in address? Use a separate table
        # cur.execute('''CREATE TABLE IF NOT EXISTS feedback (
        #     id INTEGER PRIMARY KEY AUTOINCREMENT,
        #     user_id INTEGER,
        #     prediction_id INTEGER,
        #     N REAL, P REAL, K REAL, temperature REAL, humidity REAL, ph REAL, rainfall REAL,
        #     actual_crop TEXT,
        #     created_at TEXT NOT NULL
        # )''')
        # if prediction_id supplied, try to copy features from predictions
        N = P = K = temperature = humidity = ph = rainfall = None
        if prediction_id:
            cur.execute('SELECT N,P,K,temperature,humidity,ph,rainfall,input_source FROM predictions WHERE id = ?', (prediction_id,))
            p = cur.fetchone()
            if p:
                # only accept feedback copied from IoT-origin predictions
                input_source = p['input_source'] if 'input_source' in p.keys() else None
                if input_source != 'iot':
                    conn.close()
                    return jsonify({'error': 'feedback allowed only for predictions originating from IoT sensors'}), 403
                N = p['N']; P = p['P']; K = p['K']; temperature = p['temperature']; humidity = p['humidity']; ph = p['ph']; rainfall = p['rainfall']
        # insert feedback row
        cur.execute('''INSERT INTO feedback (user_id,prediction_id,N,P,K,temperature,humidity,ph,rainfall,actual_crop,created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?)''', (user_id, prediction_id, N, P, K, temperature, humidity, ph, rainfall, actual_crop, datetime.utcnow().isoformat()))
        conn.commit()
        conn.close()
        return jsonify({'status': 'feedback recorded'})
    except Exception as e:
        conn.close()
        return jsonify({'error': str(e)}), 500





# --- Background retrain job system (simple in-memory) ---
_jobs = {}
_jobs_lock = threading.Lock()

def _run_retrain_job(job_id):
    global model, scaler, le  #Declare globals before usage

    with _jobs_lock:
        _jobs[job_id] = {
            'status': 'running',
            'started_at': datetime.utcnow().isoformat(),
            'progress': 0,
            'error': None,
            'finished_at': None
        }

    try:
        import pandas as pd
        from sklearn.preprocessing import MinMaxScaler, LabelEncoder
        from sklearn.ensemble import RandomForestClassifier

        # Load dataset
        csvpath = os.path.join(os.path.dirname(__file__), 'data', 'Crop_recommendation.csv')
        if not os.path.exists(csvpath):
            raise RuntimeError('Original dataset not found')
        df = pd.read_csv(csvpath)
        if 'Crop' not in df.columns:
            raise RuntimeError('Dataset missing Crop column')

        # Ensure feedback table exists and load feedback
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                prediction_id INTEGER,
                N REAL, P REAL, K REAL, temperature REAL, humidity REAL, ph REAL, rainfall REAL,
                actual_crop TEXT,
                created_at TEXT NOT NULL
            )
        ''')
        cur.execute('SELECT N,P,K,temperature,humidity,ph,rainfall,actual_crop FROM feedback')
        rows = cur.fetchall()
        conn.close()

        # Prepare feedback DataFrame
        fb = pd.DataFrame(rows, columns=['N','P','K','temperature','humidity','ph','rainfall','actual_crop']) if rows else pd.DataFrame()
        all_crops = pd.concat([df['Crop'].astype(str), fb['actual_crop'].astype(str)], ignore_index=True)

        # Encode crop labels
        le_local = LabelEncoder()
        le_local.fit(all_crops)
        df['crop_label'] = le_local.transform(df['Crop'].astype(str))
        if not fb.empty:
            fb['crop_label'] = le_local.transform(fb['actual_crop'].astype(str))
            df_fb = fb[['N','P','K','temperature','humidity','ph','rainfall','crop_label']]
            df = pd.concat([df, df_fb], ignore_index=True)

        # Prepare training data
        feature_cols = ['N','P','K','temperature','humidity','ph','rainfall']
        X_full = df[feature_cols].astype(float).fillna(0)
        y_full = df['crop_label']

        _jobs[job_id]['progress'] = 20

        # Train model and scaler
        ms = MinMaxScaler()
        X_scaled = ms.fit_transform(X_full)

        _jobs[job_id]['progress'] = 50

        rfc = RandomForestClassifier(n_estimators=200, random_state=42)
        rfc.fit(X_scaled, y_full)

        _jobs[job_id]['progress'] = 90

        # Save artifacts
        with open(MODEL_PATH, 'wb') as f: pickle.dump(rfc, f)
        with open(SCALER_PATH, 'wb') as f: pickle.dump(ms, f)
        le_path = os.path.join(os.path.dirname(__file__), 'model', 'label_encoder.pkl')
        with open(le_path, 'wb') as f: pickle.dump(le_local, f)

        # Validate saves
        for path in [MODEL_PATH, SCALER_PATH, le_path]:
            if not os.path.exists(path):
                raise RuntimeError(f"Missing saved file: {path}")

        # Update global references
        model = rfc
        scaler = ms
        le = le_local

        _jobs[job_id]['progress'] = 100
        _jobs[job_id]['status'] = 'finished'
        _jobs[job_id]['finished_at'] = datetime.utcnow().isoformat()

        print(f"[{job_id}] Retraining completed successfully.")

    except Exception as e:
        _jobs[job_id]['status'] = 'error'
        _jobs[job_id]['error'] = str(e)
        _jobs[job_id]['finished_at'] = datetime.utcnow().isoformat()
        print(f"[{job_id}] Retraining failed: {e}")



def cleanup_old_jobs(max_age_minutes=60, max_jobs=100):
    now = datetime.utcnow()
    with _jobs_lock:
        # Remove jobs older than max_age_minutes
        expired = [job_id for job_id, job in _jobs.items()
                   if job['finished_at'] and
                   (now - datetime.fromisoformat(job['finished_at'])).total_seconds() > max_age_minutes * 60]
        for job_id in expired:
            del _jobs[job_id]

        # If still too many jobs, trim oldest
        if len(_jobs) > max_jobs:
            sorted_jobs = sorted(_jobs.items(), key=lambda x: x[1]['finished_at'] or x[1]['started_at'] or now)
            for job_id, _ in sorted_jobs[:len(_jobs) - max_jobs]:
                del _jobs[job_id]




@app.route('/api/admin/retrain', methods=['POST'])
@admin_required
def admin_retrain():
    """Retrain model using original dataset plus feedback when available.
    This is a simple retrain: fits MinMaxScaler on combined features and trains RandomForest.
    """
    cleanup_old_jobs()
    # Enqueue retrain as a background job and return job id
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {'status': 'queued', 'progress': 0, 'started_at': None, 'finished_at': None, 'error': None}
    t = threading.Thread(target=_run_retrain_job, args=(job_id,))
    t.daemon = True
    t.start()
    return jsonify({'job_id': job_id, 'status': 'queued'})

@app.route('/api/admin/download-label-encoder', methods=['GET'])
def download_label_encoder():
    token = request.headers.get('X-Admin-Token')
    if token != 'admintoken123':
        return jsonify({'error': 'Unauthorized'}), 401

    le_path = os.path.join(os.path.dirname(__file__), 'model', 'label_encoder.pkl')
    if not os.path.exists(le_path):
        return jsonify({'error': 'File not found'}), 404

    return send_file(le_path, as_attachment=True)
    
@app.route('/api/admin/retrain/status/<job_id>', methods=['GET'])
@admin_required
def retrain_status(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return jsonify({'error': 'job not found'}), 404
        return jsonify(job)


@app.route('/health')
def health():
    return 'OK', 200


if __name__ == '__main__':
    init_db()
    port =int(os.environ.get("PORT",5000))
    app.run(host='0.0.0.0',port=port,debug=True)























