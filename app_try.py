# API/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import re
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, text

app = Flask(__name__)
CORS(app)

# --------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------- #
MODEL_PATH = 'model/car_recommender.pkl'
ENCODERS_PATH = 'model/encoders.pkl'
os.makedirs('model', exist_ok=True)

DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'icrr2'
}

model = None
encoders = {}
last_metrics = {}

VALID_DURATIONS = ['12 Hours', '24 Hours', 'Weekly', 'Monthly', 'Yearly']
VALID_CAR_TYPES = ['Sedan','SUV','MPV','Van','Mini Truck']
VALID_FUEL_TYPES = ['Gasoline','Diesel','Electric']

# Additional numeric features to enhance prediction
ADDITIONAL_FEATURES = [
    'comfort', 'space', 'compartment', 'compact',
    'road_compatibility', 'usage', 'other_features'
]

# Default values for additional features (1-5 scale)
DEFAULT_FEATURES = {f: 3 for f in ADDITIONAL_FEATURES}

# --------------------------------------------------------------------- #
# DB CONNECTION
# --------------------------------------------------------------------- #
def connect_db():
    try:
        engine = create_engine(
            f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@"
            f"{DB_CONFIG['host']}/{DB_CONFIG['database']}"
        )
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("[DB] Connected successfully.")
        return engine
    except Exception as e:
        print(f"[DB] Connection failed: {e}")
        return None

# --------------------------------------------------------------------- #
# EXTRACT TRAINING DATA
# --------------------------------------------------------------------- #
def extract_training_data():
    engine = connect_db()
    if not engine:
        return pd.DataFrame()

    sql = """
    SELECT
        ci.car_type,
        ci.capacity,
        ci.fuel_type,
        TIMESTAMPDIFF(HOUR, bh.start_date, bh.end_date) AS hours,
        bh.total_amount AS budget,
        COALESCE(r.comment, '') AS comment
    FROM booking_history bh
    JOIN car_inventory ci ON bh.car_id = ci.id
    LEFT JOIN reviews r ON r.booking_id = bh.id
    WHERE bh.booking_status IN ('confirmed','completed')
      AND bh.total_amount > 0
      AND ci.car_type IS NOT NULL
      AND ci.car_type IN ('Sedan','SUV','MPV','Van','Mini Truck')
      AND ci.capacity IS NOT NULL
    """
    try:
        df = pd.read_sql(sql, engine)
        print(f"[TRAIN] Loaded {len(df)} raw booking records.")
        return df
    except Exception as e:
        print(f"[TRAIN] SQL error: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------- #
# SAFE BINNING (for budget only)
# --------------------------------------------------------------------- #
def safe_qcut(s, q=5):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0 or s.nunique() <= 1:
        return pd.Series(['Medium'] * len(s), index=s.index)
    bins = min(q, s.nunique())
    labels = [f'Level {i+1}' for i in range(bins)]
    try:
        return pd.qcut(s, q=bins, duplicates='drop', labels=labels).astype(str)
    except:
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)
    
def load_model():
    global model, encoders
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            print("[MODEL] Loaded from disk.")
            return True
        except Exception as e:
            print(f"[MODEL] Load error: {e}")
    print("[MODEL] Training new model...")
    return train_model()


# --------------------------------------------------------------------- #
# TRAIN MODEL
# --------------------------------------------------------------------- #
def train_model():
    global model, encoders, last_metrics
    df = extract_training_data()
    if df.empty or len(df) < 15:
        print("[TRAIN] Not enough data for training.")
        return False

    df = df[df['car_type'].isin(VALID_CAR_TYPES)].copy()
    if df.empty:
        return False

    # Clean data
    df['capacity'] = pd.to_numeric(df['capacity'], errors='coerce')
    df = df.dropna(subset=['capacity'])
    df['capacity'] = df['capacity'].astype(int)
    df['fuel_type'] = df['fuel_type'].astype(str).str.strip()
    df['comment'] = df['comment'].astype(str).fillna('')

    # Duration binning
    df['duration'] = pd.cut(
        df['hours'],
        bins=[0, 12, 24, 168, 720, float('inf')],
        labels=VALID_DURATIONS,
        include_lowest=True
    ).astype(str)

    df['budget_bin'] = safe_qcut(df['budget'])
    df['review_keywords'] = df['comment'].apply(
        lambda x: ' '.join(re.findall(r'\w+', x.lower())) if x else ''
    )

    # Base features
    X = df[['capacity', 'duration', 'budget_bin', 'fuel_type', 'review_keywords']].copy()
    y = df['car_type']

    # ---- DERIVE ADDITIONAL FEATURES ---- #
    X['space'] = df['capacity'].apply(lambda x: min(max(x // 2, 1), 5))
    X['comfort'] = df['budget'].apply(lambda x: min(max(int(x / 1000), 1), 5))
    X['usage'] = df['duration'].map({
        '12 Hours': 1,
        '24 Hours': 3,
        'Weekly': 5,
        'Monthly': 5,
        'Yearly': 5
    }).fillna(3)
    X['compact'] = 5 - X['space']
    X['road_compatibility'] = df['capacity'].apply(lambda x: min(max(x // 3, 1), 5))
    X['other_features'] = 3  # default constant

    # Label encoding for categorical
    le_dict = {}
    for col in ['duration', 'budget_bin', 'fuel_type', 'review_keywords']:
        le = LabelEncoder()
        X[col] = X[col].astype(str)
        le.fit(X[col])
        X[col] = le.transform(X[col])
        le_dict[col] = le

    le_car = LabelEncoder()
    le_car.fit(VALID_CAR_TYPES)
    y_enc = le_car.transform(y)
    le_dict['car_type'] = le_car

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    # Train model
    model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = precision_recall_fscore_support(y_test, y_pred, average='macro', zero_division=0)[2]

    # Save
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_dict, ENCODERS_PATH)
    encoders.update(le_dict)

    last_metrics.update({
        "accuracy": round(acc, 3),
        "macro_f1": round(macro_f1, 3),
        "data_points": len(df),
        "class_distribution": pd.Series(y).value_counts().to_dict()
    })

    print(f"[TRAIN] Model trained successfully | Acc: {acc:.3f} | Data: {len(df)}")
    return True

# --------------------------------------------------------------------- #
# /predict with ML + inventory recommendation
# --------------------------------------------------------------------- #
@app.route('/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    if not model:
        return jsonify({"error": "Model not trained."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON"}), 400

    try:
        seater = data.get('seater')
        capacity = data.get('capacity')
        capacity = int(seater or capacity or 7)

        duration = str(data.get('duration', '12 Hours')).strip()
        budget = float(data.get('budget', 3000))
        car_type_user = str(data.get('car_type', '')).strip() or None
        inquiry = str(data.get('inquiry', '')).lower()
        fuel_req = str(data.get('fuel_type', 'Gasoline'))

        budget_bin = safe_qcut(pd.Series([budget]))[0]
        source = "ml_prediction"
        recommended_type = car_type_user

        # ---- DERIVE ADDITIONAL FEATURES ---- #
        inp_dict = {
            'capacity': capacity,
            'duration': duration,
            'budget_bin': budget_bin,
            'fuel_type': fuel_req,
            'review_keywords': inquiry or 'none',
            'space': min(max(capacity // 2, 1), 5),
            'comfort': min(max(int(budget / 1000), 1), 5),
            'usage': 1 if duration=='12 Hours' else 3 if duration=='24 Hours' else 5,
            'compact': 5 - min(max(capacity // 2, 1), 5),
            'road_compatibility': min(max(capacity // 3, 1), 5),
            'other_features': 3
        }

        inp = pd.DataFrame([inp_dict])

        # Encode categorical
        for col in ['duration', 'budget_bin', 'fuel_type', 'review_keywords']:
            le = encoders.get(col)
            if le:
                if inp.loc[0, col] not in le.classes_:
                    inp.loc[0, col] = le.classes_[0]
                inp[col] = le.transform(inp[col])

        # Predict car type
        predicted = model.predict(inp)[0]
        recommended_type = encoders['car_type'].inverse_transform([predicted])[0]

        # Fetch inventory of predicted type
        engine = connect_db()
        sql = """
        SELECT ci.*, sa.shop_name, sa.location,
               (SELECT COUNT(*) FROM booking_history bh
                WHERE bh.car_id = ci.id AND bh.booking_status IN ('confirmed','completed')
               ) AS booking_count
        FROM car_inventory ci
        JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
        JOIN car_availability ca ON ci.id = ca.car_id
        WHERE ca.status = 'available'
          AND ci.car_type = %s
        ORDER BY booking_count DESC, ci.capacity DESC
        LIMIT 12
        """
        df_cars = pd.read_sql(sql, engine, params=(recommended_type,))
        cars = df_cars.to_dict(orient='records')

        # Static "why recommended" (no rules)
        why_text = {}
        for c in cars:
            why_text[c['id']] = f"Recommended <strong>{recommended_type}</strong> • {c['capacity']} seats"

        return jsonify({
            "recommended_type": recommended_type,
            "source": source,
            "cars": cars,
            "why_recommended": why_text,
            "model_accuracy": last_metrics.get("accuracy", 0.0),
            "data_points_used": last_metrics.get("data_points", 0)
        })

    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        return jsonify({"error": "Prediction failed"}), 500

# --------------------------------------------------------------------- #
# ROUTES
# --------------------------------------------------------------------- #
@app.route('/')
def home():
    return "AI LIVE – Car Recommender API v2 (Seater-Aware)"

@app.route('/train', methods=['POST'])
def train():
    success = train_model()
    return jsonify({
        "status": "success" if success else "failed",
        "metrics": last_metrics
    })

# --------------------------------------------------------------------- #
# START
# --------------------------------------------------------------------- #
if __name__ == '__main__':
    print("[STARTUP] Initializing Car Recommender API (v2 – Seater Aware)...")
    load_model()
    print("[SERVER] Running at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
