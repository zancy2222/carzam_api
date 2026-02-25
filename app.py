# api/app.py → PURE DOUBLE DTA + INQUIRY KEYWORD FILTERING (100% Decision Tree Logic)
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Model paths
TYPE_MODEL_PATH = 'model/dt_car_type.pkl'
NAME_MODEL_PATH = 'model/dt_car_name.pkl'
ENCODER_PATH = 'model/encoders.pkl'
FEATURES_PATH = 'model/features.pkl'

# DB Config
DB_CONFIG = {
    'user': 'u784630674_root',
    'password': 'Carzam123.',
    'host': 'srv2101.hstgr.io',  # <-- not localhost
    'database': 'u784630674_icrr'
}

# Global models
type_model = None
name_model = None
encoders = {}
features = []

def extract_training_data():
    engine = create_engine(f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
    sql = """
    SELECT
        ci.id AS car_id,
        ci.car_name,
        ci.car_type,
        ci.capacity,
        bh.total_amount AS budget,
        TIMESTAMPDIFF(DAY, bh.start_date, bh.end_date) + 1 AS duration_days,
        JSON_UNQUOTE(JSON_EXTRACT(bh.trip_purpose, '$[0]')) AS trip_purpose
    FROM booking_history bh
    JOIN car_inventory ci ON bh.car_id = ci.id
    WHERE bh.booking_status = 'completed'
      AND bh.payment_status = 'paid'
      AND JSON_LENGTH(bh.trip_purpose) > 0
    """
    df = pd.read_sql(sql, engine)
    print(f"Training on {len(df):,} real completed bookings")
    return df

def train_double_dt():
    global type_model, name_model, encoders, features
    df = extract_training_data()
    if df.empty:
        print("No training data!")
        return False

    df = df.dropna()
    df['duration_days'] = df['duration_days'].clip(1, 90)
    features = ['capacity', 'budget', 'duration_days', 'trip_purpose']
    X = df[features].copy()

    le_purpose = LabelEncoder()
    X['trip_purpose'] = le_purpose.fit_transform(X['trip_purpose'].astype(str))
    encoders['trip_purpose'] = le_purpose

    y_type = df['car_type']
    type_model = DecisionTreeClassifier(max_depth=10, random_state=42, class_weight='balanced')
    type_model.fit(X, y_type)

    y_name = df['car_name']
    name_model = DecisionTreeClassifier(max_depth=14, random_state=42, class_weight='balanced')
    name_model.fit(X, y_name)

    os.makedirs('model', exist_ok=True)
    joblib.dump(type_model, TYPE_MODEL_PATH)
    joblib.dump(name_model, NAME_MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(features, FEATURES_PATH)

    print("PURE DOUBLE DTA + INQUIRY SUPPORT TRAINED")
    return True

def load_models():
    global type_model, name_model, encoders, features
    paths = [TYPE_MODEL_PATH, NAME_MODEL_PATH, ENCODER_PATH, FEATURES_PATH]
    if all(os.path.exists(p) for p in paths):
        type_model = joblib.load(TYPE_MODEL_PATH)
        name_model = joblib.load(NAME_MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        features = joblib.load(FEATURES_PATH)
        print("PURE DOUBLE DTA + INQUIRY LOADED")
        return True
    else:
        print("Models not found. Training...")
        return train_double_dt()

def predict_with_double_dt(payload):
    if type_model is None or name_model is None:
        return None, None, 0.0, 0.0

    row = pd.DataFrame([{
        'capacity': payload.get('capacity', 5),
        'budget': payload.get('budget', 3000),
        'duration_days': payload.get('duration_days', 1),
        'trip_purpose': payload.get('trip_purpose', 'Daily Use')
    }])

    le = encoders['trip_purpose']
    purpose = row['trip_purpose'].iloc[0]
    if purpose not in le.classes_:
        purpose = le.classes_[0]
    row['trip_purpose'] = le.transform([purpose])[0]

    X_input = row[features]

    type_proba = type_model.predict_proba(X_input)[0]
    type_conf = max(type_proba)
    predicted_type = type_model.classes_[type_proba.argmax()]

    name_proba = name_model.predict_proba(X_input)[0]
    name_conf = max(name_proba)
    predicted_name = name_model.classes_[name_proba.argmax()]

    return predicted_type, predicted_name, type_conf, name_conf

# === INQUIRY KEYWORD MAPPING (AI reads every word) ===
def matches_inquiry(car_row, inquiry_text):
    if not inquiry_text:
        return True
    inquiry = inquiry_text.lower().strip()

    matches = 0
    total_keywords = 0

    # Keyword to field mapping
    keywords = {
        'diesel': str(car_row['fuel_type']).lower(),
        'gasoline': str(car_row['fuel_type']).lower(),
        'automatic': str(car_row['transmission']).lower(),
        'auto': str(car_row['transmission']).lower(),
        'manual': str(car_row['transmission']).lower(),
        'child': str(car_row['child_seat']).lower(),
        'baby': str(car_row['child_seat']).lower(),
        'seat': str(car_row['child_seat']).lower(),
        'red': str(car_row['color']).lower(),
        'black': str(car_row['color']).lower(),
        'white': str(car_row['color']).lower(),
        'silver': str(car_row['color']).lower(),
        'blue': str(car_row['color']).lower(),
        'ac': str(car_row['aircon']).lower(),
        'aircon': str(car_row['aircon']).lower(),
        'strong ac': str(car_row['aircon']).lower(),
        'wide': str(car_row['wide_leg_room']).lower(),
        'leg room': str(car_row['wide_leg_room']).lower(),
        'compartment': str(car_row['wide_compartment']).lower(),
        'big trunk': str(car_row['wide_compartment']).lower(),
        'trunk': str(car_row['wide_compartment']).lower(),
        'pwd': str(car_row['special_needs_friendly']).lower(),
        'senior': str(car_row['special_needs_friendly']).lower(),
        'handicap': str(car_row['special_needs_friendly']).lower(),
    }

    for keyword, field_value in keywords.items():
        if keyword in inquiry:
            total_keywords += 1
            if keyword in field_value or field_value == 'y' or 'yes' in field_value:
                matches += 1

    # Also allow car name match
    if any(word in str(car_row['car_name']).lower() for word in inquiry.split()):
        matches += 1
        total_keywords += 1

    return matches > 0 and (matches / max(total_keywords, 1)) >= 0.5  # at least 50% keyword match

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json() or {}
    user_capacity = payload.get('capacity')
    user_trip_purpose = payload.get('trip_purpose', 'Daily Use')
    inquiry = payload.get('inquiry', '').strip()

    predicted_type, predicted_name, type_conf, name_conf = predict_with_double_dt(payload)
    if predicted_type is None:
        return jsonify({"error": "Models not loaded"}), 500

    engine = create_engine(f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

    # First: Get cars matching Double DTA predictions + capacity
    sql = """
    SELECT
        ci.*,
        sa.shop_name,
        sa.location,
        cr.rate_inside_zambales_12hrs,
        COUNT(bh.id) AS booking_count
    FROM car_inventory ci
    JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
    LEFT JOIN car_rates cr ON ci.id = cr.car_id
    LEFT JOIN booking_history bh ON ci.id = bh.car_id
        AND bh.booking_status = 'completed'
        AND bh.payment_status = 'paid'
        AND JSON_UNQUOTE(JSON_EXTRACT(bh.trip_purpose, '$[0]')) = :trip_purpose
    WHERE ci.car_type = :predicted_type
      AND ci.car_name = :predicted_name
      AND ci.capacity = :capacity
      AND cr.rate_inside_zambales_12hrs IS NOT NULL
    GROUP BY ci.id
    LIMIT 20
    """

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), {
                "predicted_type": predicted_type,
                "predicted_name": predicted_name,
                "capacity": user_capacity,
                "trip_purpose": user_trip_purpose
            })
            df_cars = pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # Second: Apply inquiry keyword filtering (AI reads every word)
    filtered_cars = []
    for _, row in df_cars.iterrows():
        if matches_inquiry(row, inquiry):
            filtered_cars.append(row)

    recommendations = []
    for _, row in pd.DataFrame(filtered_cars).iterrows():
        recommendations.append({
            "car_id": int(row['id']),
            "car_name": row['car_name'],
            "car_type": row['car_type'],
            "capacity": int(row['capacity']),
            "rate": float(row['rate_inside_zambales_12hrs'] or 3000),
            "image": row['car_image'] or "PLACEHOLDER.png",
            "shop_name": row['shop_name'] or "Unknown Shop",
            "color": row['color'] or "Silver",
            "fuel_type": row['fuel_type'] or "Gasoline",
            "transmission": row['transmission'] or "Auto",
            "special_needs_friendly": row['special_needs_friendly'] or "N",
            "child_seat": row['child_seat'] or "N",
            "wide_leg_room": row['wide_leg_room'] or "N",
            "terrain": row['terrain'] or "Mixed",
            "budget_friendly": row['budget_friendly'] or "N",
            "aircon": row['aircon'] or "Y",
            "wide_compartment": row['wide_compartment'] or "N",
            "booking_count": int(row['booking_count'] or 0),
            "ml_type_confidence": round(type_conf * 100, 1),
            "ml_name_confidence": round(name_conf * 100, 1)
        })

    return jsonify({
        "ml_predicted_car_type": predicted_type,
        "ml_predicted_best_car": predicted_name,
        "type_confidence_percent": round(type_conf * 100, 1),
        "name_confidence_percent": round(name_conf * 100, 1),
        "user_trip_purpose": user_trip_purpose,
        "user_capacity": user_capacity,
        "user_inquiry": inquiry,
        "total_found": len(recommendations),
        "recommendations": recommendations[:12],
        "ranking_by": "pure_double_dta_with_inquiry_keyword_filtering",
        "model_type": "Pure Double Decision Tree + Natural Language Inquiry",
        "status": "success"
    })
@app.route('/dbtest')
def dbtest():
    from sqlalchemy import create_engine
    try:
        engine = create_engine(f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
        conn = engine.connect()
        conn.close()
        return {"status": "ok", "message": "DB connected"}
    except Exception as e:
        return {"status": "fail", "error": str(e)}
    
@app.route('/')
def home():
    return "<h1>CARZAM AI v6 — PURE DOUBLE DTA + INQUIRY KEYWORDS (100% ML)</h1>"

if __name__ == '__main__':
    load_models()
    app.run(port=5000, debug=False)