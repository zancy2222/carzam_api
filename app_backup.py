# api/app.py → FINAL v4.0 | WITH AI EXPLANATION + INQUIRY SUPPORT
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

MODEL_PATH = 'model/pure_dt_model.pkl'
ENCODER_PATH = 'model/pure_encoders.pkl'
FEATURES_PATH = 'model/pure_features.pkl'

DB_CONFIG = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'icrr'
}

model = None
encoders = {}
features = []

def extract_training_data():
    engine = create_engine(f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")
    sql = """
    SELECT 
        ci.capacity,
        bh.total_amount AS budget,
        TIMESTAMPDIFF(DAY, bh.start_date, bh.end_date) + 1 AS duration_days,
        JSON_UNQUOTE(JSON_EXTRACT(bh.trip_purpose, '$[0]')) AS trip_purpose,
        ci.car_type
    FROM booking_history bh
    JOIN car_inventory ci ON bh.car_id = ci.id
    WHERE bh.booking_status = 'completed' 
      AND bh.payment_status = 'paid'
      AND JSON_LENGTH(bh.trip_purpose) > 0
    """
    df = pd.read_sql(sql, engine)
    print(f"Training on {len(df)} real bookings")
    return df

def train_model():
    global model, encoders, features
    df = extract_training_data()
    df = df.dropna()
    df['duration_days'] = df['duration_days'].clip(1, 365)

    features = ['capacity', 'budget', 'duration_days', 'trip_purpose']
    X = df[features].copy()
    y = df['car_type']

    le_purpose = LabelEncoder()
    X['trip_purpose'] = le_purpose.fit_transform(X['trip_purpose'].astype(str))
    encoders['trip_purpose'] = le_purpose

    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    encoders['car_type'] = le_target

    model = DecisionTreeClassifier(max_depth=12, min_samples_leaf=3, class_weight='balanced', random_state=42)
    model.fit(X, y_encoded)

    # Metrics
    y_pred = model.predict(X)
    y_true = le_target.inverse_transform(y_encoded)
    y_pred_labels = le_target.inverse_transform(y_pred)

    print("\n" + "="*80)
    print("        CARZAM AI v4.0 — PURE ML + REAL BEHAVIOR + EXPLAINABLE")
    print("="*80)
    print(f"Training Data      : {len(df):,} real bookings")
    print(f"Accuracy           : {accuracy_score(y_encoded, y_pred):.4%}")
    print(f"Classes            : {len(le_target.classes_)} → {list(le_target.classes_)}")
    report = classification_report(y_true, y_pred_labels, output_dict=True, zero_division=0)
    for car_type in le_target.classes_:
        support = int(report[car_type]['support'])
        print(f"   {car_type:12} → Booked {support:,} times")
    print("="*80 + "\n")

    os.makedirs('model', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODER_PATH)
    joblib.dump(features, FEATURES_PATH)
    return True

def load_model():
    global model, encoders, features
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        encoders = joblib.load(ENCODER_PATH)
        features = joblib.load(FEATURES_PATH)
        print("CARZAM AI v4.0 Loaded — Pure ML + Explainable + Real Ranking")
        return True
    else:
        return train_model()

def generate_explanation(payload, predicted_type):
    capacity = payload['capacity']
    purpose = payload['trip_purpose']
    budget = payload['budget']
    duration = payload['duration_days']
    inquiry = payload.get('inquiry', '').strip().lower()

    reasons = []

    # Capacity-based
    if capacity <= 5:
        reasons.append(f"Compact size for {capacity} passengers")
    elif capacity <= 12:
        reasons.append(f"Perfect for groups of {capacity}")
    else:
        reasons.append(f"High-capacity transport for {capacity} people")

    # Purpose-based
    purpose_map = {
        'cargo delivery': 'ideal for hauling goods',
        'off-road': 'excellent ground clearance & durability',
        'family trip': 'spacious, comfortable, safe',
        'shuttle service': 'designed for multiple passengers',
        'airport': 'luggage space + comfort',
        'daily use': 'fuel-efficient & easy to drive',
        'travel': 'long-distance comfort'
    }
    for key, phrase in purpose_map.items():
        if key in purpose.lower():
            reasons.append(phrase.title())

    # Budget
    if budget < 2500:
        reasons.append("Best value within tight budget")
    elif budget > 10000:
        reasons.append("Premium comfort fits your budget")

    # Inquiry keywords
    if any(word in inquiry for word in ['child', 'baby', 'kid']):
        reasons.append("Family-friendly with child seat support")
    if any(word in inquiry for word in ['offroad', 'rough', 'mountain']):
        reasons.append("Built for tough terrain")
    if 'ac' in inquiry or 'cool' in inquiry:
        reasons.append("Strong air conditioning")
    if 'luxury' in inquiry or 'vip' in inquiry:
        reasons.append("Premium experience")

    # Final
    if predicted_type in ['Pickup', 'SUV']:
        reasons.append("Rugged & reliable")
    elif predicted_type in ['Van', 'MPV']:
        reasons.append("Maximum space & comfort")
    elif predicted_type == 'Sedan':
        reasons.append("Smooth, efficient, city-friendly")

    return f"We recommended <strong>{predicted_type}</strong> because: " + " • ".join(reasons[:4])

def predict_car_type(payload):
    if not model:
        return "Sedan", 0.0

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
    row = row[features]

    pred_idx = model.predict(row)[0]
    proba = model.predict_proba(row)[0]
    confidence = max(proba)
    predicted_type = encoders['car_type'].inverse_transform([pred_idx])[0]

    return predicted_type, confidence

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json() or {}
    user_capacity = payload.get('capacity')
    user_trip_purpose = payload.get('trip_purpose')
    inquiry = payload.get('inquiry', '')

    predicted_type, confidence = predict_car_type(payload)
    explanation = generate_explanation(payload, predicted_type)

    engine = create_engine(f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}")

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
    WHERE ci.car_type = :car_type
      AND ci.capacity = :capacity
      AND cr.rate_inside_zambales_12hrs IS NOT NULL
    GROUP BY ci.id
    ORDER BY booking_count DESC, cr.rate_inside_zambales_12hrs ASC
    LIMIT 12
    """

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), {
                "car_type": predicted_type,
                "capacity": user_capacity,
                "trip_purpose": user_trip_purpose
            })
            df_cars = pd.DataFrame(result.fetchall(), columns=result.keys())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    recommendations = []
    for i, row in df_cars.iterrows():
        bookings = int(row['booking_count'] or 0)
        recommendations.append({
            "car_id": int(row['id']),
            "car_name": row['car_name'],
            "car_type": row['car_type'],
            "capacity": int(row['capacity']),
            "rate": float(row['rate_inside_zambales_12hrs'] or 0),
            "image": row['car_image'] or "PLACEHOLDER.png",
            "shop_name": row['shop_name'] or "Unknown Shop",
            "location": row['location'] or "Zambales",
            "ml_confidence": round(confidence * 100, 1),
            "booking_count": bookings,
            "rank": i + 1
        })

    return jsonify({
        "ml_predicted_car_type": predicted_type,
        "ml_confidence_percent": round(confidence * 100, 1),
        "explanation": explanation,
        "user_inquiry": inquiry,
        "user_trip_purpose": user_trip_purpose,
        "user_capacity": user_capacity,
        "total_found": len(recommendations),
        "recommendations": recommendations,
        "ranking_by": "real_bookings_for_this_purpose",
        "pure_ml": True,
        "status": "success"
    })

@app.route('/')
def home():
    return "<h1>CARZAM AI v4.0 — Pure ML • Explainable • Real Behavior Ranked</h1>"

if __name__ == '__main__':
    load_model()
    app.run(port=5000, debug=False)