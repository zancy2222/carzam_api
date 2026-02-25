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
    'database': 'icrr'
}
model = None
encoders = {}
last_metrics = {}
VALID_CAPACITIES = ['Small', 'Medium', 'Medium Group', 'Large Group', 'Small Group']
VALID_DURATIONS = ['12 Hours', '24 Hours', 'Weekly', 'Monthly', 'Yearly']
VALID_CAR_TYPES = ['Sedan','SUV','MPV','Van','Mini Truck']
VALID_FUEL_TYPES = ['Gasoline','Diesel','Electric']

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
    if not engine: return pd.DataFrame()
    sql = """
    SELECT
        ci.car_type,
        CASE
            WHEN ci.capacity <= 5 THEN 'Small'
            WHEN ci.capacity <= 8 THEN 'Medium'
            WHEN ci.capacity <= 15 THEN 'Large Group'
            ELSE 'Small Group'
        END AS capacity,
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
    """
    try:
        df = pd.read_sql(sql, engine)
        print(f"[TRAIN] Loaded {len(df)} rows.")
        return df
    except Exception as e:
        print(f"[TRAIN] SQL error: {e}")
        return pd.DataFrame()

# --------------------------------------------------------------------- #
# SAFE BINNING
# --------------------------------------------------------------------- #
def safe_qcut(s, q=5):
    s = pd.to_numeric(s, errors='coerce').dropna()
    if len(s) == 0 or s.nunique() <= 1:
        # return same-index series so caller can use .iloc[0]
        return pd.Series(['Level 1'] * len(s), index=s.index)
    bins = min(q, s.nunique())
    labels = [f'Level {i+1}' for i in range(bins)]
    try:
        return pd.qcut(s, q=bins, duplicates='drop', labels=labels).astype(str)
    except:
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(str)

# --------------------------------------------------------------------- #
# TRAIN MODEL + CLEAN METRICS
# --------------------------------------------------------------------- #
def train_model():
    global model, encoders, last_metrics
    df = extract_training_data()
    if df.empty or len(df) < 10:
        print("[TRAIN] Not enough data.")
        return False

    # -----------------------------------------------------------------
    # 1. FILTER + COPY → avoid SettingWithCopyWarning
    # -----------------------------------------------------------------
    df = df[df['car_type'].isin(VALID_CAR_TYPES)].copy()

    # -----------------------------------------------------------------
    # 2. Feature engineering (same as before)
    # -----------------------------------------------------------------
    df['capacity'] = df['capacity'].astype(str).str.strip()
    df['fuel_type'] = df['fuel_type'].astype(str).str.strip()
    df['comment']   = df['comment'].astype(str)

    df['duration'] = pd.cut(
        df['hours'],
        bins=[0,12,24,168,720,float('inf')],
        labels=VALID_DURATIONS,
        include_lowest=True
    ).astype(str)

    df['budget_bin'] = safe_qcut(df['budget'])
    df['review_keywords'] = df['comment'].apply(
        lambda x: ' '.join(re.findall(r'\w+', x.lower())) if x else ''
    )

    # -----------------------------------------------------------------
    # 3. Features / target
    # -----------------------------------------------------------------
    X = df[['capacity','duration','budget_bin','fuel_type','review_keywords']].copy()
    y = df['car_type']

    # -----------------------------------------------------------------
    # 4. Encode – **use .loc** to silence the warning
    # -----------------------------------------------------------------
    le_dict = {}
    for col in X.columns:
        le = LabelEncoder()
        le.fit(X[col])
        X.loc[:, col] = le.transform(X[col])
        le_dict[col] = le

    le_car = LabelEncoder()
    le_car.fit(VALID_CAR_TYPES)
    y_enc = le_car.transform(y)
    le_dict['car_type'] = le_car

    # -----------------------------------------------------------------
    # 5. Train / test split
    # -----------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.3, random_state=42, stratify=y_enc
    )

    model = DecisionTreeClassifier(max_depth=8, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # -----------------------------------------------------------------
    # 6. METRICS – only classes that appear in the test set
    # -----------------------------------------------------------------
    test_labels = np.unique(y_test)
    present_classes = [le_car.inverse_transform([i])[0] for i in test_labels]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred,
        labels=test_labels,
        average=None,
        zero_division=0          # silence UndefinedMetricWarning
    )

    acc = accuracy_score(y_test, y_pred)
    macro_f1   = precision_recall_fscore_support(y_test, y_pred, average='macro',   zero_division=0)[2]
    weighted_f1 = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)[2]

    # import confusion / report here because y_test/y_pred are available
    from sklearn.metrics import confusion_matrix, classification_report

    # -----------------------------------------------------------------
    # 7. PRETTY CONSOLE OUTPUT
    # -----------------------------------------------------------------
    print("\n" + "="*60)
    print(" " * 15 + "MODEL EVALUATION METRICS")
    print("="*60)
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>8} {'F1-Score':>10} {'Support':>8}")
    print("-" * 60)
    for idx, cls in enumerate(present_classes):
        print(f"{cls:<12} {precision[idx]:>10.3f} {recall[idx]:>8.3f} {f1[idx]:>10.3f} {support[idx]:>8}")
    print("-" * 60)
    print(f"{'Accuracy':<12} {acc:>10.3f} {'':>8} {'':>10} {len(y_test):>8}")
    print(f"{'Macro F1':<12} {macro_f1:>10.3f}")
    print(f"{'Weighted F1':<12} {weighted_f1:>10.3f}")
    print(f"{'Data Points':<12} {len(df):>10}")
    print("="*60 + "\n")

    # Add confusion matrix and classification report here
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # -----------------------------------------------------------------
    # 8. Save model & encoders
    # -----------------------------------------------------------------
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le_dict, ENCODERS_PATH)
    encoders.update(le_dict)

    last_metrics.update({
        "accuracy": round(acc, 3),
        "macro_f1": round(macro_f1, 3),
        "weighted_f1": round(weighted_f1, 3),
        "data_points": len(df)
    })

    return True

# --------------------------------------------------------------------- #
# LOAD MODEL
# --------------------------------------------------------------------- #
def load_model():
    global model, encoders
    if os.path.exists(MODEL_PATH) and os.path.exists(ENCODERS_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            print("[MODEL] Loaded.")
            return True
        except Exception as e:
            print(f"[MODEL] Load error: {e}")
    print("[MODEL] Training on startup.")
    return train_model()

# --------------------------------------------------------------------- #
# KEYWORD → CAR-INVENTORY MATCHING (EXPANDED)
# --------------------------------------------------------------------- #
def build_inventory_matches(car_row, words):
    reasons = []
    if car_row.get('color'):
        color_norm = car_row['color'].strip().lower()
        if any(w in color_norm for w in words):
            reasons.append(f"Color: <strong>{car_row['color']}</strong>")
    if car_row.get('transmission'):
        trans_norm = car_row['transmission'].lower()
        if any(w in trans_norm for w in words):
            reasons.append(f"Transmission: <strong>{car_row['transmission']}</strong>")
    if car_row.get('fuel_type'):
        fuel_norm = car_row['fuel_type'].lower()
        if any(w in fuel_norm for w in words):
            reasons.append(f"Fuel: <strong>{car_row['fuel_type']}</strong>")
    bool_map = {
        'wide_compartment' : 'Wide Compartment',
        'wide_leg_room' : 'Wide Leg Room',
        'child_seat' : 'Child Seat',
        'aircon' : 'Air Conditioned',
        'budget_friendly' : 'Budget Friendly',
        'special_needs_friendly': 'PWD & Senior Friendly',
    }
    for field, label in bool_map.items():
        if car_row.get(field) == 'Y':
            search_text = f"{field.replace('_', ' ')} {label}".lower()
            if any(w in search_text for w in words):
                reasons.append(label)
    if car_row.get('terrain'):
        terrain_norm = car_row['terrain'].lower()
        if any(w in terrain_norm for w in words):
            reasons.append(f"Terrain: <strong>{car_row['terrain']}</strong>")
    if car_row.get('with_driver') and 'driver' in words:
        reasons.append("With <strong>Driver</strong>")
    if car_row.get('capacity'):
        cap = car_row['capacity']
        cap_str = str(cap)
        if cap_str in ''.join(words):
            reasons.append(f"Capacity: <strong>{cap} seats</strong>")
        if cap <= 5 and any(w in {'small', 'compact'} for w in words):
            reasons.append("Small Capacity")
        elif cap <= 8 and any(w in {'medium', 'mid'} for w in words):
            reasons.append("Medium Capacity")
        elif cap <= 15 and any(w in {'large', 'group'} for w in words):
            reasons.append("Large Group Capacity")
    if car_row.get('car_type'):
        ctype_norm = car_row['car_type'].lower()
        if any(w in ctype_norm for w in words):
            reasons.append(f"Type: <strong>{car_row['car_type']}</strong>")
    return reasons

# --------------------------------------------------------------------- #
# FETCH CARS + WHY + PRIORITIZE MATCHES
# --------------------------------------------------------------------- #
def fetch_cars_and_reasons(recommended_type, user_inputs):
    engine = connect_db()
    if not engine: return [], {}
    sql = """
    SELECT
        ci.*,
        sa.shop_name, sa.location,
        cr.rate_inside_zambales_12hrs,
        cr.rate_inside_zambales_24hrs,
        cr.rate_outside_zambales_12hrs,
        cr.rate_outside_zambales_24hrs,
        cr.rate_baguio_12hrs,
        cr.rate_baguio_24hrs,
        cr.with_driver,
        cr.driver_price_12hrs,
        cr.driver_price_24hrs,
        (SELECT COUNT(*) FROM booking_history bh
         WHERE bh.car_id = ci.id
           AND bh.booking_status IN ('confirmed','completed')) AS booking_count
    FROM car_inventory ci
    JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
    LEFT JOIN car_rates cr ON ci.id = cr.car_id
    JOIN car_availability ca ON ci.id = ca.car_id
    WHERE ca.status = 'available'
      AND ci.car_type = %s
    ORDER BY booking_count DESC
    LIMIT 12
    """
    df_cars = pd.read_sql(sql, engine, params=(recommended_type,))
    cars = df_cars.to_dict(orient='records')
    inquiry = user_inputs.get('inquiry', '').lower()
    fuel_req = user_inputs.get('fuel_req')
    words = set(re.findall(r'\w+', inquiry)) if inquiry else set()
    why = {}
    matched_cars = []
    non_matched_cars = []

    if words and cars:
        ids = tuple(c['id'] for c in cars) if cars else (0,)
        rev_sql = """
        SELECT r.car_id, r.comment
        FROM reviews r
        JOIN bookings b ON r.booking_id = b.id
        WHERE b.car_id IN %s
        """
        try:
            rev_df = pd.read_sql(rev_sql, engine, params=(ids,))
            for car in cars:
                cid = car['id']
                matches = rev_df[rev_df['car_id'] == cid]
                rev_reasons = []
                for _, row in matches.iterrows():
                    if words & set(re.findall(r'\w+', row['comment'].lower())):
                        rev_reasons.append(row['comment'])
                if rev_reasons:
                    why.setdefault(cid, []).extend(rev_reasons[:2])
        except Exception as e:
            print(f"[REVIEWS] Error: {e}")

    for car in cars:
        cid = car['id']
        match_reasons = why.get(cid, [])[:]
        if fuel_req and str(car.get('fuel_type','')).strip() == fuel_req:
            match_reasons.append(f"Fuel: <strong>{fuel_req}</strong>")
        inv_reasons = build_inventory_matches(car, words)
        match_reasons.extend(inv_reasons)
        if match_reasons:
            why[cid] = " • ".join(match_reasons)
            matched_cars.append(car)
        else:
            non_matched_cars.append(car)
    final_cars = matched_cars + non_matched_cars
    return final_cars, why

# --------------------------------------------------------------------- #
# /predict
# --------------------------------------------------------------------- #
@app.route('/predict', methods=['POST','OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    if not model:
        return jsonify({"error":"Model not trained."}), 503
    data = request.get_json()
    if not data:
        return jsonify({"error":"No JSON"}), 400
    try:
        capacity = str(data.get('capacity','')).strip() or 'Medium'
        duration = str(data.get('duration','')).strip() or '12 Hours'
        budget = float(data.get('budget',0))
        car_type_user = str(data.get('car_type','')).strip()
        inquiry = str(data.get('inquiry','')).strip().lower()

        if capacity not in VALID_CAPACITIES: capacity = 'Medium'
        if duration not in VALID_DURATIONS: duration = '12 Hours'
        if car_type_user and car_type_user not in VALID_CAR_TYPES: car_type_user = None

        fuel_req = None
        if re.search(r'\bdiesel\b', inquiry): fuel_req = 'Diesel'
        elif re.search(r'\b(gasoline|gas)\b', inquiry): fuel_req = 'Gasoline'

        budget_bin = safe_qcut(pd.Series([budget])).iloc[0]
        source = "user_selected" if car_type_user else "ml_prediction"
        recommended_type = car_type_user

        if source == "ml_prediction":
            inp = pd.DataFrame([{
                'capacity': capacity,
                'duration': duration,
                'budget_bin': budget_bin,
                'fuel_type': fuel_req or 'Gasoline',
                'review_keywords': inquiry
            }])
            for col in inp.columns:
                le = encoders[col]
                val = inp.loc[0,col]
                # fallback to a known class if unseen
                if val not in le.classes_:
                    val = le.classes_[0]
                inp[col] = le.transform([val])[0]
            recommended_type = encoders['car_type'].inverse_transform(model.predict(inp))[0]

        user_inputs = {'inquiry': inquiry, 'fuel_req': fuel_req}
        cars, why_dict = fetch_cars_and_reasons(recommended_type, user_inputs)

        why_text = {}
        for c in cars:
            parts = [
                f"Car Type: <strong>{recommended_type}</strong>",
                f"Budget: <strong>₱{budget:,.0f}</strong>",
                f"Duration: <strong>{duration}</strong>",
                f"Capacity: <strong>{capacity}</strong>"
            ]
            if c['id'] in why_dict:
                parts.append(why_dict[c['id']])
            why_text[c['id']] = " • ".join(parts)

        return jsonify({
            "recommended_type": recommended_type,
            "source": source,
            "cars": cars,
            "why_recommended": why_text,
            "model_accuracy": last_metrics.get("accuracy",0.0),
            "macro_f1": last_metrics.get("macro_f1",0.0),
            "weighted_f1": last_metrics.get("weighted_f1",0.0),
            "data_points_used": last_metrics.get("data_points",0)
        })
    except Exception as e:
        print(f"[PREDICT] Error: {e}")
        return jsonify({"error":"Prediction failed"}), 500

# --------------------------------------------------------------------- #
# HOME & TRAIN
# --------------------------------------------------------------------- #
@app.route('/')
def home():
    return "AI LIVE – Car Recommender API"

@app.route('/train', methods=['POST'])
def train():
    success = train_model()
    if success:
        return jsonify({
            "status":"success",
            "message":f"Model trained on {last_metrics['data_points']} bookings.",
            "metrics": last_metrics
        })
    return jsonify({"error":"Training failed"}), 400

# --------------------------------------------------------------------- #
# START
# --------------------------------------------------------------------- #
if __name__ == '__main__':
    print("[STARTUP] Initializing...")
    load_model()
    print("[SERVER] Running at http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000, debug=False)
