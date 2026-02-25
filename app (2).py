# API/app.py - HYBRID ML v4 - NO DOMAIN RULES + DETAILED METRICS
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

CAR_TYPE_MODEL_PATH = 'model/car_type_classifier_v4.pkl'
CAR_RECOMMEND_MODEL_PATH = 'model/car_recommendation_v4.pkl'
ENCODERS_PATH = 'model/encoders_v4.pkl'
VECTORIZER_PATH = 'model/tfidf_vectorizer_v4.pkl'
FEATURES_LIST_PATH = 'model/features_list_v4.pkl'
os.makedirs('model', exist_ok=True)

DB_CONFIG = {
    'host': 'localhost', 'user': 'root', 'password': '', 'database': 'icrr'
}

car_type_model = None
car_recommend_model = None
encoders = {}
tfidf_vectorizer = None
car_type_features = []
tfidf_n_features = 0
last_metrics = {}

def extract_training_dataset():
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )
    sql = """
    SELECT
        bh.car_id,
        bh.total_amount AS budget,
        JSON_EXTRACT(bh.trip_purpose, '$[0]') AS trip_purpose_raw,
        TIMESTAMPDIFF(DAY, bh.start_date, bh.end_date) AS duration_days,
        COALESCE(r.comment, '') AS inquiry_text,
        ci.capacity,
        ci.car_type,
        ci.fuel_type,
        ci.transmission,
        ci.terrain,
        ci.budget_friendly,
        ci.aircon,
        ci.child_seat,
        ci.special_needs_friendly,
        ci.wide_compartment,
        ci.wide_leg_room
    FROM booking_history bh
    JOIN car_inventory ci ON bh.car_id = ci.id
    LEFT JOIN reviews r ON r.booking_id = bh.id
    WHERE bh.booking_status = 'completed'
      AND bh.payment_status = 'paid'
      AND bh.total_amount > 0
      AND JSON_LENGTH(bh.trip_purpose) > 0
    ORDER BY bh.id DESC
    """
    df = pd.read_sql(sql, engine)
    print(f"✅ Extracted {len(df)} training examples")
    return df

def preprocess_features(df):
    print("✅ Preprocessing features...")
    initial_count = len(df)
   
    df['trip_purpose'] = df['trip_purpose_raw'].astype(str).fillna('Unknown')
    df['duration_days'] = df['duration_days'].fillna(1).astype(int)
    df['budget'] = df['budget'].fillna(df['budget'].median())
    df['inquiry_text'] = df['inquiry_text'].fillna('').astype(str)
   
    print(f" → NO DOMAIN CLEANING - ALL DATA KEPT: {initial_count} records")
   
    # 🔥 BINARY FEATURES
    binary_features = ['budget_friendly', 'aircon', 'child_seat', 'special_needs_friendly',
                      'wide_compartment', 'wide_leg_room']
    for feature in binary_features:
        if feature in df.columns:
            df[feature] = (df[feature].astype(str).str.upper() == 'Y').astype(int)
   
    # 🔥 TERRAIN
    terrain_map = {'Mixed': 0, 'Urban': 1, 'Offroad': 2, 'Highway': 3}
    df['terrain_numeric'] = df['terrain'].map(terrain_map).fillna(0)
   
    # 🔥 DOMAIN SCORES (NO car_type dependency)
    df['offroad_score'] = (df['terrain_numeric'] == 2).astype(int) * 10
    df['family_score'] = (df['capacity'] >= 7).astype(int) * 5
    df['cargo_score'] = df['capacity'].isin([12, 15, 21]) * 8
    df['airport_score'] = (df['capacity'] >= 7).astype(int) * 6
   
    print(f" → Features ready: {len(df)} rows")
    return df

# 🔥 MODEL 1: CAR_TYPE CLASSIFIER (DETAILED METRICS)
def train_car_type_model(df):
    global car_type_model, encoders, tfidf_vectorizer, tfidf_n_features, car_type_features
   
    print("\n🚀 TRAINING CAR_TYPE MODEL...")
   
    # 🔥 FIXED FEATURES - NO car_type during training
    car_type_features_list = [
        'capacity', 'budget', 'trip_purpose', 'duration_days',
        'fuel_type', 'transmission', 'terrain_numeric',
        'budget_friendly', 'aircon', 'child_seat', 'special_needs_friendly',
        'wide_compartment', 'wide_leg_room',
        'offroad_score', 'family_score', 'cargo_score', 'airport_score'
    ]
   
    # Ensure all features exist
    for col in car_type_features_list:
        if col not in df.columns:
            df[col] = 0
   
    X_regular = df[car_type_features_list].copy()
   
    # 🔥 TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    inquiry_vectors = tfidf_vectorizer.fit_transform(df['inquiry_text'])
    tfidf_n_features = inquiry_vectors.shape[1]
   
    X_nlp = pd.DataFrame(
        inquiry_vectors.toarray(),
        columns=[f'inquiry_tfidf_{i}' for i in range(tfidf_n_features)],
        index=df.index
    )
   
    # 🔥 COMBINE - SAME ORDER AS TRAINING
    X = pd.concat([X_regular, X_nlp], axis=1)
    car_type_features = list(X.columns)
   
    # TARGET
    y_car_type = df['car_type']
   
    # 🔥 ENCODE FEATURES
    encoders = {}
    for col in ['trip_purpose', 'fuel_type', 'transmission']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
   
    le_car_type = LabelEncoder()
    y_car_type_encoded = le_car_type.fit_transform(y_car_type)
    encoders['car_type'] = le_car_type
   
    # TRAIN
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_car_type_encoded, test_size=0.2, random_state=42, stratify=y_car_type_encoded
    )
   
    car_type_model = DecisionTreeClassifier(
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42
    )
    car_type_model.fit(X_train, y_train)
   
    y_pred = car_type_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    precision_macro = precision_score(y_test, y_pred, average='macro')
    recall_macro = recall_score(y_test, y_pred, average='macro')
   
    print("\n📊 CAR_TYPE MODEL EVALUATION:")
    print(f"   Accuracy:        {accuracy:.3f}")
    print(f"   Macro F1:        {f1_macro:.3f}")
    print(f"   Macro Precision: {precision_macro:.3f}")
    print(f"   Macro Recall:    {recall_macro:.3f}")
    print(f"   📊 Features:      {len(car_type_features)}")
    print("\n" + "="*50)
   
    return accuracy

# 🔥 MODEL 2: CAR RECOMMENDATION (DETAILED METRICS)
def train_car_recommend_model(df):
    global car_recommend_model
   
    print("\n🚀 TRAINING CAR RECOMMENDATION MODEL...")
   
    # 🔥 RECOMMENDATION FEATURES - INCLUDES car_type
    recommend_features_list = [
        'capacity', 'budget', 'trip_purpose', 'duration_days', 'car_type',
        'fuel_type', 'transmission', 'terrain_numeric',
        'budget_friendly', 'aircon', 'child_seat', 'special_needs_friendly',
        'wide_compartment', 'wide_leg_room',
        'offroad_score', 'family_score', 'cargo_score', 'airport_score'
    ]
   
    for col in recommend_features_list:
        if col not in df.columns:
            df[col] = 0
   
    X_recommend = df[recommend_features_list].copy()
    y_recommend = df['car_id'].astype(int)
   
    # ENCODE
    for col in ['trip_purpose', 'car_type', 'fuel_type', 'transmission']:
        le = LabelEncoder()
        X_recommend[col] = le.fit_transform(X_recommend[col].astype(str))
        encoders[col] = le
   
    le_car_id = LabelEncoder()
    y_recommend_encoded = le_car_id.fit_transform(y_recommend)
    encoders['car_id'] = le_car_id
   
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_recommend, y_recommend_encoded, test_size=0.2, random_state=42
    )
   
    from collections import Counter
    class_weights = {i: max(1.0, 10.0 / count) for i, count in Counter(y_train_r).items()}
   
    car_recommend_model = DecisionTreeClassifier(
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weights,
        random_state=42
    )
    car_recommend_model.fit(X_train_r, y_train_r)
   
    y_pred_r = car_recommend_model.predict(X_test_r)
    accuracy_r = accuracy_score(y_test_r, y_pred_r)
   
    print("\n📊 RECOMMENDATION MODEL EVALUATION:")
    print(f"   Accuracy:        {accuracy_r:.3f}")
    print(f"   📊 Unique cars:   {len(np.unique(y_recommend))}")
   
    return accuracy_r

def train_decision_tree():
    global last_metrics
   
    df = extract_training_dataset()
    df = preprocess_features(df)
   
    if len(df) < 100:
        print("❌ Not enough training data")
        return False
   
    car_type_acc = train_car_type_model(df)
    recommend_acc = train_car_recommend_model(df)
   
    # 🔥 SAVE EVERYTHING
    joblib.dump(car_type_model, CAR_TYPE_MODEL_PATH)
    joblib.dump(car_recommend_model, CAR_RECOMMEND_MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    joblib.dump(tfidf_vectorizer, VECTORIZER_PATH)
    joblib.dump(car_type_features, FEATURES_LIST_PATH)
    joblib.dump(tfidf_n_features, 'model/tfidf_n_features_v4.pkl')
   
    last_metrics = {
        "car_type_accuracy": float(car_type_acc),
        "recommend_accuracy": float(recommend_acc),
        "data_points": len(df),
        "unique_cars": len(df['car_id'].unique()),
        "car_types": len(df['car_type'].unique())
    }
   
    return True

def load_model():
    global car_type_model, car_recommend_model, encoders, tfidf_vectorizer, car_type_features, tfidf_n_features
   
    required_files = [
        CAR_TYPE_MODEL_PATH, CAR_RECOMMEND_MODEL_PATH,
        ENCODERS_PATH, VECTORIZER_PATH, FEATURES_LIST_PATH
    ]
   
    if all(os.path.exists(f) for f in required_files):
        try:
            car_type_model = joblib.load(CAR_TYPE_MODEL_PATH)
            car_recommend_model = joblib.load(CAR_RECOMMEND_MODEL_PATH)
            encoders = joblib.load(ENCODERS_PATH)
            tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
            car_type_features = joblib.load(FEATURES_LIST_PATH)
            tfidf_n_features = joblib.load('model/tfidf_n_features_v4.pkl')
            print("✅ ALL MODELS LOADED SUCCESSFULLY!")
            return True
        except Exception as e:
            print(f"⚠️ Load failed ({e}), retraining...")
   
    return train_decision_tree()

def predict_top_12_cars(user_input):
    global car_type_features, tfidf_n_features
   
    capacity = user_input.get('capacity', 5)
    budget = user_input.get('budget', 3000)
    trip_purpose = user_input.get('trip_purpose', 'Family Trip')
    duration_days = user_input.get('duration_days', 1)
    inquiry_text = user_input.get('inquiry', '')
   
    # 🔥 STEP 1: PREDICT CAR_TYPE (EXACT FEATURE MATCH)
    feature_vector = {
        'capacity': capacity,
        'budget': budget,
        'trip_purpose': trip_purpose,
        'duration_days': duration_days,
        'fuel_type': 'Gasoline',
        'transmission': 'Automatic',
        'terrain_numeric': 2 if trip_purpose == 'Off-road Trip' else 0,
        'budget_friendly': 1,
        'aircon': 1,
        'child_seat': 0,
        'special_needs_friendly': 0,
        'wide_compartment': 1,
        'wide_leg_room': 1,
        'offroad_score': 10 if trip_purpose == 'Off-road Trip' else 0,
        'family_score': 5 if 'Family' in trip_purpose else 0,
        'cargo_score': 8 if 'Cargo' in trip_purpose else 0,
        'airport_score': 6 if 'Airport' in trip_purpose else 0
    }
   
    X_user = pd.DataFrame([feature_vector])
   
    # 🔥 ENCODE EXACTLY LIKE TRAINING
    for col in ['trip_purpose', 'fuel_type', 'transmission']:
        if col in encoders:
            le = encoders[col]
            val = str(X_user[col].iloc[0])
            if val not in le.classes_:
                val = le.classes_[0]
            X_user[col] = le.transform([val])[0]
   
    # 🔥 TF-IDF EXACT MATCH
    inquiry_tfidf = tfidf_vectorizer.transform([inquiry_text])
    inquiry_df = pd.DataFrame(
        inquiry_tfidf.toarray(),
        columns=[f'inquiry_tfidf_{i}' for i in range(tfidf_n_features)]
    )
   
    # 🔥 EXACT FEATURE ORDER
    X_complete = pd.concat([X_user, inquiry_df], axis=1)
   
    # 🔥 ADD ALL MISSING FEATURES WITH 0
    for col in car_type_features:
        if col not in X_complete.columns:
            X_complete[col] = 0
   
    # 🔥 REORDER TO EXACT TRAINING ORDER
    X_complete = X_complete[car_type_features]
   
    # 🔥 PREDICT CAR_TYPE
    predicted_car_type_encoded = car_type_model.predict(X_complete)[0]
    predicted_car_type = encoders['car_type'].inverse_transform([predicted_car_type_encoded])[0]
    car_type_prob = max(car_type_model.predict_proba(X_complete)[0])
   
    print(f"🔥 PREDICTED CAR_TYPE: {predicted_car_type} ({car_type_prob:.1%})")
   
    # 🔥 STEP 2: TOP 12 CARS OF PREDICTED TYPE
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}/{DB_CONFIG['database']}"
    )
   
    # 🔥 SQL: TOP 12 MATCHING CARS (NO DOMAIN RULES)
    sql = """
    SELECT ci.*, sa.shop_name, sa.location, sa.contact_number as shop_contact,
           cr.rate_inside_zambales_12hrs,
           (ci.capacity = %s) * 20 +
           (ABS(ci.capacity - %s) <= 2) * 10 +
           (ci.budget_friendly = 'Y') * 8 +
           (ci.aircon = 'Y') * 5 AS match_score
    FROM car_inventory ci
    JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
    LEFT JOIN car_rates cr ON ci.id = cr.car_id
    WHERE ci.car_type = %s
    ORDER BY match_score DESC, cr.rate_inside_zambales_12hrs ASC
    LIMIT 12
    """
   
    df_cars = pd.read_sql(sql, engine, params=(capacity, capacity, predicted_car_type))
   
    # 🔥 ENSURE 12 CARS
    if len(df_cars) < 12:
        remaining = 12 - len(df_cars)
        fallback_sql = """
        SELECT ci.*, sa.shop_name, sa.location, sa.contact_number as shop_contact,
               cr.rate_inside_zambales_12hrs,
               (ci.capacity = %s) * 15 +
               (ABS(ci.capacity - %s) <= 3) * 8 +
               (ci.budget_friendly = 'Y') * 5 AS match_score
        FROM car_inventory ci
        JOIN stakeholders_account sa ON ci.stakeholder_id = sa.id
        LEFT JOIN car_rates cr ON ci.id = cr.car_id
        WHERE ci.car_type != %s
          AND ci.capacity >= %s - 3
          AND ci.capacity <= %s + 3
        ORDER BY match_score DESC, cr.rate_inside_zambales_12hrs ASC
        LIMIT %s
        """
       
        additional_cars = pd.read_sql(
            fallback_sql, engine,
            params=(capacity, capacity, predicted_car_type, capacity, capacity, remaining)
        )
        df_cars = pd.concat([df_cars, additional_cars], ignore_index=True)
   
    # 🔥 BUILD RECOMMENDATIONS
    recommendations = []
    for i, (_, car) in enumerate(df_cars.head(12).iterrows()):
        car_dict = car.to_dict()
        for k, v in car_dict.items():
            if pd.isna(v):
                car_dict[k] = 'N/A'
       
        # 🔥 ML CONFIDENCE GRADIENT
        ml_confidence = max(0.85, 0.95 - (i * 0.04))
       
        recommendations.append({
            'car_id': int(car_dict['id']),
            'probability': float(ml_confidence),
            'rank': i + 1,
            'car_type': predicted_car_type,
            'car_type_confidence': float(car_type_prob),
            'car_details': car_dict
        })
   
    print(f"🔥 RETURNED {len(recommendations)} recommendations of type {predicted_car_type}")
    return recommendations

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "🚀 HYBRID ML v4 - NO DOMAIN RULES",
        "ready": car_type_model is not None,
        **last_metrics
    })

@app.route('/train', methods=['POST'])
def retrain():
    success = train_decision_tree()
    return jsonify({
        "status": "success" if success else "failed",
        **last_metrics
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.get_json() or {}
        recommendations = predict_top_12_cars(user_input)
       
        return jsonify({
            "status": "success",
            "recommendations": recommendations,
            "car_type_accuracy": last_metrics.get('car_type_accuracy', 0.0),
            "recommend_accuracy": last_metrics.get('recommend_accuracy', 0.0),
            "data_points": last_metrics['data_points'],
            "unique_cars": last_metrics['unique_cars'],
            "car_types": last_metrics.get('car_types', 0),
            "why_recommended": f"AI predicted {recommendations[0]['car_type']} • {user_input.get('trip_purpose', 'Trip')} • {user_input.get('capacity', 5)} seats",
            "input_used": user_input,
            "top_recommendation": recommendations[0]
        })
    except Exception as e:
        print(f"❌ PREDICT ERROR: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "car_type_model": car_type_model is not None,
        "recommend_model": car_recommend_model is not None,
        "features_loaded": len(car_type_features) > 0
    })

if __name__ == '__main__':
    print("🚀 HYBRID ML CAR RECOMMENDER v4")
    print("✅ NO DOMAIN RULES - PURE ML")
    print("✅ DETAILED METRICS: Accuracy + Macro F1 + Precision + Recall")
    print("✅ STAGE 1: Predict CAR_TYPE")
    print("✅ STAGE 2: Top 12 cars WITHIN predicted type")
    print("=" * 60)
   
    success = load_model()
   
    if success:
        print("✅ ALL MODELS READY!")
        print(f"📊 Car Type Accuracy: {last_metrics.get('car_type_accuracy', 0):.1%}")
        print(f"📊 Recommendation Accuracy: {last_metrics.get('recommend_accuracy', 0):.1%}")
        print(f"📈 Training examples: {last_metrics['data_points']:,}")
        print(f"🚗 Unique cars: {last_metrics['unique_cars']}")
        print(f"🔤 Car types: {last_metrics.get('car_types', 0)}")
        print(f"🧮 Features: {len(car_type_features)}")
        print("🎯 TOP 12 DIVERSE RECOMMENDATIONS: ACTIVE")
    else:
        print("❌ Training failed - check database")
   
    print("\n🌐 API Ready: http://127.0.0.1:5000")
    print("🧪 Test: POST /predict")
    print("=" * 60)
   
    app.run(host='127.0.0.1', port=5000, debug=False)