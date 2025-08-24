import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction import FeatureHasher

st.set_page_config(page_title="Used Car Price Predictor- 2025 using Machine Learning", page_icon="🚗", layout="centered")
st.title("🚗 Used Car Price Predictor- 2025 using Machine Learning")

@st.cache_resource
def load_artifacts():
    model = joblib.load("model_carprice.pkl")
    artifacts = joblib.load("artifacts_carprice.pkl")
    return model, artifacts

model, artifacts = load_artifacts()
scaler_age_own = artifacts["scaler_age_own"]
scaler_km = artifacts["scaler_km"]
train_columns = artifacts["train_columns"]
HASH_VAR = artifacts["hasher_variant_n"]
HASH_MOD = artifacts["hasher_model_n"]

def preprocess_for_inference(df):
    df = df.copy()

    # Fuel Type → one-hot
    if "Fuel Type" in df.columns:
        df = pd.get_dummies(df, columns=["Fuel Type"], prefix="Fuel", dtype=int)

    # Ownership mapping
    ownership_map = {
        "1st owner": 1, "2nd owner": 2, "3rd owner": 3, "4th owner": 4,
        "5th owner": 5, "6th owner": 6, "7th owner": 7, "8th owner": 8,
        "9th owner": 9, "10th owner": 10,
    }
    if "Ownership" in df.columns:
        df["Ownership"] = df["Ownership"].map(ownership_map).fillna(0)

    # Transmission Type → binary
    if "Transmission Type" in df.columns:
        df["Is_Manual"] = (df["Transmission Type"] == "Manual").astype(int)
        df.drop("Transmission Type", axis=1, inplace=True)

    # Car Age + Ownership scaling
    needed_cols = {"Car Age", "Ownership"}
    if needed_cols.issubset(df.columns):
        df[["Car Age (scaled)", "Ownership (scaled)"]] = scaler_age_own.transform(
            df[["Car Age", "Ownership"]]
        )

    # KM Driven → log + scaling
    if "KM Driven" in df.columns:
        df["KM Driven (log)"] = np.log1p(df["KM Driven"])
        df[["KM Driven (scaled)"]] = scaler_km.transform(df[["KM Driven (log)"]])

    # Feature hashing: Car Variant
    if "Car Variant" in df.columns:
        hasher_variant = FeatureHasher(n_features=HASH_VAR, input_type="dict")
        hashed_variant = hasher_variant.transform(
            [{"Car Variant": val} for val in df["Car Variant"]]
        )
        hashed_variant_df = pd.DataFrame(
            hashed_variant.toarray(),
            columns=[f"CarVariant_hash_{i}" for i in range(hashed_variant.shape[1])],
        )
        df = pd.concat([df.reset_index(drop=True), hashed_variant_df], axis=1)

    # Feature hashing: Car Model
    if "Car Model" in df.columns:
        hasher_model = FeatureHasher(n_features=HASH_MOD, input_type="string")
        hashed_model = hasher_model.transform([[val] for val in df["Car Model"].astype(str)])
        hashed_model_df = pd.DataFrame(
            hashed_model.toarray(),
            columns=[f"CarModel_hash_{i}" for i in range(hashed_model.shape[1])],
        )
        df = pd.concat([df.reset_index(drop=True), hashed_model_df], axis=1)

    # Align to training feature space (missing columns → 0)
    df = df.reindex(columns=train_columns, fill_value=0)

    return df


list1=['Maruti Swift',
    'Maruti Swift Dzire',
    'Mahindra XUV500',
    'Maruti OMNI E',
    'Volkswagen Vento',
    'Maruti Omni',
    'Tata Hexa',
    'Hyundai Creta',
    'Tata PUNCH EV',
    'Hyundai i20',
    'Maruti Alto 800',
    'Maruti Wagon R 1.0',
    'Maruti A Star',
    'Maruti Ciaz',
    'Volkswagen Polo',
    'Tata Indigo ECS',
    'Hyundai Verna',
    'Maruti Ertiga',
    'Hyundai VENUE',
    'Honda City',
    'Honda BR-V',
    'Maruti Dzire',
    'Ford FREESTYLE',
    'BMW X1',
    'Hyundai Santa Fe',
    'Maruti Vitara Brezza',
    'Maruti Ritz',
    'Renault Kwid',
    'Hyundai Elite i20',
    'MG HECTOR PLUS',
    'Hyundai i10',
    'Honda Amaze',
    'Datsun Redi Go',
    'Renault Duster',
    'Maruti Grand Vitara',
    'Ford Figo Aspire',
    'Maruti Alto',
    'Nissan Terrano',
    'Mahindra Xylo',
    'Hyundai Eon',
    'Hyundai Grand i10',
    'Ford Figo',
    'Volkswagen Ameo',
    'Maruti Baleno',
    'Tata PUNCH',
    'Renault TRIBER',
    'Honda Brio',
    'Honda City ZX',
    'Toyota Etios',
    'Renault Lodgy',
    'Tata Harrier',
    'Skoda Superb',
    'Maruti XL6',
    'Renault Kiger',
    'Maruti IGNIS',
    'Toyota Fortuner',
    'Tata Tiago',
    'Tata NEXON',
    'Jeep Compass',
    'KIA SELTOS',
    'KIA SONET',
    'MG HECTOR',
    'Audi A4',
    'Tata Safari Storme',
    'Audi A8L',
    'Landrover Freelander 2',
    'Skoda Kodiaq',
    'Tata Manza',
    'Hyundai ALCAZAR',
    'Maruti S Cross',
    'Mercedes Benz E Class',
    'Chevrolet Beat',
    'Maruti Celerio',
    'Hyundai NEW SANTRO',
    'Ford Ecosport',
    'Maruti Eeco',
    'Mahindra Kuv100',
    'Maruti Alto K10',
    'Tata Zest',
    'Hyundai i20 Active',
    'Datsun Go',
    'Toyota Etios Liva',
    'Fiat Grand Punto',
    'Tata Bolt',
    'Ford Fiesta Classic',
    'Honda Jazz',
    'Maruti New Wagon-R',
    'Chevrolet Captiva',
    'Mahindra XUV700',
    'Maruti S PRESSO',
    'Skoda SLAVIA',
    'Hyundai New Elantra',
    'Hyundai GRAND I10 NIOS',
    'Mahindra MARAZZO',
    'Toyota Innova',
    'Hyundai Tucson',
    'Renault Captur',
    'BMW 3 Series',
    'Toyota Innova Crysta',
    'Hyundai NEW I20',
    'Tata TIGOR',
    'Chevrolet Spark',
    'Chevrolet Cruze',
    'Volkswagen Passat',
    'KIA CARENS',
    'Mahindra TUV300',
    'Honda WR-V',
    'Skoda KUSHAQ',
    'Toyota YARIS',
    'Mahindra Bolero',
    'Tata Nano',
    'Mahindra KUV 100 NXT',
    'Mahindra Thar',
    'Audi A6',
    'Mahindra BOLERO NEO',
    'Maruti BREZZA',
    'Ssangyong Rexton',
    'Audi A3',
    'Audi Q5',
    'Tata ALTROZ',
    'Skoda Rapid',
    'Mahindra Scorpio',
    'Mahindra NUVOSPORT',
    'Toyota URBAN CRUISER',
    'Maruti FRONX',
    'Skoda Fabia',
    'Hyundai Xcent',
    'Toyota Corolla Altis',
    'Skoda Laura',
    'Mercedes Benz GL Class',
    'Audi Q7',
    'Chevrolet Sail',
    'Ford Fiesta',
    'Volkswagen VIRTUS',
    'Datsun Go Plus',
    'Ford New Figo',
    'Maruti Wagon R Duo',
    'Honda Mobilio',
    'MG ASTOR',
    'Mercedes Benz C Class',
    'Mahindra XUV300',
    'Mahindra TUV 300 PLUS',
    'Mercedes Benz GLE',
    'Nissan Micra',
    'Mahindra Verito',
    'Tata TIAGO NRG',
    'Maruti Zen Estilo',
    'Renault Pulse',
    'Nissan Sunny',
    'Chevrolet Optra Magnum',
    'Tata Indica Vista',
    'Maruti SX4',
    'Toyota Glanza',
    'Volkswagen Jetta',
    'Honda Civic',
    'Mahindra E20 Plus',
    'Jaguar XF',
    'Skoda Octavia',
    'Audi Q3',
    'Volkswagen TAIGUN',
    'Ford Endeavour',
    'Mahindra XUV 3XO',
    'Tata Safari',
    'Tata Tigor Buzz',
    'Tata Curvv EV',
    'MG ZS EV',
    'Renault Koleos',
    'Tata TIGOR EV',
    'Hyundai Santro Xing',
    'Mahindra SCORPIO CLASSIC',
    'Maruti Wagon R Stingray',
    'Chevrolet Enjoy',
    'BMW X3',
    'Mahindra SCORPIO-N',
    'Mahindra ALTURAS G4',
    'Mitsubishi Pajero Sport',
    'KIA CARNIVAL',
    'Tata Indigo CS',
    'Nissan MAGNITE',
    'Tata Curvv',
    'Hyundai XCENT PRIME',
    'Ford Classic',
    'Maruti 800',
    'Renault Scala',
    'Hyundai AURA',
    'Mercedes Benz CLA Class',
    'Mahindra Thar Roxx',
    'CITROEN C3 AIRCROSS',
    'Fiat Punto EVO',
    'Skoda Yeti',
    'Tata NEXON EV',
    'Chevrolet Sail UVA',
    'Fiat Linea',
    'Hyundai Accent',
    'BMW 7 Series',
    'Tata MAGIC',
    'KIA SYROS',
    'Chevrolet Tavera',
    'Toyota URBAN CRUISER HYRYDER',
    'Maruti JIMNY',
    'Mercedes Benz B Class',
    'Mercedes Benz Ml Class',
    'Hyundai EXTER',
    'CITROEN C3',
    'Tata TIAGO EV',
    'Landrover DISCOVERY SPORT',
    'BMW 5 Series',
    'Maruti Celerio X',
    'Mini Cooper',
    'Toyota Urban Cruiser Taisor',
    'Nissan Micra Active',
    'ISUZU D-Max V Cross',
    'Ford Ikon',
    'Landrover Range Rover Evoque',
    'Fiat Avventura',
    'Nissan Kicks',
    'Hyundai VENUE N LINE',
    'Jeep WRANGLER',
    'Jeep MERIDIAN',
    'Volkswagen TIGUAN',
    'Mercedes Benz GLA Class',
    'MG GLOSTER',
    'BMW X5',
    'Honda CRV',
    'Toyota Camry',
    'Maruti Wagon R',
    'Toyota RUMION',
    'Volvo S60',
    'Honda Accord',
    'Nissan Teana',
    'Mini Cooper Countryman',
    'Skoda Karoq',
    'BMW M340i',
    'Volvo XC 40',
    'BMW 3 SERIES GRAN LIMOUSINE',
    'Toyota Lexus',
    'Honda ELEVATE',
    'Mercedes Benz GLC CLASS',
    'Hyundai NEW I20 N LINE',
    'CITROEN C5 AIRCROSS',
    'ISUZU MU-X',
    'Volvo XC 90',
    'Volvo XC 60',
    'Tata Indica EV2',
    'Landrover Range Rover Sport',
    'Landrover Range Rover',
    'Mahindra XUV400',
    'Tata Aria',
    'Porsche Cayenne',
    'Toyota Corolla',
    'Mahindra E VERITO',
    'CITROEN Basalt',
    'Mahindra Quanto',
    'Volvo V 40',
    'Jaguar F- PACE',
    'Nissan Evalia',
    'Fiat PUNTO PURE',
    'Toyota HILUX',
    'Renault Fluence',
    'Force Motors GURKHA',
    'Jaguar XJ L',
    'Mercedes Benz M Class',
    'Toyota INNOVA HYCROSS',
    'Maruti Invicto',
    'BMW 1 Series',
    'Landrover Discovery 4',
    'Mercedes Benz S Class',
    'BMW Z4',
    'Mercedes Benz CLS Class',
    'Mercedes Benz GLC COUPE',
    'Mini Clubman',
    'Hyundai Sonata',
    'CITROEN E C3',
    'Mercedes Benz A CLASS LIMOUSINE']

# ---------- UI ----------
col1, col2 = st.columns(2)

with col1:
    fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG", "Electric"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])
    ownership = st.selectbox(
        "Ownership",
        ["1st owner","2nd owner","3rd owner","4th owner","5th owner",
         "6th owner","7th owner","8th owner","9th owner","10th owner"]
    )

with col2:
    km_driven = st.number_input("KM Driven", min_value=0, value=50000, step=500)
    car_age = st.slider(
    "Select Car Age",
    min_value=0,       # int
    max_value=20,      # int
    value=5,           # int
    step=1             # int
    )

    



    # --- Car Model Dropdown ---
    car_model = st.selectbox(
        "Select Car Model",
        options=list1,  # list1 contains your 274 car models
        index=0,        # default first one
        help="Start typing to quickly search for a car model"
    )



if st.button("Predict Price"):
    input_df = pd.DataFrame({
        "Fuel Type": [fuel],
        "Ownership": [ownership],
        "Transmission Type": [transmission],
        "KM Driven": [km_driven],
        "Car Age": [car_age],
        "Car Model": [car_model],
    })

    X = preprocess_for_inference(input_df)
    pred = model.predict(X)[0]
    st.success(f"Estimated Price: {pred:,.2f} Lakhs")
