# Importing Libraries
import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
import requests

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score



# TRAIN MODEL

@st.cache_resource
def train_model():
    df = pd.read_csv("crop_data.csv")

    X = df.drop("yield_tons_per_hectare", axis=1)
    y = df["yield_tons_per_hectare"]

    categorical_cols = ["soil_type", "crop"]
    numeric_cols = ["rainfall_mm", "temperature_c",
                    "fertilizer_kg_per_ha", "area_hectares"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(), categorical_cols),
            ("num", "passthrough", numeric_cols)
        ]
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    joblib.dump(pipeline, "crop_model.pkl")

    return pipeline, mae, r2, preprocessor, model


pipeline, mae, r2, preprocessor, model = train_model()



# PDF GENERATOR

def create_pdf(prediction, fertilizer, crop, rainfall, temperature):
    file_path = "farm_report.pdf"

    doc = SimpleDocTemplate(file_path)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("Crop Yield Prediction Report", styles["Title"]))
    story.append(Spacer(1, 12))

    story.append(Paragraph(f"Crop: {crop}", styles["Normal"]))
    story.append(Paragraph(f"Predicted Yield: {prediction:.2f} tons/ha", styles["Normal"]))
    story.append(Paragraph(f"Fertilizer Used: {fertilizer} kg/ha", styles["Normal"]))
    story.append(Paragraph(f"Rainfall: {rainfall} mm", styles["Normal"]))
    story.append(Paragraph(f"Temperature: {temperature} °C", styles["Normal"]))

    doc.build(story)

    return file_path



# WEATHER FUNCTION

def get_weather(city):
    api_key = "ce86bdbfd1445eac2788f22acb7bf60b"

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"

    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
        return temp, humidity
    else:
        return None, None



# STREAMLIT UI

st.title("🌾 Crop Yield Prediction & Farm Insights")

st.write("""
Predict crop yield using Machine Learning and receive fertilizer guidance,
crop comparison, Ghana insights, and weather data.
""")



# MODEL PERFORMANCE

st.subheader("📊 Model Performance")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**R² Score:** {r2:.2f}")



# FEATURE IMPORTANCE

st.subheader("📈 Feature Importance")

feature_names = (
    preprocessor.named_transformers_["cat"]
    .get_feature_names_out(["soil_type", "crop"])
    .tolist()
    + ["rainfall_mm", "temperature_c",
       "fertilizer_kg_per_ha", "area_hectares"]
)

importances = model.feature_importances_

fig, ax = plt.subplots()
ax.barh(feature_names, importances)
ax.set_title("Feature Importance")
st.pyplot(fig)



# USER INPUT

st.subheader("🌱 Enter Farm Details")

rainfall = st.number_input("Rainfall (mm)", 50.0, 300.0, 120.0)
temperature = st.number_input("Temperature (°C)", 10.0, 45.0, 28.0)
fertilizer = st.number_input("Fertilizer (kg/ha)", 10.0, 200.0, 80.0)
area = st.number_input("Area (hectares)", 0.5, 20.0, 2.0)

soil = st.selectbox("Soil Type", ["Loamy", "Sandy", "Clay"])
crop = st.selectbox("Crop Type", ["Maize", "Rice", "Wheat"])



# PREDICTION BUTTON

if st.button("Predict Yield"):

    input_df = pd.DataFrame({
        "rainfall_mm": [rainfall],
        "temperature_c": [temperature],
        "fertilizer_kg_per_ha": [fertilizer],
        "area_hectares": [area],
        "soil_type": [soil],
        "crop": [crop]
    })

    prediction = pipeline.predict(input_df)[0]

    st.success(f"🌾 Predicted Yield: {prediction:.2f} tons per hectare")


    
    # Yield Comparison
    
    st.subheader("🌾 Yield Comparison Across Crops")

    comparison_data = []

    for crop_type in ["Maize", "Rice", "Wheat"]:
        temp_df = input_df.copy()
        temp_df["crop"] = crop_type

        pred = pipeline.predict(temp_df)[0]
        comparison_data.append(pred)

    fig2, ax2 = plt.subplots()
    ax2.bar(["Maize", "Rice", "Wheat"], comparison_data)
    ax2.set_ylabel("Yield (tons/ha)")
    ax2.set_title("Crop Yield Comparison")

    st.pyplot(fig2)


    
    # Fertilizer Recommendation
    
    st.subheader("🧪 Fertilizer Recommendation")

    if fertilizer < 50:
        st.warning("Fertilizer level is low. Consider increasing application.")
    elif fertilizer > 120:
        st.warning("Fertilizer level may be excessive.")
    else:
        st.success("Fertilizer level is optimal.")


    
    # Optimal Fertilizer
    
    st.subheader("🧪 Optimal Fertilizer Recommendation")

    fert_range = range(20, 151, 10)

    best_yield = 0
    best_fert = fertilizer

    for f in fert_range:
        temp_df = input_df.copy()
        temp_df["fertilizer_kg_per_ha"] = f

        pred = pipeline.predict(temp_df)[0]

        if pred > best_yield:
            best_yield = pred
            best_fert = f

    st.success(f"Recommended Fertilizer: {best_fert} kg/ha")
    st.write(f"Expected Yield: {best_yield:.2f} tons/ha")


    
    # Ghana Insights
    
    st.subheader("Ghana Agricultural Insights")

    if rainfall < 80:
        st.info("Low rainfall similar to Northern Ghana dry seasons.")
    elif rainfall > 200:
        st.info("High rainfall similar to Southern Ghana rainy season.")
    else:
        st.info("Moderate rainfall suitable for most crops.")

    if temperature > 32:
        st.info("High temperature may reduce maize yields.")


    
    # PDF Download
    
    pdf_file = create_pdf(prediction, fertilizer, crop, rainfall, temperature)

    with open(pdf_file, "rb") as f:
        st.download_button(
            "📄 Download Farm Report",
            f,
            file_name="farm_report.pdf"
        )



# WEATHER SECTION

st.subheader("🌦️ Live Weather Data")

city = st.text_input("Enter Location (e.g., Accra)")

if st.button("Get Weather"):
    temp, humidity = get_weather(city)

    if temp:
        st.write(f"Temperature: {temp} °C")
        st.write(f"Humidity: {humidity}%")
    else:
        st.error("Could not fetch weather data")
