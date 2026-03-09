from flask import Flask, render_template, send_file
import pandas as pd
import joblib

app = Flask(__name__)

SHEET_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSAyFvoVNHjPq42pwuTJEk6onIuIJFvmAR_VMa_qbxqtdAer6AHiiB-fbN8pNl_5nSkRqnpLlROr4gW/pub?output=csv"

model = joblib.load("aqi_xgboost_model.pkl")


def load_data():

    df = pd.read_csv(SHEET_URL)

    return df


def predict_aqi(df):

    features = df[['PM2.5','PM10','NO2','SO2','CO','O3','Temperature','Humidity','Wind Speed']]

    df["Predicted_AQI"] = model.predict(features)

    return df


def aqi_category(aqi):

    if aqi <= 50:
        return "Good","green"
    elif aqi <= 100:
        return "Moderate","yellow"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups","orange"
    elif aqi <= 200:
        return "Unhealthy","red"
    elif aqi <= 300:
        return "Very Unhealthy","purple"
    else:
        return "Hazardous","maroon"


@app.route("/")
def home():

    df = load_data()

    df = predict_aqi(df)

    latest = df.iloc[-1]

    last20 = df.tail(20)

    aqi = round(latest["Predicted_AQI"],2)

    category,color = aqi_category(aqi)

    return render_template(
        "index.html",
        aqi=aqi,
        category=category,
        color=color,
        timestamps=list(last20["Timestamp"]),
        values=list(last20["Predicted_AQI"]),
        records=last20.to_dict(orient="records")
    )


@app.route("/download")
def download():

    df = load_data()

    df = predict_aqi(df)

    file_name="campus_aqi.xlsx"

    df.to_excel(file_name,index=False)

    return send_file(file_name,as_attachment=True)


if __name__ == "__main__":

    app.run(debug=True)