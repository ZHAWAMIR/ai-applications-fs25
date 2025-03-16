# %%
import gradio as gr
import pandas as pd
import numpy as np
import pickle

# Load model from file
model_filename = "random_forest_regression_distance_to_city_center.pkl"
with open(model_filename, mode="rb") as f:
    model = pickle.load(f)

def predict_price(rooms, area, distance_to_city_center):
    # Modellvorhersage
    input_data = np.array([[rooms, area, distance_to_city_center]])
    prediction = model.predict(input_data)
    return f"Geschätzter Mietpreis: {round(prediction[0], 2)} CHF"

app = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Anzahl Zimmer"),
        gr.Number(label="Fläche in m²"),
        gr.Number(label="Entfernung zum Stadtzentrum (km)")
    ],
    outputs=gr.Textbox(label="Vorhersage"),
)

app.launch()