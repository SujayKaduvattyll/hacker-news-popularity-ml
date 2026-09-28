from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Hacker News Popularity Prediction API",
    description="Predicts Hacker News post popularity as Low, Medium, or High.",
    version="1.0"
)

model = joblib.load("hacker_news_popularity_model.joblib")


class PostData(BaseModel):
    post_type: str
    domain: str
    hour_posted: int
    day_of_week: int
    month: int
    is_weekend: int
    title_length: int
    word_count: int
    avg_word_length: float
    uppercase_count: int
    number_count: int
    special_char_count: int
    question_mark: int
    exclamation_mark: int
    contains_ai: int
    contains_python: int
    contains_open_source: int
    contains_security: int
    contains_database: int
    contains_linux: int
    contains_startup: int
    is_show_hn: int
    is_ask_hn: int
    is_job_post: int


@app.get("/")
def home():
    return {
        "message": "Hacker News Popularity Prediction API",
        "status": "running"
    }


@app.post("/predict")
def predict(data: PostData):
    input_data = pd.DataFrame([data.model_dump()])
    prediction = model.predict(input_data)[0]

    class_names = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    return {
        "prediction": int(prediction),
        "popularity_class": class_names[int(prediction)]
    }
