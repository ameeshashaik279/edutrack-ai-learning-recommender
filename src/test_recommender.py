
import pandas as pd
from recommender import train_model, recommend_courses

data = pd.read_csv("data/raw/aiml_content_engagement.csv")

data.columns = (
    data.columns.str.strip().str.lower().str.replace(" ", "_")
)

model = train_model(data)

student_id = data["student_id"].iloc[0]
recommendations = recommend_courses(model, student_id, n=5)

print("Test student:", student_id)
print("Recommendations:")
for item in recommendations:
    print(item)
