from tensorflow.keras.models import load_model

MODEL_PATH = "model.keras"  # Новый формат .keras
model = load_model(MODEL_PATH, compile=False)