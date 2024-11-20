from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import sqlite3
import keras
from tensorflow.keras.models import load_model
import threading

app = Flask(__name__)

# Загрузка обученной модели
MODEL_PATH = "model.keras"  # Используйте .keras или .h5 в зависимости от формата вашей модели
model = load_model(MODEL_PATH, compile=False)

# Классы объектов и их цены (замените на ваши товары и цены)
CLASSES = {
    'Snickers': 50,
    'Mars': 45,
    'KitKat': 40
}

# Глобальные переменные
camera = None
is_analyzing = False

# Инициализация базы данных
def init_db():
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            price INTEGER NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Добавление товара в базу данных
def add_product_to_db(name, price):
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    cursor.execute('INSERT INTO products (name, price) VALUES (?, ?)', (name, price))
    conn.commit()
    conn.close()

# Получение всех товаров из базы данных
def get_all_products():
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()
    cursor.execute('SELECT name, price FROM products')
    products = cursor.fetchall()
    conn.close()
    return products

# Функция для предобработки изображения
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))  # Приводим размер изображения к входному модели
    img = img / 255.0  # Нормализация
    img = np.expand_dims(img, axis=0)  # Добавляем измерение для батча
    return img

# Функция для анализа видео
def analyze_video():
    global is_analyzing, camera
    detected_items = set()  # Используем set, чтобы избежать дублирования
    camera = cv2.VideoCapture(0)  # Подключение к камере
    while is_analyzing:
        ret, frame = camera.read()
        if not ret:
            break

        # Предобработка кадра
        img = preprocess_frame(frame)

        # Предсказание модели
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        class_name = list(CLASSES.keys())[class_index]
        confidence = predictions[0][class_index]

        # Добавление товара в базу данных, если уверенность высокая
        if confidence > 0.8 and class_name not in detected_items:
            detected_items.add(class_name)
            add_product_to_db(class_name, CLASSES[class_name])

        # Отображение видеопотока (опционально)
        cv2.putText(frame, f"{class_name} ({confidence:.2f})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    camera.release()

# Главная страница
@app.route('/')
def index():
    products = get_all_products()
    return render_template('index.html', products=products)

# Запуск анализа
@app.route('/start', methods=['POST'])
def start_analysis():
    global is_analyzing
    if not is_analyzing:
        is_analyzing = True
        threading.Thread(target=analyze_video).start()
    return jsonify({'status': 'analyzing started'})

# Остановка анализа
@app.route('/stop', methods=['POST'])
def stop_analysis():
    global is_analyzing
    is_analyzing = False
    return jsonify({'status': 'analyzing stopped'})

# Видеопоток
@app.route('/video_feed')
def video_feed():
    return Response(analyze_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    init_db()
    app.run(debug=True)
