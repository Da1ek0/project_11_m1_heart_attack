from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import joblib
from io import BytesIO
import numpy as np
from fastapi.responses import JSONResponse, HTMLResponse
import re

app = FastAPI(title="Heart Attack Prediction API"
              description="Предсказание рисков сердечного приступа",
              version="1.0",
    contact={
        "name": "Alexandr Sorokin",
        "email": "as1983@yandex.ru"
    })

try:
    model = joblib.load("model.pkl")
    print("✅ Модель успешно загружена")
    if hasattr(model, 'feature_names_in_'):
        EXPECTED_FEATURES = model.feature_names_in_.tolist()
        print(f"📋 Ожидаемые фичи: {EXPECTED_FEATURES}")
    else:
        EXPECTED_FEATURES = [
            'age', 'cholesterol', 'heart_rate', 'diabetes', 'family_history', 'smoking', 
            'obesity', 'alcohol_consumption', 'exercise_hours_per_week', 'diet', 
            'previous_heart_problems', 'medication_use', 'stress_level', 'sedentary_hours_per_day', 
            'income', 'bmi', 'triglycerides', 'physical_activity_days_per_week', 'sleep_hours_per_day', 
            'blood_sugar', 'ckmb', 'troponin', 'gender', 'systolic_blood_pressure', 'diastolic_blood_pressure'
        ]
except Exception as e:
    print(f"❌ Ошибка загрузки модели: {e}")
    model = None
    EXPECTED_FEATURES = []

threshold = 0.3

@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <head>
            <title>Heart Attack Risk Prediction</title>
        </head>
        <body>
            <h1>Добро пожаловать в модель предсказания риска сердечного приступа</h1>
            <p>Загрузите CSV файл через <a href="/docs">Swagger UI</a></p>
            <p>Или отправьте POST запрос на /predict/ с файлом</p>
        </body>
    </html>
    """

def to_snake_case(column_name: str) -> str:
    snake_name = re.sub(r'[\s\-\.\(\)\/]+', '_', column_name)
    snake_name = re.sub(r'[^a-zA-Z0-9_]', '', snake_name)
    snake_name = snake_name.lower()
    snake_name = re.sub(r'_+', '_', snake_name)
    snake_name = snake_name.strip('_')
    return snake_name

def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    df_normalized = df.copy()
    new_columns = [to_snake_case(col) for col in df_normalized.columns]
    df_normalized.columns = new_columns
    return df_normalized

def fix_gender_column(df):
    if 'gender' in df.columns:
        print(f"🔧 Преобразование gender. Исходные значения: {df['gender'].unique()}")
        
        df['gender'] = df['gender'].astype(str).str.strip().str.lower()
        
        gender_mapping = {
            '1': 1, '1.0': 1, 'male': 1, 'м': 1, 'm': 1, 'мужской': 1,
            '0': 0, '0.0': 0, 'female': 0, 'ж': 0, 'f': 0, 'женский': 0
        }
        
        df['gender'] = df['gender'].map(gender_mapping)
        df['gender'] = df['gender'].fillna(0)
        df['gender'] = df['gender'].astype(int)
        
        print(f"🔧 Gender после преобразования: {df['gender'].unique()}")
    
    return df

def convert_numpy_types(obj):
    """
    Рекурсивно преобразует numpy типы в native Python типы для JSON сериализации
    """
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    else:
        return obj

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Только CSV файлы поддерживаются")
    
    if model is None:
        raise HTTPException(status_code=500, detail="Модель не загружена")
    
    try:
        contents = await file.read()
        df = pd.read_csv(BytesIO(contents))
        
        print(f"📊 Исходные колонки: {list(df.columns)}")
        print(f"📊 Количество записей: {len(df)}")

        # Сохраняем оригинальные ID если они есть
        if 'id' in df.columns:
            ids = df['id'].tolist()
        else:
            ids = [i+1 for i in range(len(df))]
        
        # Нормализуем названия колонок
        df_normalized = normalize_dataframe_columns(df)
        print(f"🔄 Нормализованные колонки: {list(df_normalized.columns)}")
        
        # Исправляем колонку gender
        df_normalized = fix_gender_column(df_normalized)
        
        # Создаем недостающие колонки с значениями по умолчанию
        for col in EXPECTED_FEATURES:
            if col not in df_normalized.columns:
                df_normalized[col] = 0
                print(f"⚠️ Добавлена недостающая колонка: {col}")
        
        # Убедимся что порядок колонок правильный
        df_final = df_normalized[EXPECTED_FEATURES]
        
        # Предсказание
        y_proba = model.predict_proba(df_final)[:, 1]
        predictions = (y_proba > threshold).astype(int)

        # Формируем результат в формате ID - Prediction
        results = []
        for i, (id_val, prediction, probability) in enumerate(zip(ids, predictions, y_proba)):
            results.append({
                "id": convert_numpy_types(id_val),
                "prediction": convert_numpy_types(prediction),
                "probability": convert_numpy_types(probability),
                "risk_category": "high_risk" if prediction == 1 else "low_risk"
            })

        response_data = {
            "filename": file.filename,
            "records_count": len(df),
            "results": results,
            "summary": {
                "high_risk_count": convert_numpy_types(sum(predictions)),
                "low_risk_count": convert_numpy_types(len(predictions) - sum(predictions)),
                "high_risk_percentage": f"{(sum(predictions) / len(predictions) * 100):.1f}%"
            }
        }
        
        return convert_numpy_types(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Ошибка обработки файла: {str(e)}")

@app.post("/predict_single/")
async def predict_single(data: dict):
    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Модель не загружена")
        
        # Преобразование в DataFrame
        df = pd.DataFrame([data])
        
        # Извлекаем ID если есть
        record_id = data.get('id', 1)
        
        # Нормализуем названия колонок
        df_normalized = normalize_dataframe_columns(df)
        
        # Исправляем колонку gender
        df_normalized = fix_gender_column(df_normalized)
        
        # Создаем недостающие колонки с значениями по умолчанию
        for col in EXPECTED_FEATURES:
            if col not in df_normalized.columns:
                df_normalized[col] = 0
        
        # Убедимся что порядок колонок правильный
        df_final = df_normalized[EXPECTED_FEATURES]
        
        # Предсказание
        y_proba = model.predict_proba(df_final)[:, 1][0]
        prediction = int(y_proba > threshold)
        
        response_data = {
            "id": convert_numpy_types(record_id),
            "prediction": convert_numpy_types(prediction),
            "probability": convert_numpy_types(y_proba),
            "risk_category": "high_risk" if prediction == 1 else "low_risk"
        }
        
        return convert_numpy_types(response_data)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)