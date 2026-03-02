from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import os
from typing import Any, Dict
import joblib

app = FastAPI(
    title="Modelo de Clasificación de Clientes Bancarios",
    description="API para predecir si un cliente bancario suscribirá un depósito a plazo fijo.",
    version="1.0.0",
)

class PredictionRequest(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    housing: Optional[str] = None
    loan: Optional[str] = None
    contact: str
    month: str
    day_of_week: str
    duration: int
    campaign: int
    previous: int
    poutcome: str
    emp_var_rate: float
    cons_price_idx: float
    cons_conf_idx: float
    euribor3m: float
    nr_employed: float
    contacted_before: str

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35,
                "job": "technician",
                "marital": "married",
                "education": "tertiary",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 245,
                "campaign": 1,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp_var_rate": 1.4,
                "cons_price_idx": 93.918,
                "cons_conf_idx": -42.7,
                "euribor3m": 4.857,
                "nr_employed": 5191.0,
                "contacted_before": "no"
            }
        }

class PredictionResponse(BaseModel):
    prediction: str  # yes o no
    probability: Dict[str, float]
    model_info: Dict[str, Any]

# Cargar el modelo y el preprocesador
MODEL_PATH = "models/decision_tree_model.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = None
preprocessor = None

if not os.path.exists(MODEL_PATH):
    print(f"⚠️  ADVERTENCIA: No se encontró el modelo en {MODEL_PATH}")
else:
    try:
        model = joblib.load(MODEL_PATH)
        print(f"✅ Modelo cargado exitosamente desde {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")

if not os.path.exists(PREPROCESSOR_PATH):
    print(f"⚠️  ADVERTENCIA: No se encontró el preprocesador en {PREPROCESSOR_PATH}")
else:
    try:
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        print(f"✅ Preprocesador cargado exitosamente desde {PREPROCESSOR_PATH}")
    except Exception as e:
        print(f"❌ Error al cargar el preprocesador: {e}")

@app.get("/")
def root():
    return {
        "message": "API del modelo de clasificación de clientes bancarios",
        "status": "activa",
        "modelo_cargado": model is not None,
        "preprocesador_cargado": preprocessor is not None
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=500, 
            detail="El modelo o preprocesador no está cargado correctamente."
        )

    try:
        # Convertir la solicitud a DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # APLICAR EL MISMO PREPROCESAMIENTO QUE EN ENTRENAMIENTO
        # 1. Convertir columnas enteras a float (como en create_preprocessor)
        int_cols = input_data.select_dtypes(include=['int64']).columns
        for col in int_cols:
            input_data[col] = input_data[col].astype('float64')
        
        # 2. Aplicar el preprocesador (OneHotEncoder, RobustScaler, etc.)
        input_prep = preprocessor.transform(input_data)
        
        # 3. Hacer predicción con el modelo
        prediction = model.predict(input_prep)[0]
        probability = model.predict_proba(input_prep)[0]

        # Convertir predicción numérica a etiqueta
        prediction_label = "yes" if prediction == 1 else "no"
        
        class_labels = model.classes_
        probability_dict = {str(class_labels[i]): float(probability[i]) for i in range(len(class_labels))}
        
        model_info = {
            "model_type": type(model).__name__,
        }

        return PredictionResponse(
            prediction=prediction_label,
            probability=probability_dict,
            model_info=model_info
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error al procesar la solicitud: {str(e)}")

@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None and preprocessor is not None else "degraded",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None,
        "model_path": MODEL_PATH,
        "preprocessor_path": PREPROCESSOR_PATH
    }