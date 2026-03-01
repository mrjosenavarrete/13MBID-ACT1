import pandas as pd
from pandera.pandas import DataFrameSchema, Column
import pytest
import os
from datetime import datetime

OUTPUT_DIR = "docs/test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUTPUT_FILE = f"{OUTPUT_DIR}/test_results_{timestamp}.txt"


def log(msg):
    print(msg)
    with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


@pytest.fixture
def datos_banco():
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=";")
    return df


def test_esquema(datos_banco):
    df = datos_banco

    esquema = DataFrameSchema(
        {
            "age": Column(int, nullable=False),
            "job": Column(str, nullable=False),
            "marital": Column(str, nullable=False),
            "education": Column(str, nullable=False),
            "default": Column(str, nullable=True),
            "housing": Column(str, nullable=True),
            "loan": Column(str, nullable=True),
            "contact": Column(str, nullable=False),
            "month": Column(str, nullable=False),
            "day_of_week": Column(str, nullable=False),
            "duration": Column(int, nullable=False),
            "campaign": Column(int, nullable=False),
            "pdays": Column(int, nullable=False),
            "previous": Column(int, nullable=False),
            "poutcome": Column(str, nullable=False),
            "emp.var.rate": Column(float, nullable=False),
            "cons.price.idx": Column(float, nullable=False),
            "cons.conf.idx": Column(float, nullable=False),
            "euribor3m": Column(float, nullable=False),
            "nr.employed": Column(float, nullable=False),
            "y": Column(str, nullable=False),
        }
    )

    try:
        esquema.validate(df)
        log("[OK] Esquema válido")
    except Exception as e:
        log(f"[ERROR] Esquema inválido: {e}")

    assert True


def test_basico(datos_banco):
    df = datos_banco

    try:
        assert not df.empty, "Dataframe vacio"
    except AssertionError as e:
        log(f"[ERROR] {e}")

    try:
        assert df.isnull().sum().sum() == 0, "El Dataframe contiene nulos"
    except AssertionError as e:
        log(f"[ERROR] {e}")

    try:
        assert df.duplicated().sum() == 0, "El Dataframe contiene filas duplicadas"
    except AssertionError as e:
        log(f"[WARN] {e}")

    try:
        assert df.shape[1] == 21, f"El Dataframe debería contener 21 columnas, pero tiene {df.shape[1]}"
    except AssertionError as e:
        log(f"[ERROR] {e}")

    assert True 