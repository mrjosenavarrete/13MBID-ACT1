import pandas as pd
from pandera.pandas import DataFrameSchema, Column
import pytest

@pytest.fixture
def datos_banco():
    df = pd.read_csv('data/raw/bank-additional-full.csv', sep=';')
    return df

def test_esquema(datos_banco):
    df = datos_banco
    esquema = DataFrameSchema(
        {
            # Datos del cliente
            "age": Column(int, nullable=False),
            "job": Column(str, nullable=False),
            "marital": Column(str, nullable=False),
            "education": Column(str, nullable=False),
            "default": Column(str, nullable=True),
            "housing": Column(str, nullable=True),
            "loan": Column(str, nullable=True),

            # Datos de la última campaña
            "contact": Column(str, nullable=False),
            "month": Column(str, nullable=False),
            "day_of_week": Column(str, nullable=False),
            "duration": Column(int, nullable=False),

            # Otros atributos
            "campaign": Column(int, nullable=False),
            "pdays": Column(int, nullable=False),
            "previous": Column(int, nullable=False),
            "poutcome": Column(str, nullable=False),

            # Contexto social y económico
            "emp.var.rate": Column(float, nullable=False),
            "cons.price.idx": Column(float, nullable=False),
            "cons.conf.idx": Column(float, nullable=False),
            "euribor3m": Column(float, nullable=False),
            "nr.employed": Column(float, nullable=False),

            # Variable objetivo 
            "y": Column(str, nullable=False),
        }
    )

    esquema.validate(df)