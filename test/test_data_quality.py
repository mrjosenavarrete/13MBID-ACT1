import pandas as pd
from pandera.pandas import DataFrameSchema, Column
import pytest
import os
from datetime import datetime

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

def test_basico(datos_banco):
    df = datos_banco
    # Verificar que el DataFrame no está vacio
    assert not df.empty, "Dataframe vacio"
    # Verificar Nulos
    assert df.isnull().sum().sum() == 0, "El Dataframe contiene nulos"
    # Verificar duplicados
    assert df.duplicated.sum() == 0, "El Dataframe contiene filas duplicadas"
    # Verificar cantidad de columnas
    assert df.shape[1] == 21, f"El Dataframe debería contener 21 columnas, pero tiene {df.shape[1]}"

if __name__ == "__main__":

    output_dir = "docs/test_results"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = f"{output_dir}/test_results_{timestamp}.txt"

    try:
        test_esquema(datos_banco())
        test_basico(datos_banco())
        print("Todos los tests pasaron exitosamente.")
        with open(output_file, "w") as f:
            f.write("Todos los test pasaron exitosamente.\n")
    except AssertionError as e:
        print(f"Test fallido: {e}")
        with open(output_file, "w") as f:
            f.write(f"Test fallido: {e}\n")