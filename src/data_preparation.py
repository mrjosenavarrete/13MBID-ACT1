import pandas as pd
import numpy as np
import datetime

INPUT_CSV = 'data/raw/bank-additional-full.csv'
OUTPUT_CSV = 'data/processed/bank-processed.csv'


def preprocess_data(input_path=INPUT_CSV, output_path=OUTPUT_CSV):
    df = pd.read_csv(input_path, sep=';')

    # Adaptar nombre de columnas
    df.columns = df.columns.str.replace(".", "_")

    # Reemplazar valores 'unknown' por NaN
    df.replace('unknown', np.nan, inplace=True)

    # Eliminar columna 'default'
    df.drop(columns=["default"], inplace=True)

    # Eliminar filas con valores nulos
    df.dropna(inplace=True)

    # Eliminar duplicados
    df = df.drop_duplicates()

    # Transformaciones adicionales

    # 1. Codificar variable objetivo
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    # 2. Crear rangos de edad
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 100],
        labels=["young", "adult", "senior", "elder"]
    )

    # 3. Normalizar variables categóricas
    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].apply(
        lambda x: x.str.lower().str.strip()
    )

    # Guardar dataset procesado
    df.to_csv(output_path, index=False)

    return df.shape


if __name__ == "__main__":
    dimensiones = preprocess_data()

    # Generar timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'docs/transformations/transformation_{timestamp}.txt'

    with open(filename, 'w', encoding="utf-8") as f:
        f.write(f"Transformaciones realizadas - {timestamp}\n")
        f.write("-" * 50 + "\n")
        f.write("- Se reemplazaron los valores 'unknown' por NaN\n")
        f.write("- Se eliminaron las filas con valores nulos\n")
        f.write("- Se eliminararon filas duplicadas\n")
        f.write("- Se eliminó la columna 'default' por alta proporción de valores faltantes\n")
        f.write("- Se transformó la variable objetivo 'y' a formato binario ('target')\n")
        f.write("- Se creó la variable derivada 'age_group' a partir de la edad\n")
        f.write("- Se normalizaron las variables categóricas (lowercase y trim)\n")
        f.write(f"- Cantidad de filas finales: {dimensiones[0]}\n")
        f.write(f"- Cantidad de columnas finales: {dimensiones[1]}\n")