import pandas as pd
import numpy as np
from datetime import datetime

INPUT_CSV = 'data/raw/bank-additional-full.csv'
OUTPUT_CSV = 'data/processed/bank-processed.csv'

def preprocess_data(input_pach=INPUT_CSV, output_path = OUTPUT_CSV):
    df = pd.read_csv(input_pach, sep = ';')

    # Adaptar nombre de columnas
    df.columns = df.columns.str.replace(".", "_")

    # Valores unknown
    df.replace('unknown', np.nan, inplace=True)

    #Se elimina la columna default
    df.drop(columns=["default"], inplace=True)

    # Se hace un filtro para eliminar las filas que tienen valores nulos
    df.dropna(inplace=True)

    # Se eliminan duplicados
    df = df.drop_duplicates()

    # Guardar en path de salida
    df.to_csv(output_path, index = False)

    return df.shape

import datetime

if __name__ == "__main__":
    dimensiones = preprocess_data()
    
    # Generar timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'docs/transformations/transformation_{timestamp}.txt'
    
    with open(filename, 'w') as f:
        f.write(f"Transformaciones realizadas - {timestamp}\n")
        f.write("-" * 40 + "\n")
        f.write("- Se reemplazaron los valores 'unknown' por NaN\n")
        f.write("- Se eliminaron las filas con valores nulos\n")
        f.write("- Se eliminaron las filas duplicadas\n")
        f.write("- Se elimino la columna 'default' debido a la alta cantidad de valores nulos\n")
        f.write(f"- Cantidad de filas finales: {dimensiones[0]}\n")
        f.write(f"- Cantidad de columnas finales: {dimensiones[1]}\n")