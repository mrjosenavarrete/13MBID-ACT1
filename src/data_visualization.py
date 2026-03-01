import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def visualizar_datos(
    fuente: str = "data/raw/bank-additional-full.csv",
    salida: str = "docs/figures/"
):
    """
    Genera un conjunto de visualizaciones representativas del dataset
    Bank Marketing y las exporta como imágenes.
    """

    # Configuración visual
    sns.set(style="whitegrid")

    # Crear directorio de salida si no existe
    Path(salida).mkdir(parents=True, exist_ok=True)

    # Leer datos
    df = pd.read_csv(fuente, sep=';')

    # 1. Distribución de la variable objetivo
    plt.figure(figsize=(6, 4))
    sns.countplot(x="y", data=df)
    plt.title("Distribución de la variable objetivo")
    plt.xlabel("¿Contrató un depósito a plazo?")
    plt.ylabel("Cantidad de clientes")
    plt.tight_layout()
    plt.savefig(f"{salida}/01_distribucion_variable_objetivo.png")
    plt.close()

    # 2. Tipo de contacto vs aceptación del depósito
    plt.figure(figsize=(6, 4))
    sns.countplot(x="contact", hue="y", data=df)
    plt.title("Tipo de contacto vs aceptación del depósito")
    plt.xlabel("Tipo de contacto")
    plt.ylabel("Cantidad de clientes")
    plt.legend(title="Contrató depósito")
    plt.tight_layout()
    plt.savefig(f"{salida}/02_contacto_vs_objetivo.png")
    plt.close()

    # 3. Distribución de edad según aceptación
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="y", y="age", data=df)
    plt.title("Distribución de la edad según aceptación del depósito")
    plt.xlabel("¿Contrató un depósito a plazo?")
    plt.ylabel("Edad")
    plt.tight_layout()
    plt.savefig(f"{salida}/03_edad_vs_objetivo.png")
    plt.close()

    # 4. Número de contactos en la campaña vs aceptación
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="y", y="campaign", data=df)
    plt.title("Cantidad de contactos en la campaña vs aceptación")
    plt.xlabel("¿Contrató un depósito a plazo?")
    plt.ylabel("Cantidad de contactos")
    plt.tight_layout()
    plt.savefig(f"{salida}/04_campaign_vs_objetivo.png")
    plt.close()

    # 5. Resultado de campañas previas vs aceptación
    plt.figure(figsize=(7, 4))
    sns.countplot(x="poutcome", hue="y", data=df)
    plt.title("Resultado de campañas previas vs aceptación")
    plt.xlabel("Resultado de campaña previa")
    plt.ylabel("Cantidad de clientes")
    plt.legend(title="Contrató depósito")
    plt.tight_layout()
    plt.savefig(f"{salida}/05_poutcome_vs_objetivo.png")
    plt.close()

    # 6. Duración de la llamada vs aceptación (exploratorio)
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="y", y="duration", data=df)
    plt.title("Duración de la llamada vs aceptación (análisis exploratorio)")
    plt.xlabel("¿Contrató un depósito a plazo?")
    plt.ylabel("Duración de la llamada (segundos)")
    plt.tight_layout()
    plt.savefig(f"{salida}/06_duration_vs_objetivo.png")
    plt.close()

        # 7. Matriz de correlación de variables numéricas
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=["int64", "float64"])
    corr_matrix = numeric_cols.corr()

    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        annot=False,
        fmt=".2f",
        linewidths=0.5
    )
    plt.title("Matriz de correlación de variables numéricas")
    plt.tight_layout()
    plt.savefig(f"{salida}/07_matriz_correlacion.png")
    plt.close()

    # 8. Tasa de aceptación del depósito por mes
    tasa_mes = (
        df.groupby("month")["y"]
        .apply(lambda x: (x == "yes").mean())
        .reset_index(name="tasa_aceptacion")
    )

    plt.figure(figsize=(8, 4))
    sns.barplot(
        x="month",
        y="tasa_aceptacion",
        data=tasa_mes,
        order=["jan", "feb", "mar", "apr", "may", "jun",
               "jul", "aug", "sep", "oct", "nov", "dec"]
    )
    plt.title("Tasa de aceptación del depósito por mes")
    plt.xlabel("Mes")
    plt.ylabel("Tasa de aceptación")
    plt.tight_layout()
    plt.savefig(f"{salida}/08_tasa_aceptacion_por_mes.png")
    plt.close()


if __name__ == "__main__":
    visualizar_datos()