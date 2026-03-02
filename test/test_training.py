import sys
from pathlib import Path
import json
import pytest
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.train_model import train_model


def test_training_metrics_regression(tmp_path):
    """
    Test de regresión de métricas:
    - Compara métricas actuales contra una baseline
    - Deja evidencia en docs/test_training con timestamp
    """

    project_root = Path(__file__).resolve().parents[1]
    baseline_path = project_root / "metrics" / "model_metrics.json"

    # Directorio de outputs
    output_dir = project_root / "docs" / "test_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"training_test_{timestamp}.txt"

    # Si no existe baseline → skip + log
    if not baseline_path.exists():
        with open(output_file, "w") as f:
            f.write("TEST TRAINING METRICS\n")
            f.write("=" * 40 + "\n")
            f.write("Baseline no encontrada.\n")
            f.write(f"Ruta esperada: {baseline_path}\n")
            f.write("Test marcado como SKIPPED.\n")

        pytest.skip(
            f"No se encontró la baseline en {baseline_path}. "
            "Ejecuta el entrenamiento para generarla."
        )

    # Cargar baseline
    with open(baseline_path) as f:
        baseline = json.load(f)

    # Rutas aisladas
    data_path = project_root / "data" / "processed" / "bank-processed.csv"
    model_out = tmp_path / "model.pkl"
    prep_out = tmp_path / "preprocessor.pkl"
    metrics_out = tmp_path / "train_metrics.json"

    # Ejecutar entrenamiento
    _, _, metrics = train_model(
        data_path=str(data_path),
        model_output_path=str(model_out),
        preprocessor_output_path=str(prep_out),
        metrics_output_path=str(metrics_out),
    )

    # ---- LOG DE RESULTADOS (SIEMPRE SE ESCRIBE) ----
    with open(output_file, "w") as f:
        f.write("TEST TRAINING METRICS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Timestamp: {timestamp}\n\n")

        f.write("Baseline metrics:\n")
        for k, v in baseline.items():
            f.write(f"  {k}: {v}\n")

        f.write("\nCurrent metrics:\n")
        for k, v in metrics.items():
            f.write(f"  {k}: {v}\n")

    # ---- ASSERTS (pytest decide PASS / FAIL) ----
    assert set(metrics.keys()) == set(baseline.keys())

    atol = 1e-9
    for k in baseline:
        assert metrics[k] == pytest.approx(
            baseline[k], rel=0, abs=atol
        ), f"Métrica {k} cambió: baseline={baseline[k]} nueva={metrics[k]}"