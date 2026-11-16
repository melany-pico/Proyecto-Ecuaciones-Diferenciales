
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
ORIGEN = BASE / "03_tablas" / "tablas_y_figuras_articulo"
DESTINO = BASE / "04_manuscrito" / "revision_resultados.txt"

archivos = [
    "tabla_1_escasez.csv",
    "tabla_2_ruido_tres_metodos.csv",
    "tabla_3_primarias_experimento_34.csv",
    "tabla_4_primarias_experimento_44.csv",
]

with DESTINO.open("w", encoding="utf-8") as salida:
    salida.write("REVISION DE RESULTADOS CIENTIFICOS\n")
    salida.write("=" * 80 + "\n\n")

    for nombre in archivos:
        ruta = ORIGEN / nombre
        df = pd.read_csv(ruta)

        salida.write(f"\n{nombre}\n")
        salida.write("-" * 80 + "\n")
        salida.write(f"Filas: {len(df)}\n")
        salida.write(f"Columnas: {len(df.columns)}\n\n")
        salida.write(df.to_string(index=False))
        salida.write("\n\n")

print("Informe generado correctamente:")
print(DESTINO)
