
from pathlib import Path
from zipfile import ZipFile
import shutil

BASE = Path(__file__).resolve().parent
ORIGEN = BASE / "05_zips_originales"

PRINCIPALES = BASE / "01_figuras_principales"
SUPLEMENTARIAS = BASE / "02_figuras_suplementarias"
TABLAS = BASE / "03_tablas"

for carpeta in (PRINCIPALES, SUPLEMENTARIAS, TABLAS):
    carpeta.mkdir(parents=True, exist_ok=True)

total = 0

for archivo_zip in sorted(ORIGEN.glob("*.zip")):
    print(f"\nProcesando: {archivo_zip.name}")

    # Cada ZIP se conserva en su propia subcarpeta
    nombre_grupo = archivo_zip.stem

    with ZipFile(archivo_zip, "r") as zip_ref:
        for elemento in zip_ref.infolist():
            if elemento.is_dir():
                continue

            nombre = Path(elemento.filename).name
            if not nombre or nombre.startswith("."):
                continue

            extension = Path(nombre).suffix.lower()

            if extension == ".csv":
                destino_base = TABLAS
            elif extension in (".png", ".jpg", ".jpeg",
                               ".pdf", ".svg", ".eps"):
                destino_base = SUPLEMENTARIAS
            else:
                continue

            destino = destino_base / nombre_grupo / nombre
            destino.parent.mkdir(parents=True, exist_ok=True)

            # No sobrescribir archivos existentes
            if destino.exists():
                print(f"  Ya existe: {nombre}")
                continue

            with zip_ref.open(elemento) as origen:
                with destino.open("wb") as salida:
                    shutil.copyfileobj(origen, salida)

            print(f"  Extraido: {nombre}")
            total += 1

print("\n====================================")
print("ORGANIZACION FINALIZADA")
print(f"Archivos nuevos extraidos: {total}")
print("ZIP originales conservados.")
print("====================================")
