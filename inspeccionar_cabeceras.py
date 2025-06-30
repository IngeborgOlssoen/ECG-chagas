import os
import wfdb

# Carpeta que quieres inspeccionar
TRAINING_DATA_DIR = r'C:\Users\nicos\OneDrive\Escritorio\2025\Señales\proyecto\training_data'

print(f"--- Inspeccionando las cabeceras (.hea) en la carpeta: '{TRAINING_DATA_DIR}' ---")

# Lista para guardar los archivos que sí tienen la etiqueta positiva
positivos_encontrados = []

if not os.path.isdir(TRAINING_DATA_DIR):
    print(f"[ERROR] No se encuentra la carpeta '{TRAINING_DATA_DIR}'.")
    exit()

# Tomamos una muestra de los primeros 30 archivos para no llenar la pantalla
lista_archivos = [f for f in os.listdir(TRAINING_DATA_DIR) if f.endswith('.hea')][:30]

for filename in lista_archivos:
    record_path = os.path.join(TRAINING_DATA_DIR, os.path.splitext(filename)[0])
    try:
        header = wfdb.rdheader(record_path)
        print(f"\n--- Archivo: {filename} ---")
        print("Comentarios encontrados:")
        if header.comments:
            for comment in header.comments:
                print(f"  - '{comment}'")
                # Verificamos si encontramos la etiqueta que buscamos
                if 'chagas label: true' in comment.lower():
                    positivos_encontrados.append(filename)
        else:
            print("  - (Este archivo no tiene comentarios)")
    except Exception as e:
        print(f"Error leyendo {filename}: {e}")

print("\n--- Resumen de la Inspección ---")
if positivos_encontrados:
    print(f"Se encontraron {len(positivos_encontrados)} archivos con la etiqueta de Chagas positivo en la muestra.")
    print("Ejemplos:", positivos_encontrados)
else:
    print("¡PROBLEMA CONFIRMADO! No se encontró ningún archivo con el comentario 'Chagas label: True' en la muestra analizada.")
    print("Debes corregir tus scripts de preparación de datos iniciales para que añadan este comentario a los archivos de SAMITROP.")