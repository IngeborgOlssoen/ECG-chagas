import os
import numpy as np
import pandas as pd
import wfdb
import h5py
from tqdm import tqdm
import scipy.signal

# --- CONFIGURACIÓN ---
TRAINING_DATA_DIR = r'C:\Users\nicos\OneDrive\Escritorio\2025\Señales\proyecto\training_data'
OUTPUT_SIGNALS_FILE = 'x_chagas.hdf5'
OUTPUT_LABELS_FILE = 'y_chagas.csv'
TARGET_LENGTH = 1000
# --------------------

def get_label_from_source(header):
    """
    Lee los comentarios y determina la etiqueta basándose en la FUENTE del dato.
    """
    for comment in header.comments:
        cleaned_comment = comment.strip().lstrip('#').strip().lower()
        if cleaned_comment.startswith('source:'):
            source_str = cleaned_comment.split(':')[1].strip().lower()
            if 'sami-trop' in source_str or 'code' in source_str:
                return 1 # Chagas Positivo
            elif 'ptb-xl' in source_str:
                return 0 # Chagas Negativo
    return None

def run_preparation(data_dir):
    print(f"--- Iniciando la creación de archivos HDF5 y CSV desde: '{data_dir}' ---")

    if not os.path.isdir(data_dir):
        print(f"[ERROR] El directorio de datos '{data_dir}' no fue encontrado. Verifica la ruta.")
        return

    all_signals, all_labels = [], []

    record_files = [f for f in os.listdir(data_dir) if f.endswith('.hea')]
    
    if not record_files:
        print(f"[ERROR] No se encontraron archivos .hea en '{data_dir}'.")
        return

    print(f"Encontrados {len(record_files)} registros para procesar.")
    
    for record_file in tqdm(record_files, desc="Procesando y unificando TODOS los ECGs"):
        record_path_base = os.path.join(data_dir, os.path.splitext(record_file)[0])
        
        try:
            header = wfdb.rdheader(record_path_base)
            label = get_label_from_source(header)

            if label is None:
                continue

            signal, meta = wfdb.rdsamp(record_path_base)
            current_length = signal.shape[0]
            
            # ===================================================================
            # === LÓGICA DE RESAMPLING UNIVERSAL ===
            # ===================================================================
            # Si la señal no tiene la longitud objetivo, la remuestreamos.
            if current_length != TARGET_LENGTH:
                processed_signal = scipy.signal.resample(signal, TARGET_LENGTH)
            else:
                # Si ya tiene la longitud correcta, la usamos tal cual.
                processed_signal = signal
            
            all_signals.append(processed_signal)
            all_labels.append(label)

        except Exception as e:
            print(f"\nError procesando el archivo {record_file}: {e}. Saltando.")

    if not all_signals:
        print("No se procesó ningún archivo con éxito.")
        return

    X = np.array(all_signals, dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32).reshape(-1, 1)

    print(f"\nProcesamiento completo. Se procesaron {len(all_signals)} registros.")
    print(f"Forma final de X: {X.shape}, Forma final de y: {y.shape}")
    print(f"Total de casos positivos (Chagas=1) encontrados: {np.sum(y)}")
    print(f"Total de casos negativos (Chagas=0) encontrados: {len(y) - np.sum(y)}")

    print(f"Guardando señales en '{OUTPUT_SIGNALS_FILE}'...")
    with h5py.File(OUTPUT_SIGNALS_FILE, 'w') as f:
        f.create_dataset('tracings', data=X, compression='gzip')

    print(f"Guardando etiquetas en '{OUTPUT_LABELS_FILE}'...")
    pd.DataFrame(y, columns=['chagas']).to_csv(OUTPUT_LABELS_FILE, index=False)
    
    print("\n¡Éxito! Los archivos de datos para el entrenamiento han sido creados con TODAS las muestras disponibles.")

if __name__ == '__main__':
    run_preparation(TRAINING_DATA_DIR)