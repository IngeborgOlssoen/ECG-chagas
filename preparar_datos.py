import pandas as pd
import numpy as np
import wfdb
import h5py
import os

# --- CONFIGURACIÓN ---
# ¡IMPORTANTE! Modifica esta línea para que apunte a la carpeta donde descomprimiste PTB-XL.
PATH_A_PTBXL = r'C:\Users\nicos\OneDrive\Escritorio\2025\Señales\proyecto\ptbxl'
# --------------------

def preprocesar_ptbxl(path_a_datos, sampling_rate=100):
    """
    Esta función lee los datos crudos del dataset PTB-XL, procesa las señales y las etiquetas,
    y guarda los resultados en los archivos x.hdf5 y y.csv que necesita Train.py.
    """
    print("--- Iniciando pre-procesamiento de PTB-XL ---")

    try:
        # 1. Cargar el archivo CSV principal que contiene toda la metadata
        print(f"Leyendo metadatos de: {os.path.join(path_a_datos, 'ptbxl_database.csv')}")
        ptbxl_data = pd.read_csv(os.path.join(path_a_datos, 'ptbxl_database.csv'), index_col='ecg_id')

        # 2. Cargar las señales de ECG
        print(f"Cargando señales... Esto puede tardar unos minutos.")
        # La función 'rdrecord' de wfdb lee los archivos .hea y .dat
        signals = []
        for ecg_id in ptbxl_data.index:
            # Construimos el nombre del archivo. Usamos os.path.join para compatibilidad.
            # Los archivos están en carpetas 'records100/00000/' etc.
            filename = os.path.join(path_a_datos, ptbxl_data.loc[ecg_id, 'filename_lr'] if sampling_rate == 100 else ptbxl_data.loc[ecg_id, 'filename_hr'])
            signal, meta = wfdb.rdsamp(filename)
            signals.append(signal)
        
        # Convertimos la lista de señales a un único array de NumPy
        X = np.array(signals)
        print(f"Procesamiento de señales completo. Forma de X: {X.shape}")

        # 3. Procesar las etiquetas (diagnósticos)
        # El script Train.py espera 8 columnas. Usaremos 8 diagnósticos comunes.
        # Puedes adaptar esto si necesitas otras clases.
        y = ptbxl_data.copy()
        
        # Seleccionamos las columnas de diagnóstico y las convertimos a binario (0 o 1)
        y['NORM'] = y.scp_codes.str.contains('NORM').astype(int)
        y['MI'] = y.scp_codes.str.contains('MI').astype(int)
        y['STTC'] = y.scp_codes.str.contains('STTC').astype(int)
        y['CD'] = y.scp_codes.str.contains('CD').astype(int)
        y['HYP'] = y.scp_codes.str.contains('HYP').astype(int)
        y['6AVB'] = y.scp_codes.str.contains('AVB').astype(int) # Bloqueo AV
        y['LPFB'] = y.scp_codes.str.contains('LPFB').astype(int) # Fascicular izquierdo
        y['1AVB'] = y.scp_codes.str.contains('1AVB').astype(int) # Bloqueo AV de primer grado
        
        # Nos quedamos solo con las 8 columnas de etiquetas
        y = y[['NORM', 'MI', 'STTC', 'CD', 'HYP', '6AVB', 'LPFB', '1AVB']]
        print(f"Procesamiento de etiquetas completo. Forma de y: {y.shape}")

        # 4. Guardar los archivos procesados
        print("Guardando archivos procesados 'x.hdf5' y 'y.csv'...")
        with h5py.File('x.hdf5', 'w') as f:
            f.create_dataset('tracings', data=X, compression='gzip')
        
        y.to_csv('y.csv', index=False)
        
        print("\n--- ¡Pre-procesamiento finalizado con éxito! ---")
        print("Ya puedes ejecutar Train.py")

    except FileNotFoundError:
        print("\n[ERROR] No se pudo encontrar la carpeta de PTB-XL. Por favor, verifica la variable 'PATH_A_PTBXL' en este script.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error inesperado: {e}")

if __name__ == '__main__':
    # Verificamos si la librería wfdb está instalada
    try:
        import wfdb
    except ImportError:
        print("[ERROR] La librería 'wfdb' no está instalada. Por favor, ejecute 'pip install wfdb' en su entorno.")
    else:
        preprocesar_ptbxl(PATH_A_PTBXL)