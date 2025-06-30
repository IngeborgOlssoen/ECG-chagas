import torch
import torch.nn as nn
import numpy as np
import wfdb
import argparse
from Train import S4Model # Reutilizamos la definición del modelo desde Train.py

# --- CONFIGURACIÓN ---
# Estas son las 8 clases de diagnóstico que usamos en el script preparar_datos.py
# El orden es importante.
CLASES_DIAGNOSTICO = ['Normal', 'Infarto de Miocardio', 'Anomalía ST/T', 'Trastorno de Conducción', 'Hipertrofia', 'Bloqueo AV', 'Bloqueo Fascicular Izquierdo', 'Bloqueo AV 1er Grado']
# --------------------

def cargar_modelo(ruta_modelo, d_input=12, d_output=8):
    """Carga el modelo S4D y sus pesos entrenados."""
    print(f"==> Cargando modelo desde: {ruta_modelo}")
    
    modelo = S4Model(
        d_input=d_input,
        d_output=d_output,
        d_model=128,
        n_layers=4,
        dropout=0.1,
        prenorm=False
    )
    
    # Cargamos los pesos que guardamos durante el entrenamiento
    # Usamos map_location='cpu' para que funcione incluso si no tienes GPU en este momento
    modelo.load_state_dict(torch.load(ruta_modelo, map_location=torch.device('cpu')))
    
    # Ponemos el modelo en modo de evaluación
    modelo.eval()
    print("==> Modelo cargado exitosamente.")
    return modelo

def predecir_ecg(modelo, ruta_ecg):
    """Carga un solo ECG, lo preprocesa y realiza una predicción."""
    print(f"\n==> Realizando inferencia para: {ruta_ecg}")
    
    try:
        # 1. Cargar la señal del ECG usando wfdb
        # wfdb necesita la ruta sin la extensión .hea o .mat
        signal, meta = wfdb.rdsamp(ruta_ecg)
        
        # 2. Preprocesar la señal para que coincida con la entrada del modelo
        # La forma debe ser (1, L, C) -> (1, 1000, 12)
        if signal.shape[0] != 1000:
            raise ValueError(f"La señal tiene una longitud de {signal.shape[0]}, pero el modelo espera 1000.")
        
        signal = signal.astype(np.float32)
        # Añadimos una dimensión de "batch" al principio -> (1, 1000, 12)
        signal_tensor = torch.from_numpy(signal).unsqueeze(0)
        
        # 3. Realizar la inferencia
        print("==> Procesando con el modelo S4D...")
        with torch.no_grad(): # No necesitamos calcular gradientes para la inferencia
            predicciones = modelo(signal_tensor)
            
        # El resultado es un tensor de probabilidades, lo convertimos a una lista de Python
        probabilidades = predicciones.squeeze().tolist()
        
        # 4. Mostrar los resultados de forma clara
        print("\n--- Resultados de la Inferencia ---")
        for i, clase in enumerate(CLASES_DIAGNOSTICO):
            print(f"- {clase:<28}: {probabilidades[i]:.2%}")
            
    except FileNotFoundError:
        print(f"\n[ERROR] No se pudo encontrar el archivo ECG: {ruta_ecg}.hea o {ruta_ecg}.mat")
        print("Asegúrate de que la ruta sea correcta y no incluya la extensión.")
    except Exception as e:
        print(f"\n[ERROR] Ocurrió un error inesperado: {e}")

if __name__ == '__main__':
    # Creamos un parser para recibir los argumentos desde la terminal
    parser = argparse.ArgumentParser(description='Script de Inferencia para un solo ECG con S4D-ECG')
    parser.add_argument('--model_path', type=str, required=True, help='Ruta al archivo del modelo entrenado (ej. ./s4_results/S4D/model.pt)')
    parser.add_argument('--ecg_path', type=str, required=True, help='Ruta al archivo ECG a predecir (sin la extensión .hea o .mat)')
    
    args = parser.parse_args()
    
    # Cargamos el modelo
    modelo_entrenado = cargar_modelo(args.model_path)
    
    # Realizamos la predicción
    predecir_ecg(modelo_entrenado, args.ecg_path)