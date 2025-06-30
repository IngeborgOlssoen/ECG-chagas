'''
The code is modified by Zhaojing Huang based on the work below. 

Gu, Albert, et al. "On the parameterization and initialization of 
diagonal state space models." Advances in Neural Information Processing 
Systems 35 (2022): 35971-35983.

This version has been refactored to be safely importable and to handle
multiprocessing issues on Windows.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import onnx
from torch.nn.parallel import DataParallel
import matplotlib.pyplot as plt

import os
import sys # Importamos sys para detectar el sistema operativo
import argparse

import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.s4.s4 import S4
from src.models.s4.s4d import S4D
from tqdm.auto import tqdm
import csv

# ===================================================================
# SECCIÓN 1: DEFINICIONES DE CLASES Y FUNCIONES
# (Esta parte se puede importar de forma segura desde otros scripts)
# ===================================================================

# Determinar la función de dropout correcta según la versión de PyTorch
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    # Para versiones anteriores, Dropout2d puede funcionar como un apaño.
    # Idealmente, el entorno ya está corregido, pero esto añade robustez.
    dropout_fn = nn.Dropout2d


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, seed=42):
        self.X = X
        self.y = y
        self.seed = seed
        np.random.seed(self.seed)
        self.indices = np.random.permutation(len(self.X))

    def __getitem__(self, index):
        idx = self.indices[index]
        x = self.X[idx].astype(np.float32)
        label = self.y[idx].astype(np.float32)
        return torch.tensor(x), torch.tensor(label)

    def __len__(self):
        return len(self.X)

class S4Model(nn.Module):
    def __init__(self, d_input, d_output=10, d_model=256, n_layers=4, dropout=0.2, prenorm=False, lr=0.001):
        super().__init__()
        self.prenorm = prenorm
        self.encoder = nn.Linear(d_input, d_model)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S4D(d_model, dropout=dropout, transposed=True, lr=min(0.001, lr)))
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        self.decoder = nn.Linear(d_model, d_output)

    def forward(self, x):
        x = self.encoder(x)
        x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)
        x = x.transpose(-1, -2)
        x = x.mean(dim=1)
        x = self.decoder(x)
        return nn.functional.sigmoid(x)

def setup_optimizer(model, lr, weight_decay, epochs):
    all_parameters = list(model.parameters())
    params = [p for p in all_parameters if not hasattr(p, "_optim")]
    optimizer = optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    hps = [getattr(p, "_optim") for p in all_parameters if hasattr(p, "_optim")]
    hps = [dict(s) for s in sorted(list(dict.fromkeys(frozenset(hp.items()) for hp in hps)))]
    for hp in hps:
        params = [p for p in all_parameters if getattr(p, "_optim", None) == hp]
        optimizer.add_param_group({"params": params, **hp})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    # Print optimizer info
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    print("==> Optimizer Config:")
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([f"Optimizer group {i}", f"{len(g['params'])} tensors"] + [f"{k} {v}" for k, v in group_hps.items()]))
    return optimizer, scheduler

def train(model, trainloader, device, criterion, optimizer, d_output):
    model.train()
    # ... (El resto de la función train sin cambios, pero ahora recibe sus dependencias)
    train_loss = 0
    correct = 0
    total = 0
    epsilon = 1e-8

    precision_list_train = []
    recall_list_train = []
    specificity_list_train = []
    accuracy_list_train = []
    auroc_list_train = [] 

    targets_flat_all = [] 
    outputs_flat_all = [] 

    pbar = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        predicted = outputs.gt(0.5).long()
        tp = torch.zeros(d_output, dtype=torch.float)
        fp = torch.zeros(d_output, dtype=torch.float)
        fn = torch.zeros(d_output, dtype=torch.float)
        tn = torch.zeros(d_output, dtype=torch.float)
        for c in range(d_output):
            tp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 1)).sum().item()
            fp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 0)).sum().item()
            fn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 1)).sum().item()
            tn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 0)).sum().item()
        precision = tp.sum() / (tp.sum() + fp.sum()+ epsilon)
        recall = tp.sum() / (tp.sum() + fn.sum()+ epsilon)
        specificity = tn.sum() / (tn.sum() + fp.sum()+ epsilon)
        total += targets.size(0) * d_output
        correct += predicted.eq(targets).sum().item()

        targets_flat = targets.cpu().detach().numpy().flatten()
        outputs_flat = outputs.cpu().detach().numpy().flatten()
        targets_flat_all.extend(targets_flat.tolist())
        outputs_flat_all.extend(outputs_flat.tolist())

        if len(np.unique(targets_flat_all)) > 1:
            auroc_batch = roc_auc_score(targets_flat_all, outputs_flat_all)
        else:
            auroc_batch = None

        if auroc_batch is not None:
            auroc_list_train.append(auroc_batch)

        precision_list_train.append(precision.item())
        recall_list_train.append(recall.item())
        specificity_list_train.append(specificity.item())
        accuracy_list_train.append(correct / total)
        
        pbar.set_description(
            'Loss: %.3f | Acc: %.3f%% (%d/%d) | Prec: %.3f | Rec: %.3f | Spec: %.3f | AUROC: %s' %
            (train_loss / (batch_idx + 1), 100. * correct / total, correct, total,
             precision, recall, specificity, str(round(auroc_batch, 3)) if auroc_batch is not None else "N/A")
        )
    
    metrics = {
        'loss': train_loss / len(trainloader),
        'precision': np.mean(precision_list_train),
        'recall': np.mean(recall_list_train),
        'specificity': np.mean(specificity_list_train),
        'accuracy': np.mean(accuracy_list_train),
        'auroc': auroc_list_train[-1] if auroc_list_train else 0
    }
    return metrics


def eval_model(model, dataloader, device, criterion, d_output):
    model.eval()
    # ... (El resto de la función eval sin cambios, pero ahora recibe sus dependencias)
    eval_loss = 0
    correct = 0
    total = 0
    epsilon = 1e-8

    precision_list_eval = []
    recall_list_eval = []
    specificity_list_eval = []
    accuracy_list_eval = []
    auroc_list_eval = []

    targets_flat_all = [] 
    outputs_flat_all = []

    with torch.no_grad():
        pbar = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        for batch_idx, (inputs, targets) in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            eval_loss += loss.item()
            predicted = outputs.gt(0.5).long()
            tp = torch.zeros(d_output, dtype=torch.float).to(device)
            fp = torch.zeros(d_output, dtype=torch.float).to(device)
            fn = torch.zeros(d_output, dtype=torch.float).to(device)
            tn = torch.zeros(d_output, dtype=torch.float).to(device)
            for c in range(d_output):
                tp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 1)).sum().item()
                fp[c] = ((predicted[:, c] == 1) & (targets[:, c] == 0)).sum().item()
                fn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 1)).sum().item()
                tn[c] = ((predicted[:, c] == 0) & (targets[:, c] == 0)).sum().item()
            precision = tp.sum() / (tp.sum() + fp.sum()+ epsilon)
            recall = tp.sum() / (tp.sum() + fn.sum()+ epsilon)
            specificity = tn.sum() / (tn.sum() + fp.sum()+ epsilon)
            total += targets.size(0) * d_output
            correct += predicted.eq(targets).sum().item()

            targets_flat = targets.cpu().detach().numpy().flatten()
            outputs_flat = outputs.cpu().detach().numpy().flatten()
            targets_flat_all.extend(targets_flat.tolist())
            outputs_flat_all.extend(outputs_flat.tolist())

            if len(np.unique(targets_flat_all)) > 1:
                auroc_batch = roc_auc_score(targets_flat_all, outputs_flat_all)
            else:
                auroc_batch = None

            if auroc_batch is not None:
                auroc_list_eval.append(auroc_batch)

            precision_list_eval.append(precision.item())
            recall_list_eval.append(recall.item())
            specificity_list_eval.append(specificity.item())
            accuracy_list_eval.append(correct / total)
            
            pbar.set_description(
                'Loss: %.3f | Acc: %.3f%% (%d/%d) | Prec: %.3f | Rec: %.3f | Spec: %.3f | AUROC: %s' %
                (eval_loss / (batch_idx + 1), 100. * correct / total, correct, total,
                 precision, recall, specificity, str(round(auroc_batch, 3)) if auroc_batch is not None else "N/A")
            )
    
    val_acc = 100. * correct / total
    metrics = {
        'loss': eval_loss / len(dataloader),
        'precision': np.mean(precision_list_eval),
        'recall': np.mean(recall_list_eval),
        'specificity': np.mean(specificity_list_eval),
        'accuracy': np.mean(accuracy_list_eval),
        'auroc': auroc_list_eval[-1] if auroc_list_eval else 0,
        'val_acc': val_acc
    }
    return metrics


# ===================================================================
# SECCIÓN 2: LÓGICA DE EJECUCIÓN PRINCIPAL
# (Todo este código solo se ejecuta si corres "python Train.py")
# ===================================================================
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='S4D-ECG Training')
    # Optimizer
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    parser.add_argument('--file_name', default='S4D', type=str, help='Folder Name for results')
    # Scheduler
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs') # Reducido de 200 a 100 para una prueba más rápida
    # Dataloader
    # ** SOLUCIÓN AUTOMÁTICA PARA WINDOWS **
    # Si el sistema es Windows, fuerza num_workers a 0 para evitar errores de multiprocessing
    default_workers = 0 if sys.platform == "win32" else 4
    parser.add_argument('--num_workers', default=default_workers, type=int, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    # Model
    parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
    parser.add_argument('--prenorm', action='store_true', help='Prenorm')
    # General
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()

    # --- Setup de Directorios y Configuración ---
    output_directory = './s4_results/' + args.file_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Directory '{output_directory}' created successfully.")
    else:
        print(f"Directory '{output_directory}' already exists.")

    output_filepath = f'{output_directory}/argparse_config.txt'
    with open(output_filepath, 'w') as file:
        for arg, value in vars(args).items():
            file.write(f'{arg}: {value}\n')
    print(f'Arguments saved to {output_filepath}')
    
    # --- Setup de Dispositivo y Variables Globales de Entrenamiento ---
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    best_acc = 0
    start_epoch = 0

    # --- Preparación de Datos ---
    print('==> Preparing data..')
    try:
        with h5py.File('x.hdf5', 'r') as f:
            X = f['tracings'][:]
        y = pd.read_csv('y.csv').values
        print(f"Datos cargados. Forma de X: {X.shape}, Forma de y: {y.shape}")
    except FileNotFoundError:
        print("\n[ERROR CRÍTICO] No se encontraron los archivos 'x.hdf5' y 'y.csv'.")
        print("Por favor, ejecuta primero el script 'preparar_datos.py'.")
        sys.exit(1) # Salir del script si no hay datos

    trainset = MyDataset(X[:-500], y[:-500])
    valset = MyDataset(X[-500:], y[-500:])
    testset = MyDataset(X[-500:], y[-500:])

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- Construcción del Modelo y Optimizador ---
    d_input = 12
    d_output = 8
    print('==> Building model..')
    model = S4Model(
        d_input=d_input,
        d_output=d_output,
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout,
        prenorm=args.prenorm,
        lr=args.lr
    )
    model = model.to(device)
    if device == 'cuda':
        cudnn.benchmark = True

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.BCELoss()
    optimizer, scheduler = setup_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs)
    
    # --- Bucle Principal de Entrenamiento y Evaluación ---
    train_metrics_history = []
    eval_metrics_history = []
    
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training Progress")
    for epoch in pbar:
        train_metrics = train(model, trainloader, device, criterion, optimizer, d_output)
        train_metrics_history.append(train_metrics)

        val_metrics = eval_model(model, valloader, device, criterion, d_output)
        eval_metrics_history.append(val_metrics)
        
        val_acc = val_metrics['val_acc']
        
        if val_acc > best_acc:
            print(f"New best accuracy: {val_acc:.3f}%. Saving checkpoint...")
            state = {'model': model.state_dict(), 'acc': val_acc, 'epoch': epoch}
            checkpoint_dir = './checkpoint/' + args.file_name
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(state, os.path.join(checkpoint_dir, 'ckpt_best.pth'))
            best_acc = val_acc
            
        scheduler.step()
        
        pbar.set_description(f'Epoch: {epoch} | Val Acc: {val_acc:.2f}% | Train Loss: {train_metrics["loss"]:.3f} | Val Loss: {val_metrics["loss"]:.3f}')

    # --- Guardado Final de Resultados ---
    print("\n==> Training Finished. Saving results...")
    
    # Guardar métricas de evaluación
    eval_csv_file = os.path.join(output_directory, 'evaluation_metrics.csv')
    with open(eval_csv_file, 'w', newline='') as file:
        if eval_metrics_history:
            writer = csv.DictWriter(file, fieldnames=eval_metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(eval_metrics_history)

    # Guardar métricas de entrenamiento
    train_csv_file = os.path.join(output_directory, 'train_metrics.csv')
    with open(train_csv_file, 'w', newline='') as file:
        if train_metrics_history:
            writer = csv.DictWriter(file, fieldnames=train_metrics_history[0].keys())
            writer.writeheader()
            writer.writerows(train_metrics_history)

    # Guardar modelo final
    torch.save(model.state_dict(), os.path.join(output_directory, 'model.pt'))

    print(f"Results and final model saved to '{output_directory}'.")
    print('Completed')