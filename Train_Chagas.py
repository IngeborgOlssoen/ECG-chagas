'''
Train_Chagas.py
Script de entrenamiento final para la clasificación binaria de la enfermedad de Chagas.
- Versión final para datos balanceados.
'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import sys
import argparse
import h5py
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm
import csv

from src.models.s4.s4d import S4D

# ===================================================================
# SECCIÓN 1: DEFINICIONES DE CLASES Y FUNCIONES (Sin cambios)
# ===================================================================

base_torch_version = torch.__version__.split('+')[0]
if tuple(map(int, base_torch_version.split('.')))[:2] >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __getitem__(self, index):
        x = self.X[index].astype(np.float32)
        label = self.y[index].astype(np.float32)
        return torch.tensor(x), torch.tensor(label)
    def __len__(self):
        return len(self.X)

class S4Model(nn.Module):
    def __init__(self, d_input, d_output=1, d_model=256, n_layers=4, dropout=0.2, prenorm=False, lr=0.001):
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
        return torch.sigmoid(x)

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
    keys = sorted(set([k for hp in hps for k in hp.keys()]))
    print("==> Optimizer Config:")
    for i, g in enumerate(optimizer.param_groups):
        group_hps = {k: g.get(k, None) for k in keys}
        print(' | '.join([f"Optimizer group {i}", f"{len(g['params'])} tensors"] + [f"{k} {v}" for k, v in group_hps.items()]))
    return optimizer, scheduler

def calculate_metrics(model, dataloader, device, criterion, optimizer=None):
    is_training = optimizer is not None
    model.train(is_training)
    
    all_targets, all_outputs, total_loss = [], [], 0
    
    context = torch.enable_grad() if is_training else torch.no_grad()
    with context:
        pbar_desc = "Training Batch" if is_training else "Evaluating Batch"
        pbar = tqdm(dataloader, desc=pbar_desc, leave=False, total=len(dataloader))
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            if is_training:
                optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().detach().numpy())

    all_targets = np.concatenate(all_targets)
    all_outputs = np.concatenate(all_outputs)
    
    auroc = roc_auc_score(all_targets, all_outputs) if len(np.unique(all_targets)) > 1 else 0.5
        
    return {'loss': total_loss / len(dataloader), 'auroc': auroc}

# ===================================================================
# SECCIÓN 2: LÓGICA DE EJECUCIÓN PRINCIPAL
# ===================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='S4D-ECG Training for Chagas Disease (Binary)')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='Weight decay')
    parser.add_argument('--file_name', default='S4D_Chagas_Final', type=str, help='Folder Name for results')
    parser.add_argument('--epochs', default=100, type=int, help='Training epochs')
    default_workers = 0 # Forzamos 0 para máxima compatibilidad en Windows/Linux con Multiprocessing
    parser.add_argument('--num_workers', default=default_workers, type=int, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
    parser.add_argument('--n_layers', default=4, type=int, help='Number of layers')
    parser.add_argument('--d_model', default=128, type=int, help='Model dimension')
    parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')
    parser.add_argument('--prenorm', action='store_true', help='Prenorm')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    
    args = parser.parse_args()

    output_directory = os.path.join('./s4_results/', args.file_name)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Results will be saved to '{output_directory}'")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    best_auroc = 0.0
    start_epoch = 0

    print('==> Preparing data for Chagas classification..')
    try:
        with h5py.File('x_chagas.hdf5', 'r') as f:
            X = f['tracings'][:]
        y = pd.read_csv('y_chagas.csv').values
        print(f"Datos cargados. Forma de X: {X.shape}, Forma de y: {y.shape}")
    except FileNotFoundError:
        print("\n[ERROR CRÍTICO] No se encontraron 'x_chagas.hdf5' y/o 'y_chagas.csv'.")
        print("Por favor, ejecuta primero el script 'crear_hdf5_chagas.py'.")
        sys.exit(1)

    print("==> Performing stratified train-validation split...")
    indices = np.arange(X.shape[0])
    train_indices, val_indices, y_train, y_val = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val = X[train_indices], X[val_indices]
    trainset = MyDataset(X_train, y_train)
    valset = MyDataset(X_val, y_val)
    print(f"Training set: {len(trainset)} samples | Validation set: {len(valset)} samples.")
    print(f"Chagas cases in training set: {int(np.sum(y_train))} | Chagas cases in validation set: {int(np.sum(y_val))}")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    d_input = 12
    d_output = 1
    print('==> Building binary classification model..')
    model = S4Model(d_input=d_input, d_output=d_output, d_model=args.d_model, n_layers=args.n_layers, dropout=args.dropout, prenorm=args.prenorm, lr=args.lr)
    model = model.to(device)
    if device == 'cuda': cudnn.benchmark = True

    # ===================================================================
    # === CAMBIO CLAVE: Usamos la función de pérdida estándar ===
    # La lógica de pesos se elimina porque los datos ya están balanceados.
    # ===================================================================
    criterion = nn.BCELoss()
    
    optimizer, scheduler = setup_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, epochs=args.epochs)
    
    train_metrics_history, eval_metrics_history = [], []
    
    pbar = tqdm(range(start_epoch, args.epochs), desc="Training Progress")
    for epoch in pbar:
        train_metrics = calculate_metrics(model, trainloader, device, criterion, optimizer)
        train_metrics_history.append(train_metrics)
        
        val_metrics = calculate_metrics(model, valloader, device, criterion)
        eval_metrics_history.append(val_metrics)
        
        val_auroc = val_metrics['auroc']
        
        if val_auroc > best_auroc:
            print(f"\nNew best validation AUROC: {val_auroc:.4f}. Saving checkpoint...")
            state = {'model': model.state_dict(), 'auroc': val_auroc, 'epoch': epoch}
            checkpoint_dir = os.path.join('./checkpoint/', args.file_name)
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(state, os.path.join(checkpoint_dir, 'ckpt_best_auroc.pth'))
            best_auroc = val_auroc
            
        scheduler.step()
        pbar.set_description(f'Epoch: {epoch+1}/{args.epochs} | Val AUROC: {val_auroc:.4f} | Best AUROC: {best_auroc:.4f} | Train AUROC: {train_metrics["auroc"]:.4f}')

    print("\n==> Training Finished. Saving final model and results...")
    
    pd.DataFrame(eval_metrics_history).to_csv(os.path.join(output_directory, 'evaluation_metrics.csv'), index=False)
    pd.DataFrame(train_metrics_history).to_csv(os.path.join(output_directory, 'train_metrics.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(output_directory, 'model_final.pt'))

    print(f"Results and final model saved to '{output_directory}'.")
    print('Completed')