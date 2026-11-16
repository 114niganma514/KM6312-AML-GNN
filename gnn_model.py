# gnn models (GCN, GraphSAGE, GAT) and training loop

import os
import json
import time
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.nn import GCNConv, SAGEConv, GATConv
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


def set_seed(seed: int) -> None:
    """fix all random number generators so that GNN runs are reproducible"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if PYG_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class GCNModel(nn.Module):
    """2-layer Graph Convolutional Network for node classification"""

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 2, dropout: float = 0.3):
        super(GCNModel, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGEModel(nn.Module):
    """2-layer GraphSAGE model for node classification"""

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 2, dropout: float = 0.3):
        super(GraphSAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GATModel(nn.Module):
    """2-layer Graph Attention Network (GAT) for node classification"""

    def __init__(self, in_channels: int, hidden_channels: int = 64,
                 out_channels: int = 2, heads: int = 4, dropout: float = 0.3):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads,
                              dropout=dropout, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1,
                              dropout=dropout, concat=False)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGE3Layer(nn.Module):
    """3-layer GraphSAGE model for deeper graph representation learning"""

    def __init__(self, in_channels: int, hidden_channels: int = 128,
                 out_channels: int = 2, dropout: float = 0.3):
        super(GraphSAGE3Layer, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x


def get_model(model_name: str, in_channels: int, hidden_channels: int = 64,
              out_channels: int = 2, dropout: float = 0.3, heads: int = 4):
    """factory function to create a GNN model by name"""
    name = model_name.lower()
    if name == "gcn":
        return GCNModel(in_channels, hidden_channels, out_channels, dropout)
    elif name == "graphsage":
        return GraphSAGEModel(in_channels, hidden_channels, out_channels, dropout)
    elif name == "gat":
        return GATModel(in_channels, hidden_channels, out_channels, heads, dropout)
    elif name in ("graphsage3", "graphsage_3layer", "graphsage3layer"):
        return GraphSAGE3Layer(in_channels, hidden_channels, out_channels, dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}. Use 'GCN', 'GraphSAGE', 'GAT', or 'GraphSAGE3'.")


def train_gnn(model, data, class_weights: dict, epochs: int = 200,
              lr: float = 0.01, weight_decay: float = 5e-4,
              patience: int = 30, verbose: bool = True,
              use_lr_scheduler: bool = False) -> dict:
    """train a GNN model with early stopping"""
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric is not installed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    # class weights tensor
    weights = torch.tensor([class_weights.get(0, 1.0), class_weights.get(1, 1.0)],
                            dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # learning rate scheduler (optional)
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

    # training masks: only use labeled nodes (label != -1)
    labeled_mask = data.y != -1
    train_mask = data.train_mask & labeled_mask
    val_mask = data.val_mask & labeled_mask
    test_mask = data.test_mask & labeled_mask

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    if verbose:
        print(f"  Training on device: {device}")
        print(f"  Train nodes: {train_mask.sum().item()}, "
              f"Val nodes: {val_mask.sum().item()}, "
              f"Test nodes: {test_mask.sum().item()}")

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # validation
        model.eval()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            val_loss = criterion(out[val_mask], data.y[val_mask]).item()
            val_pred = out[val_mask].argmax(dim=1)
            val_acc = (val_pred == data.y[val_mask]).float().mean().item()

        history["train_loss"].append(loss.item())
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        # LR scheduler step
        if scheduler is not None:
            scheduler.step(val_loss)

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if verbose and (epoch % 20 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d}: train_loss={loss.item():.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if patience_counter >= patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
            break

    # load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"  Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch}")

    return {
        "model": model,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def evaluate_gnn(model, data, mask_name: str = "test") -> dict:
    """evaluate a trained GNN model on a specified split"""
    from sklearn.metrics import (precision_recall_curve, auc, f1_score,
                                   recall_score, precision_score, roc_auc_score,
                                   confusion_matrix)

    device = next(model.parameters()).device
    data = data.to(device)

    mask = getattr(data, f"{mask_name}_mask") & (data.y != -1)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()
        preds = out.argmax(dim=1).cpu().numpy()
        labels = data.y.cpu().numpy()

    y_true = labels[mask.cpu().numpy()]
    y_pred = preds[mask.cpu().numpy()]
    y_prob = probs[mask.cpu().numpy()]

    # PR-AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_curve, precision_curve)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0

    metrics = {
        "pr_auc": round(float(pr_auc), 4),
        "roc_auc": round(float(roc_auc), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    return metrics


def predict_unlabeled(model, data) -> np.ndarray:
    """predict labels for unlabeled nodes (transductive setting)"""
    device = next(model.parameters()).device
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = F.softmax(out, dim=1)[:, 1].cpu().numpy()

    return probs


def run_gnn_pipeline(preprocessed: dict, model_name: str = "GraphSAGE",
                      hidden_channels: int = 64, dropout: float = 0.3,
                      epochs: int = 200, lr: float = 0.01,
                      heads: int = 4, use_lr_scheduler: bool = False,
                      save_dir: str = None, seed: int = 42) -> dict:
    """run the full GNN training and evaluation pipeline"""
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric is not installed. "
                          "Install with: pip install torch torch-geometric")

    print("=" * 60)
    print(f"Training GNN: {model_name}")
    print("=" * 60)

    data = preprocessed["pyg_data"]
    class_weights = preprocessed["class_weights"]
    in_channels = data.x.shape[1]

    # create model (seeded for reproducibility)
    set_seed(seed)
    model = get_model(model_name, in_channels, hidden_channels, 2, dropout, heads)
    print(f"  Model: {model_name}, hidden={hidden_channels}, dropout={dropout}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # train
    train_result = train_gnn(model, data, class_weights, epochs=epochs, lr=lr,
                              use_lr_scheduler=use_lr_scheduler)

    # evaluate
    print(f"\n  Evaluation on test set:")
    test_metrics = evaluate_gnn(train_result["model"], data, "test")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"    {k:15s}: {v}")
    print(f"    confusion_matrix: {test_metrics['confusion_matrix']}")

    # count predictions on unlabeled nodes only
    all_probs = predict_unlabeled(train_result["model"], data)
    unlabeled_mask = (data.y == -1).cpu().numpy()
    unlabeled_probs = all_probs[unlabeled_mask]
    n_predicted_illicit = int((unlabeled_probs >= 0.5).sum())
    print(f"\n  Unlabeled nodes predicted as illicit (prob >= 0.5): "
          f"{n_predicted_illicit} of {int(unlabeled_mask.sum())}")

    # save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_dir = os.path.join(save_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        # save model
        torch.save(train_result["model"].state_dict(),
                   os.path.join(model_dir, f"{model_name.lower()}_best.pt"))

        # save results
        result_summary = {
            "model_name": model_name,
            "hidden_channels": hidden_channels,
            "dropout": dropout,
            "best_epoch": train_result["best_epoch"],
            "best_val_loss": train_result["best_val_loss"],
            "test_metrics": test_metrics,
            "n_unlabeled_predicted_illicit": int(n_predicted_illicit),
        }
        with open(os.path.join(save_dir, f"gnn_{model_name.lower()}_results.json"), "w") as f:
            json.dump(result_summary, f, indent=2)
        print("  Model and results saved")

    print("\n[GNN] Complete.")
    return {
        "model": train_result["model"],
        "test_metrics": test_metrics,
        "unlabeled_probs": unlabeled_probs,
        "history": train_result["history"],
    }


if __name__ == "__main__":
    from data_loader import load_all
    from preprocessing import preprocess_pipeline

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR
    RESULTS_DIR = os.path.join(BASE_DIR, "results")

    _, edgelist, _, merged = load_all(DATA_DIR)
    preprocessed = preprocess_pipeline(merged, edgelist, temporal=False)

    # train both GCN and GraphSAGE
    for model_name in ["GCN", "GraphSAGE"]:
        run_gnn_pipeline(preprocessed, model_name=model_name, save_dir=RESULTS_DIR)
