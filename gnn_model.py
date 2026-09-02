"""
Graph Neural Network Models for Illicit Transaction Detection.

This module implements two GNN architectures for node classification
on the Elliptic Bitcoin transaction graph:

  - GCN (Graph Convolutional Network): Kipf & Welling (2017)
  - GraphSAGE (Graph SAmple and aggreGatE): Hamilton et al. (2017)

Both models are implemented using PyTorch Geometric and trained with
class-weighted cross-entropy loss to handle severe class imbalance.

The module includes:
  - Model definitions
  - Training loop with early stopping
  - Evaluation on validation and test sets
  - Prediction for unlabeled nodes (transductive setting)
"""

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


class GCNModel(nn.Module):
    """2-layer Graph Convolutional Network for node classification.

    Args:
        in_channels: Number of input features (166 for Elliptic).
        hidden_channels: Number of hidden layer units.
        out_channels: Number of output classes (2: licit/illicit).
        dropout: Dropout probability.
    """

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
    """2-layer GraphSAGE model for node classification.

    GraphSAGE uses neighbor sampling and aggregation, making it suitable
    for inductive learning on large graphs.

    Args:
        in_channels: Number of input features (166 for Elliptic).
        hidden_channels: Number of hidden layer units.
        out_channels: Number of output classes (2: licit/illicit).
        dropout: Dropout probability.
    """

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
    """2-layer Graph Attention Network (GAT) for node classification.

    GAT uses masked self-attention on graph-structured data, allowing
    the model to assign different importance weights to different neighbors.

    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden layer units per attention head.
        out_channels: Number of output classes.
        heads: Number of multi-head attention heads in first layer.
        dropout: Dropout probability (also applied to attention weights).
    """

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
    """3-layer GraphSAGE model for deeper graph representation learning.

    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden layer units.
        out_channels: Number of output classes.
        dropout: Dropout probability.
    """

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
    """Factory function to create a GNN model by name.

    Args:
        model_name: 'GCN', 'GraphSAGE', 'GAT', or 'GraphSAGE3'.
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output classes.
        dropout: Dropout probability.
        heads: Number of attention heads (for GAT only).

    Returns:
        PyTorch model instance.
    """
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
    """Train a GNN model with early stopping.

    Args:
        model: PyTorch GNN model.
        data: PyG Data object with x, edge_index, y, train_mask, val_mask, test_mask.
        class_weights: Dictionary mapping class index to weight.
        epochs: Maximum number of training epochs.
        lr: Learning rate.
        weight_decay: L2 regularization coefficient.
        patience: Early stopping patience (epochs without val improvement).
        verbose: Whether to print training progress.
        use_lr_scheduler: Whether to use ReduceLROnPlateau learning rate scheduler.

    Returns:
        Dictionary with training history and best model state.
    """
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric is not installed.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    data = data.to(device)

    # Class weights tensor
    weights = torch.tensor([class_weights.get(0, 1.0), class_weights.get(1, 1.0)],
                            dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler (optional)
    scheduler = None
    if use_lr_scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

    # Training masks: only use labeled nodes (label != -1)
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
        # Training
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Validation
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

        # Early stopping
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

    # Load best model
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
    """Evaluate a trained GNN model on a specified split.

    Args:
        model: Trained PyTorch GNN model.
        data: PyG Data object.
        mask_name: 'train', 'val', or 'test'.

    Returns:
        Dictionary with evaluation metrics and predictions.
    """
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
    """Predict labels for unlabeled nodes (transductive setting).

    Args:
        model: Trained PyTorch GNN model.
        data: PyG Data object.

    Returns:
        Array of predicted probabilities for the illicit class,
        for all nodes (including unlabeled).
    """
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
                      save_dir: str = None) -> dict:
    """Run the full GNN training and evaluation pipeline.

    Args:
        preprocessed: Dictionary from preprocessing.preprocess_pipeline().
        model_name: 'GCN', 'GraphSAGE', 'GAT', or 'GraphSAGE3'.
        hidden_channels: Number of hidden units.
        dropout: Dropout probability.
        epochs: Maximum training epochs.
        lr: Learning rate.
        heads: Number of attention heads (for GAT only).
        use_lr_scheduler: Whether to use ReduceLROnPlateau scheduler.
        save_dir: Optional directory to save results and model.

    Returns:
        Dictionary with trained model, metrics, and predictions.
    """
    if not PYG_AVAILABLE:
        raise ImportError("PyTorch Geometric is not installed. "
                          "Install with: pip install torch torch-geometric")

    print("=" * 60)
    print(f"Training GNN: {model_name}")
    print("=" * 60)

    data = preprocessed["pyg_data"]
    class_weights = preprocessed["class_weights"]
    in_channels = data.x.shape[1]

    # Create model
    model = get_model(model_name, in_channels, hidden_channels, 2, dropout, heads)
    print(f"  Model: {model_name}, hidden={hidden_channels}, dropout={dropout}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    train_result = train_gnn(model, data, class_weights, epochs=epochs, lr=lr,
                              use_lr_scheduler=use_lr_scheduler)

    # Evaluate
    print(f"\n  Evaluation on test set:")
    test_metrics = evaluate_gnn(train_result["model"], data, "test")
    for k, v in test_metrics.items():
        if k != "confusion_matrix":
            print(f"    {k:15s}: {v}")
    print(f"    confusion_matrix: {test_metrics['confusion_matrix']}")

    # Predict unlabeled nodes
    unlabeled_probs = predict_unlabeled(train_result["model"], data)
    n_predicted_illicit = (unlabeled_probs >= 0.5).sum()
    print(f"\n  Unlabeled nodes predicted as illicit (prob >= 0.5): {n_predicted_illicit}")

    # Save
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_dir = os.path.join(save_dir, "models")
        os.makedirs(model_dir, exist_ok=True)

        # Save model
        torch.save(train_result["model"].state_dict(),
                   os.path.join(model_dir, f"{model_name.lower()}_best.pt"))

        # Save results
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
        print(f"  Model and results saved to {save_dir}")

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

    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    DATA_DIR = BASE_DIR
    RESULTS_DIR = BASE_DIR

    _, edgelist, _, merged = load_all(DATA_DIR)
    preprocessed = preprocess_pipeline(merged, edgelist, temporal=False)

    # Train both GCN and GraphSAGE
    for model_name in ["GCN", "GraphSAGE"]:
        run_gnn_pipeline(preprocessed, model_name=model_name, save_dir=RESULTS_DIR)
