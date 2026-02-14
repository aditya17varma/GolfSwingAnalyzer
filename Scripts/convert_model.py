"""
Convert SwingNet (MobileNetV2 + BiLSTM) from PyTorch to CoreML.

Splits into two models:
  1. SwingNetCNN.mlpackage  — single frame -> 1280-dim feature vector
  2. SwingNetLSTM.mlpackage — feature sequence -> event logits

Usage:
  cd submodule/GolfDB
  python ../../Scripts/convert_model.py

Requires: torch, coremltools, numpy
"""

import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add submodule/ to path so `from GolfDB.MobileNetV2 import ...` works inside model.py
# Also add submodule/GolfDB/ so we can do `from model import EventDetector` directly
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'submodule'))       # for GolfDB.MobileNetV2
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'submodule', 'GolfDB', 'golfdb'))  # for model, MobileNetV2
from model import EventDetector
from MobileNetV2 import MobileNetV2

DEFAULT_BEST_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'submodule', 'GolfDB', 'golfdb', 'models', 'swingnet_best.pth.tar')
DEFAULT_LEGACY_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'submodule', 'GolfDB', 'golfdb', 'models', 'swingnet_1800.pth.tar')
MOBILENET_WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'submodule', 'GolfDB', 'golfdb', 'mobilenet_v2.pth.tar')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'Scripts', 'output')


def resolve_weights_path():
    override = os.getenv('GOLFDB_CONVERT_CKPT')
    if override:
        return override
    if os.path.exists(DEFAULT_BEST_WEIGHTS_PATH):
        return DEFAULT_BEST_WEIGHTS_PATH
    return DEFAULT_LEGACY_WEIGHTS_PATH


class CNNFeatureExtractor(nn.Module):
    """MobileNetV2 feature backbone from SwingNet. Single frame in, 1280-dim vector out."""
    def __init__(self, cnn_layers):
        super().__init__()
        self.cnn = cnn_layers

    def forward(self, x):
        # x: (1, 3, 160, 160)
        c_out = self.cnn(x)        # (1, 1280, 5, 5)
        c_out = c_out.mean(3).mean(2)  # global avg pool -> (1, 1280)
        return c_out


class SequenceClassifier(nn.Module):
    """BiLSTM + Linear head from SwingNet. Feature sequence in, event logits out."""
    def __init__(self, rnn, lin):
        super().__init__()
        self.rnn = rnn
        self.lin = lin

    def forward(self, features):
        # features: (1, T, 1280)
        h0 = torch.zeros(2, 1, 256)  # 2 directions * 1 layer
        c0 = torch.zeros(2, 1, 256)
        r_out, _ = self.rnn(features, (h0, c0))  # (1, T, 512)
        out = self.lin(r_out)  # (1, T, 9)
        return out


def load_swingnet():
    """Load the full SwingNet model with trained weights."""
    # EventDetector.__init__ calls torch.load('mobilenet_v2.pth.tar') from CWD when pretrain=True.
    # We chdir to the GolfDB directory so it can find the weights file.
    original_dir = os.getcwd()
    os.chdir(os.path.join(PROJECT_ROOT, 'submodule', 'GolfDB', 'golfdb'))

    model = EventDetector(
        pretrain=True,
        width_mult=1.,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False
    )

    os.chdir(original_dir)

    weights_path = resolve_weights_path()
    try:
        save_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
    except TypeError:
        # Backward compatibility for PyTorch versions without weights_only.
        save_dict = torch.load(weights_path, map_location='cpu')
    if 'model_state_dict' not in save_dict:
        raise KeyError("Checkpoint missing 'model_state_dict': {}".format(weights_path))
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint: {weights_path}")
    return model


def validate_pytorch(model):
    """Run a quick forward pass to verify the model works."""
    dummy = torch.randn(1, 64, 3, 160, 160)
    with torch.no_grad():
        out = model(dummy)
    print(f"PyTorch output shape: {out.shape}")  # expect (64, 9)
    probs = F.softmax(out, dim=1)
    print(f"Sample probs (frame 0): {probs[0].numpy().round(3)}")
    return out


def convert_cnn(model):
    """Convert the CNN feature extractor to CoreML."""
    import coremltools as ct

    cnn_model = CNNFeatureExtractor(model.cnn)
    cnn_model.eval()

    dummy_input = torch.randn(1, 3, 160, 160)
    traced = torch.jit.trace(cnn_model, dummy_input)

    # Validate trace
    with torch.no_grad():
        original_out = cnn_model(dummy_input)
        traced_out = traced(dummy_input)
    diff = (original_out - traced_out).abs().max().item()
    print(f"CNN trace max diff: {diff:.2e}")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="frame", shape=(1, 3, 160, 160))],
        outputs=[ct.TensorType(name="features")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram"
    )

    output_path = os.path.join(OUTPUT_DIR, "SwingNetCNN.mlpackage")
    mlmodel.save(output_path)
    print(f"Saved CNN model to {output_path}")
    return mlmodel


def convert_lstm(model):
    """Convert the LSTM sequence classifier to CoreML."""
    import coremltools as ct

    seq_model = SequenceClassifier(model.rnn, model.lin)
    seq_model.eval()

    dummy_input = torch.randn(1, 64, 1280)
    traced = torch.jit.trace(seq_model, dummy_input)

    # Validate trace
    with torch.no_grad():
        original_out = seq_model(dummy_input)
        traced_out = traced(dummy_input)
    diff = (original_out - traced_out).abs().max().item()
    print(f"LSTM trace max diff: {diff:.2e}")

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="features", shape=(1, ct.RangeDim(1, 512, 64), 1280))],
        outputs=[ct.TensorType(name="logits")],
        minimum_deployment_target=ct.target.iOS17,
        convert_to="mlprogram"
    )

    output_path = os.path.join(OUTPUT_DIR, "SwingNetLSTM.mlpackage")
    mlmodel.save(output_path)
    print(f"Saved LSTM model to {output_path}")
    return mlmodel


def validate_coreml(pytorch_model, cnn_mlmodel, lstm_mlmodel):
    """Compare CoreML output to PyTorch output for correctness."""
    import coremltools as ct

    dummy = torch.randn(1, 64, 3, 160, 160)

    # PyTorch full forward pass
    with torch.no_grad():
        pytorch_out = pytorch_model(dummy)
    pytorch_probs = F.softmax(pytorch_out, dim=1).numpy()

    # CoreML split forward pass
    cnn_features = []
    for i in range(64):
        frame = dummy[0, i:i+1].numpy()  # (1, 3, 160, 160)
        cnn_pred = cnn_mlmodel.predict({"frame": frame})
        cnn_features.append(cnn_pred["features"])

    feature_seq = np.stack(cnn_features, axis=1)  # (1, 64, 1280)
    lstm_pred = lstm_mlmodel.predict({"features": feature_seq.astype(np.float32)})
    coreml_logits = lstm_pred["logits"]  # (1, 64, 9)
    coreml_logits = coreml_logits.reshape(-1, 9)

    # Compare
    pytorch_events = np.argmax(pytorch_probs, axis=0)[:-1]
    coreml_probs = np.exp(coreml_logits) / np.exp(coreml_logits).sum(axis=1, keepdims=True)
    coreml_events = np.argmax(coreml_probs, axis=0)[:-1]

    print(f"\nPyTorch events:  {pytorch_events}")
    print(f"CoreML events:   {coreml_events}")
    print(f"Events match: {np.array_equal(pytorch_events, coreml_events)}")

    max_prob_diff = np.abs(pytorch_probs - coreml_probs).max()
    print(f"Max probability diff: {max_prob_diff:.4f}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading SwingNet model...")
    model = load_swingnet()

    print("\nValidating PyTorch model...")
    validate_pytorch(model)

    print("\n--- Converting CNN Feature Extractor ---")
    cnn_mlmodel = convert_cnn(model)

    print("\n--- Converting LSTM Sequence Classifier ---")
    lstm_mlmodel = convert_lstm(model)

    print("\n--- Validating CoreML vs PyTorch ---")
    validate_coreml(model, cnn_mlmodel, lstm_mlmodel)

    print("\nDone! Models saved to Scripts/output/")


if __name__ == "__main__":
    main()
