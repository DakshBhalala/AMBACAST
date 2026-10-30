"""Tests for RevIN and leakage-safe scaling."""

import torch
import numpy as np
import pytest
from compactts.data.transforms import RevIN, SafeStandardScaler


def test_revin_inversion_identity():
    torch.manual_seed(42)
    # 3D tensor: (batch=4, seq_len=48, channels=3)
    x = torch.randn(4, 48, 3) * 5.0 + 10.0
    revin = RevIN(num_features=3, affine=True)

    norm_x = revin(x, mode="norm")
    assert torch.allclose(norm_x.mean(dim=1), torch.zeros(4, 3), atol=1e-3)
    assert torch.allclose(norm_x.std(dim=1, unbiased=False), torch.ones(4, 3), atol=1e-2)

    denorm_x = revin(norm_x, mode="denorm")
    assert torch.allclose(x, denorm_x, atol=1e-3)


def test_revin_2d_support():
    torch.manual_seed(42)
    x = torch.randn(8, 36) * 3.0 + 7.0
    revin = RevIN(num_features=1, affine=False)

    norm_x = revin(x, mode="norm")
    assert norm_x.shape == (8, 36)
    denorm_x = revin(norm_x, mode="denorm")
    assert torch.allclose(x, denorm_x, atol=1e-3)


def test_safe_standard_scaler():
    train_data = np.array([[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]], dtype=np.float32)
    test_data = np.array([[15.0, 150.0], [25.0, 250.0]], dtype=np.float32)

    scaler = SafeStandardScaler()
    scaler.fit(train_data)

    norm_test = scaler.transform(test_data)
    inv_test = scaler.inverse_transform(norm_test)

    assert np.allclose(test_data, inv_test, atol=1e-5)
    # Check that scaler mean matches train mean, not test mean
    assert np.allclose(scaler.mean_, np.array([[20.0, 200.0]]))
