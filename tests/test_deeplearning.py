import torch
import torch.nn as nn
import numpy as np
import math
import pytest # For test structure, though not for running in this env

# Adjust path to import from reconlib
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reconlib.operators import NUFFTOperator # For type hinting and potential use
from reconlib.deeplearning.models.resnet_denoiser import SimpleResNetDenoiser
from reconlib.deeplearning.layers.data_consistency import DataConsistencyMoDL
from reconlib.deeplearning.models.modl_network import MoDLNet

# Mock NUFFTOperator for testing DL components without full NUFFT setup
class MockNUFFTOperator:
    def __init__(self, image_shape, k_traj_len, device_):
        self.image_shape = image_shape
        self.dim = len(image_shape)
        # k_trajectory must be (num_points, dim)
        self.k_trajectory = torch.zeros(k_traj_len, self.dim, device=device_) 
        self.device = device_

    def op(self, x: torch.Tensor) -> torch.Tensor: # A
        # Forward: image -> k-space
        # Dummy op: returns tensor of shape (k_traj_len,)
        # Ensure it's on the same device and complex type as input if x is complex
        return torch.ones(self.k_trajectory.shape[0], dtype=x.dtype if x.is_complex() else torch.complex64, device=x.device) * torch.mean(torch.abs(x))

    def op_adj(self, y: torch.Tensor) -> torch.Tensor: # A_H
        # Adjoint: k-space -> image
        # Dummy op_adj: returns tensor of image_shape
        return torch.ones(self.image_shape, dtype=y.dtype if y.is_complex() else torch.complex64, device=y.device) * torch.mean(torch.abs(y))

@pytest.fixture
def device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@pytest.fixture
def setup_2d_data(device):
    img_s = 32 # Smaller for tests
    ishape = (img_s, img_s)
    k_pts = 100
    nufft_op = MockNUFFTOperator(image_shape=ishape, k_traj_len=k_pts, device_=device)
    return {
        'image_shape': ishape,
        'k_space_points': k_pts,
        'nufft_op': nufft_op,
        'denoiser_channels': 1,
        'device': device
    }

@pytest.fixture
def setup_3d_data(device):
    img_s = 16 # Smaller for tests
    ishape = (img_s, img_s, img_s)
    k_pts = 200
    nufft_op = MockNUFFTOperator(image_shape=ishape, k_traj_len=k_pts, device_=device)
    return {
        'image_shape': ishape,
        'k_space_points': k_pts,
        'nufft_op': nufft_op,
        'denoiser_channels': 1, # Assuming magnitude processing for simplicity with current ResNet
        'device': device
    }

# --- Test SimpleResNetDenoiser ---
def test_resnet_denoiser_instantiation(setup_2d_data):
    print("Test: SimpleResNetDenoiser Instantiation")
    denoiser = SimpleResNetDenoiser(
        in_channels=setup_2d_data['denoiser_channels'],
        out_channels=setup_2d_data['denoiser_channels'],
        num_internal_channels=16,
        num_blocks=1
    ).to(setup_2d_data['device'])
    assert isinstance(denoiser, nn.Module)
    print("  Passed.")

def test_resnet_denoiser_forward_2d(setup_2d_data):
    print("Test: SimpleResNetDenoiser 2D Forward Pass")
    denoiser = SimpleResNetDenoiser(
        in_channels=setup_2d_data['denoiser_channels'],
        out_channels=setup_2d_data['denoiser_channels'],
        num_internal_channels=16,
        num_blocks=1
    ).to(setup_2d_data['device'])
    
    bs = 2
    dummy_image = torch.randn(bs, setup_2d_data['denoiser_channels'], *setup_2d_data['image_shape'], device=setup_2d_data['device'])
    output = denoiser(dummy_image)
    assert output.shape == dummy_image.shape
    assert not torch.isnan(output).any() and not torch.isinf(output).any()
    assert torch.norm(output - dummy_image) > 1e-3 # Ensure it's not an identity function
    print(f"  Input shape: {dummy_image.shape}, Output shape: {output.shape}. Passed.")

# --- Test DataConsistencyMoDL ---
def test_dc_modl_instantiation(setup_2d_data):
    print("Test: DataConsistencyMoDL Instantiation")
    dc_layer = DataConsistencyMoDL(
        nufft_op=setup_2d_data['nufft_op'],
        lambda_dc=0.05,
        num_cg_iterations=2
    ).to(setup_2d_data['device'])
    assert isinstance(dc_layer, nn.Module)
    print("  Passed.")

def test_dc_modl_forward_2d(setup_2d_data):
    print("Test: DataConsistencyMoDL 2D Forward Pass & CG Behavior")
    nufft_op = setup_2d_data['nufft_op']
    dc_layer = DataConsistencyMoDL(
        nufft_op=nufft_op,
        lambda_dc=0.05,
        num_cg_iterations=3
    ).to(setup_2d_data['device'])
    
    # zk: current image estimate from denoiser
    zk = torch.randn(setup_2d_data['image_shape'], dtype=torch.complex64, device=setup_2d_data['device'])
    # y: observed k-space data
    y_obs = torch.randn(setup_2d_data['k_space_points'], dtype=torch.complex64, device=setup_2d_data['device'])
    
    output_image = dc_layer(zk, y_obs)
    assert output_image.shape == zk.shape
    assert not torch.isnan(output_image).any() and not torch.isinf(output_image).any()
    print(f"  Input zk shape: {zk.shape}, y_obs shape: {y_obs.shape}, Output shape: {output_image.shape}. Passed basic checks.")

    # Simple CG behavior check: if rhs (b) is zero, output should be close to zero
    # The (A^H A + lambda I)x = b system with b=0 should yield x=0 because (A^H A + lambda I) is positive definite for lambda > 0.
    # Let's test if the residual is reduced by CG for a non-zero RHS
    rhs_b_test = nufft_op.op_adj(y_obs) + dc_layer.lambda_dc * zk # A non-zero RHS
    x_initial_cg = torch.zeros_like(rhs_b_test) # Standard CG starts with x=0
    
    # Calculate initial residual for (A^H A + lambda I)x = rhs_b_test, with x = x_initial_cg
    initial_residual = rhs_b_test - dc_layer.operator_AHA_plus_lambda_I(x_initial_cg)
    initial_residual_norm = torch.norm(initial_residual)
    
    x_after_cg = dc_layer._cg_solve(rhs_b_test, max_iter=dc_layer.num_cg_iterations)
    
    # Calculate final residual for (A^H A + lambda I)x = rhs_b_test, with x = x_after_cg
    final_residual = rhs_b_test - dc_layer.operator_AHA_plus_lambda_I(x_after_cg)
    final_residual_norm = torch.norm(final_residual)
    
    # Residual norm should decrease or stay very small if already converged
    assert final_residual_norm < initial_residual_norm or torch.isclose(final_residual_norm, initial_residual_norm, atol=1e-5) 
    print(f"  CG residual norm check: Initial: {initial_residual_norm:.2e}, Final: {final_residual_norm:.2e}. Passed.")


# --- Test MoDLNet ---
def test_modl_net_instantiation(setup_2d_data):
    print("Test: MoDLNet Instantiation")
    denoiser = SimpleResNetDenoiser(
        in_channels=setup_2d_data['denoiser_channels'], 
        out_channels=setup_2d_data['denoiser_channels'],
        num_blocks=1).to(setup_2d_data['device'])
    modl_net = MoDLNet(
        nufft_op=setup_2d_data['nufft_op'],
        denoiser_cnn=denoiser,
        num_iterations=2,
        lambda_dc_initial=0.05,
        num_cg_iterations_dc=2
    ).to(setup_2d_data['device'])
    assert isinstance(modl_net, nn.Module)
    print("  Passed.")

def test_modl_net_forward_2d(setup_2d_data):
    print("Test: MoDLNet 2D Forward Pass")
    denoiser = SimpleResNetDenoiser(
        in_channels=setup_2d_data['denoiser_channels'], 
        out_channels=setup_2d_data['denoiser_channels'],
        num_blocks=1).to(setup_2d_data['device'])
    modl_net = MoDLNet(
        nufft_op=setup_2d_data['nufft_op'],
        denoiser_cnn=denoiser,
        num_iterations=2,
        lambda_dc_initial=0.05,
        num_cg_iterations_dc=2
    ).to(setup_2d_data['device'])
    
    y_obs = torch.randn(setup_2d_data['k_space_points'], dtype=torch.complex64, device=setup_2d_data['device'])
    x0 = torch.randn(setup_2d_data['image_shape'], dtype=torch.complex64, device=setup_2d_data['device'])
    
    reconstructed_image = modl_net(y_obs, x0)
    assert reconstructed_image.shape == setup_2d_data['image_shape']
    assert not torch.isnan(reconstructed_image).any() and not torch.isinf(reconstructed_image).any()
    # Check if output is different from input x0 (it should be after denoiser and DC)
    assert torch.norm(reconstructed_image - x0) > 1e-3 if torch.norm(x0) > 1e-9 else True
    print(f"  Input y_obs shape: {y_obs.shape}, x0 shape: {x0.shape}, Output shape: {reconstructed_image.shape}. Passed.")

    # Test with x0 = None
    reconstructed_image_no_x0 = modl_net(y_obs, None)
    assert reconstructed_image_no_x0.shape == setup_2d_data['image_shape']
    assert not torch.isnan(reconstructed_image_no_x0).any() and not torch.isinf(reconstructed_image_no_x0).any()
    print(f"  Forward pass with x0=None also successful. Output shape: {reconstructed_image_no_x0.shape}. Passed.")


# It's harder to make a simple 3D denoiser and test it without Conv3d, etc.
# For now, focus tests on 2D setup where SimpleResNetDenoiser is appropriate.
# A 3D test would be structured similarly if a 3D denoiser was available.

if __name__ == '__main__':
    # This allows running the tests directly, e.g. python tests/test_deeplearning.py
    # Pytest would discover and run functions starting with "test_"
    print("Starting DL component tests...")
    current_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {current_device}")

    # Manually create data for non-pytest environment
    data_2d_fixture = {
        'image_shape': (32,32),
        'k_space_points': 100,
        'nufft_op': MockNUFFTOperator(image_shape=(32,32), k_traj_len=100, device_=current_device),
        'denoiser_channels': 1,
        'device': current_device
    }
    
    test_resnet_denoiser_instantiation(data_2d_fixture)
    test_resnet_denoiser_forward_2d(data_2d_fixture)
    
    test_dc_modl_instantiation(data_2d_fixture)
    test_dc_modl_forward_2d(data_2d_fixture) # Includes CG behavior check
    
    test_modl_net_instantiation(data_2d_fixture)
    test_modl_net_forward_2d(data_2d_fixture)
    
    # Add calls to 3D tests here if/when a 3D denoiser is implemented
    # data_3d_fixture = setup_3d_data(current_device) # if setup_3d_data was not a fixture
    # ... (call 3D versions of tests) ...
    
    print("\nDL component tests completed.")
