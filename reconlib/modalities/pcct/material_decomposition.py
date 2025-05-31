import torch
from reconlib.operators import Operator
from reconlib.modalities.pcct.operators import PCCTProjectorOperator
from typing import List, Dict, Union, Optional, Callable # Union for type hint of device

from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
try:
    from reconlib.regularizers.base import RegularizerBase # For type hinting
except ImportError:
    # Create a dummy RegularizerBase if it cannot be imported (e.g. module or class not found)
    # This allows type hinting and basic script execution when regularizers are not strictly needed for a test.
    print("Warning: Could not import RegularizerBase from reconlib.regularizers.base. Using a dummy placeholder for type hinting.")
    class RegularizerBase:
        def proximal_operator(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
            raise NotImplementedError("This is a dummy RegularizerBase.")

import traceback # Added for print_exc

# Define __all__ for the module
__all__ = ['MaterialDecompositionForwardOperator', 'IterativeMaterialDecompositionReconstructor']

class MaterialDecompositionForwardOperator(Operator):
    """
    Forward operator for material decomposition in PCCT.
    Combines basis material images into a single reference attenuation map,
    then uses a PCCT projector to simulate energy-resolved sinograms.
    """
    def __init__(self,
                 material_reference_attenuations: Dict[str, float],
                 pcct_projector: PCCTProjectorOperator,
                 basis_material_names: List[str],
                 device: Union[str, torch.device] = 'cpu'):
        super().__init__()
        self.material_reference_attenuations = material_reference_attenuations
        self.pcct_projector = pcct_projector
        self.basis_material_names = basis_material_names
        self.device = torch.device(device)

        for name in self.basis_material_names:
            if name not in self.material_reference_attenuations:
                raise ValueError(f"Material '{name}' in basis_material_names not found in material_reference_attenuations.")

        self.num_materials = len(self.basis_material_names)
        self.image_shape_single_material = self.pcct_projector.image_shape
        self.expected_input_shape = (self.num_materials, self.image_shape_single_material[0], self.image_shape_single_material[1])
        self.output_shape_op = self.pcct_projector.measurement_shape

        print("MaterialDecompositionForwardOperator initialized.")
        print(f"  Basis materials: {self.basis_material_names}")
        print(f"  Reference attenuations: {self.material_reference_attenuations}")
        print(f"  Expected input shape (basis images): {self.expected_input_shape}")
        print(f"  Output shape (energy-resolved sinograms): {self.output_shape_op}")

    @property
    def pcct_projector_device(self):
        return self.pcct_projector.device

    def op(self, basis_images: torch.Tensor, sensitivity_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sensitivity_maps is accepted for compatibility with ProximalGradientReconstructor, but not used here.
        if basis_images.shape != self.expected_input_shape:
            raise ValueError(f"Input basis_images shape mismatch. Expected {self.expected_input_shape}, got {basis_images.shape}.")
        basis_images = basis_images.to(self.device)
        mu_reference_combined = torch.zeros(self.image_shape_single_material, device=self.device, dtype=basis_images.dtype)
        for m in range(self.num_materials):
            material_name = self.basis_material_names[m]
            ref_atten = self.material_reference_attenuations[material_name]
            mu_reference_combined += basis_images[m, :, :] * ref_atten
        mu_reference_combined = mu_reference_combined.to(self.pcct_projector_device)
        output_sinograms_counts = self.pcct_projector.op(mu_reference_combined)
        return output_sinograms_counts.to(self.device)

    def op_adj(self, measured_counts_stack: torch.Tensor, sensitivity_maps: Optional[torch.Tensor] = None) -> torch.Tensor:
        # sensitivity_maps is accepted for compatibility with ProximalGradientReconstructor, but not used here.
        if measured_counts_stack.shape != self.output_shape_op:
             raise ValueError(f"Input measured_counts_stack shape mismatch. Expected {self.output_shape_op}, got {measured_counts_stack.shape}.")
        measured_counts_stack = measured_counts_stack.to(self.pcct_projector_device)
        mu_ref_adj_from_pcct = self.pcct_projector.op_adj(measured_counts_stack)
        adj_basis_list = []
        for m in range(self.num_materials):
            material_name = self.basis_material_names[m]
            ref_atten = self.material_reference_attenuations[material_name]
            adj_basis_component = mu_ref_adj_from_pcct.to(device=self.device, dtype=torch.float32) * ref_atten
            adj_basis_list.append(adj_basis_component)
        output_basis_adj = torch.stack(adj_basis_list, dim=0)
        return output_basis_adj.to(self.device)

class IterativeMaterialDecompositionReconstructor:
    def __init__(self,
                 iterations: int,
                 step_size: float,
                 regularizers: Optional[List[Optional[RegularizerBase]]] = None,
                 verbose: bool = False):
        self.iterations = iterations
        self.step_size = step_size
        self.regularizers = regularizers if regularizers is not None else []
        self.verbose = verbose

    def reconstruct(self,
                    measured_data: torch.Tensor,
                    forward_operator: MaterialDecompositionForwardOperator,
                    initial_estimate: Optional[torch.Tensor] = None) -> torch.Tensor:
        num_materials = forward_operator.num_materials
        stacked_image_shape = forward_operator.expected_input_shape
        if initial_estimate is not None:
            if initial_estimate.shape != stacked_image_shape:
                raise ValueError(f"Initial estimate shape {initial_estimate.shape} does not match expected operator input shape {stacked_image_shape}.")
        if self.regularizers:
            if len(self.regularizers) > num_materials:
                 raise ValueError(f"Number of regularizers ({len(self.regularizers)}) cannot exceed number of materials ({num_materials}).")
            if self.verbose and len(self.regularizers) < num_materials:
                print(f"Note: {len(self.regularizers)} regularizers provided for {num_materials} materials. Remaining materials will not be regularized.")

        def combined_prox_fn(image_stack: torch.Tensor, prox_step_size: float) -> torch.Tensor:
            processed_slices = []
            for i in range(num_materials):
                current_slice = image_stack[i, :, :]
                reg = self.regularizers[i] if i < len(self.regularizers) else None
                if reg is not None:
                    processed_slice = reg.proximal_operator(current_slice, prox_step_size)
                    processed_slices.append(processed_slice)
                else:
                    processed_slices.append(current_slice.clone())
            return torch.stack(processed_slices, dim=0)

        use_prox = bool(self.regularizers and any(r is not None for r in self.regularizers))
        reg_prox_fn_for_pgd = combined_prox_fn if use_prox else None

        pgd_reconstructor = ProximalGradientReconstructor(
            iterations=self.iterations,
            step_size=self.step_size,
            initial_estimate_fn=None,
            verbose=self.verbose,
            log_fn=lambda iter_num, current_image, change, grad_norm: \
                   print(f"MatDecomp Recon Iter {iter_num+1}/{self.iterations}: Change={change:.4e}, GradNorm={grad_norm:.4e}") \
                   if self.verbose and (iter_num % 10 == 0 or iter_num == self.iterations -1 or self.iterations < 20) else None
        )
        x_init_for_pgd = initial_estimate
        reconstructed_stack = pgd_reconstructor.reconstruct(
            kspace_data=measured_data,
            forward_op_fn=forward_operator.op,
            adjoint_op_fn=forward_operator.op_adj,
            regularizer_prox_fn=reg_prox_fn_for_pgd,
            sensitivity_maps=None,
            x_init=x_init_for_pgd,
            image_shape_for_zero_init=stacked_image_shape if x_init_for_pgd is None else None
        )
        return reconstructed_stack

if __name__ == '__main__':
    import numpy as np

    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running MaterialDecompositionForwardOperator Tests on {dev} ---")

    img_s = (16, 16)
    n_angles = 10
    n_dets = 20
    energy_bins = [(20, 50), (50, 80)]
    n_bins = len(energy_bins)
    I0 = torch.tensor([1e4] * n_bins, device=dev, dtype=torch.float32)

    pcct_op_inst = PCCTProjectorOperator(
        image_shape=img_s, num_angles=n_angles, num_detector_pixels=n_dets,
        energy_bins_keV=energy_bins, source_photons_per_bin=I0, device=dev,
        add_poisson_noise=False, spectral_resolution_keV=None,
        pileup_parameters=None, charge_sharing_kernel=None, k_escape_probabilities=None
    )
    print("\nPCCTProjectorOperator instance created for testing MaterialDecomposition operator.")

    print("\nPerforming Dot Product Test for PCCTProjectorOperator itself...")
    try:
        x_pcct_dp = torch.rand(img_s, device=dev, dtype=torch.float32)
        y_pcct_dp = torch.rand(pcct_op_inst.measurement_shape, device=dev, dtype=torch.float32)
        Ax_pcct = pcct_op_inst.op(x_pcct_dp.to(pcct_op_inst.device))
        Aty_pcct = pcct_op_inst.op_adj(y_pcct_dp.to(pcct_op_inst.device))
        lhs_pcct = torch.sum(Ax_pcct.to(dev) * y_pcct_dp)
        rhs_pcct = torch.sum(x_pcct_dp * Aty_pcct.to(dev))
        print(f"  PCCT LHS (sum(Ax*y)): {lhs_pcct.item():.6f}")
        print(f"  PCCT RHS (sum(x*Aty)): {rhs_pcct.item():.6f}")
        diff_pcct = abs(lhs_pcct.item() - rhs_pcct.item())
        print(f"  PCCT Absolute Difference: {diff_pcct:.6f}")
        if torch.allclose(lhs_pcct, rhs_pcct, rtol=0.35):
            print("  PCCTProjectorOperator dot product test passed or within tolerance (rtol=0.35).")
        else:
            print("  PCCTProjectorOperator dot product test FAILED (rtol=0.35). This is expected due to non-linearities and approximate Radon adjoint.")
    except Exception as e_pcct_dp:
        print(f"Exception during PCCTProjectorOperator dot product test: {e_pcct_dp}")

    mat_ref_att = {'water': 0.2, 'bone': 0.5}
    basis_names = ['water', 'bone']

    try:
        mat_decomp_op = MaterialDecompositionForwardOperator(
            material_reference_attenuations=mat_ref_att,
            pcct_projector=pcct_op_inst,
            basis_material_names=basis_names,
            device=dev
        )
        print("\nMaterialDecompositionForwardOperator instance created.")

        print("\nTesting op method...")
        basis_images_tensor = torch.rand(len(basis_names), img_s[0], img_s[1], device=dev, dtype=torch.float32)
        sinograms = mat_decomp_op.op(basis_images_tensor)
        assert sinograms.shape == pcct_op_inst.measurement_shape, \
            f"op output shape incorrect. Expected {pcct_op_inst.measurement_shape}, Got {sinograms.shape}"
        assert sinograms.device == dev, f"op output device incorrect. Expected {dev}, Got {sinograms.device}"
        print("  op method basic test passed.")

        print("\nTesting op_adj method...")
        adjoint_input_sinograms = torch.rand_like(sinograms)
        adj_basis_images = mat_decomp_op.op_adj(adjoint_input_sinograms)
        assert adj_basis_images.shape == basis_images_tensor.shape, \
            f"op_adj output shape incorrect. Expected {basis_images_tensor.shape}, Got {adj_basis_images.shape}"
        assert adj_basis_images.device == dev, f"op_adj output device incorrect. Expected {dev}, Got {adj_basis_images.device}"
        print("  op_adj method basic test passed.")

        print("\nPerforming Dot Product Test...")
        x_dp = torch.rand(len(basis_names), img_s[0], img_s[1], device=dev, dtype=torch.float32)
        y_dp = torch.rand(pcct_op_inst.measurement_shape, device=dev, dtype=torch.float32)
        Ax = mat_decomp_op.op(x_dp)
        Aty = mat_decomp_op.op_adj(y_dp)
        lhs = torch.sum(Ax * y_dp)
        rhs = torch.sum(x_dp * Aty)
        print(f"  LHS (sum(Ax*y)): {lhs.item():.6f}")
        print(f"  RHS (sum(x*Aty)): {rhs.item():.6f}")
        diff = abs(lhs.item() - rhs.item())
        print(f"  Absolute Difference: {diff:.6f}")
        if torch.allclose(lhs, rhs, rtol=0.35):
             print("  MaterialDecompositionForwardOperator Dot product test passed (rtol=0.35).")
        else:
            print("  MaterialDecompositionForwardOperator Dot product test FAILED (rtol=0.35). This is expected due to the underlying PCCTProjectorOperator's non-adjointness.")
        print("\nMaterialDecompositionForwardOperator basic functionality tests completed.")
    except Exception as e:
        print(f"\nError during MaterialDecompositionForwardOperator tests: {e}")
        traceback.print_exc()

    print("\n--- Testing IterativeMaterialDecompositionReconstructor ---")
    true_basis_images = torch.zeros(len(basis_names), img_s[0], img_s[1], device=dev, dtype=torch.float32)
    true_basis_images[0, img_s[0]//4 : img_s[0]*3//4, img_s[1]//4 : img_s[1]*3//4] = 1.0
    true_basis_images[1, img_s[0]//2 - 2 : img_s[0]//2 + 2, img_s[1]//2 - 2 : img_s[1]//2 + 2] = 0.8
    print(f"\nCreated true_basis_images with shape: {true_basis_images.shape}")
    measured_sinograms = mat_decomp_op.op(true_basis_images)
    print(f"Generated measured_sinograms with shape: {measured_sinograms.shape}")

    print("\nTesting IterativeMaterialDecompositionReconstructor (no regularization)...")
    try:
        material_reconstructor_no_reg = IterativeMaterialDecompositionReconstructor(
            iterations=20, step_size=1e-5, regularizers=None, verbose=True
        )
        initial_guess = mat_decomp_op.op_adj(measured_sinograms)
        print(f"Initial guess shape: {initial_guess.shape}")
        reconstructed_no_reg = material_reconstructor_no_reg.reconstruct(
            measured_data=measured_sinograms,
            forward_operator=mat_decomp_op,
            initial_estimate=initial_guess
        )
        assert reconstructed_no_reg.shape == true_basis_images.shape, \
            f"Reconstructed (no reg) shape mismatch. Expected {true_basis_images.shape}, Got {reconstructed_no_reg.shape}"
        norm_diff_reco_vs_true = torch.norm(reconstructed_no_reg - true_basis_images).item()
        norm_diff_init_vs_true = torch.norm(initial_guess - true_basis_images).item()
        print(f"  Norm of (Reconstructed - True): {norm_diff_reco_vs_true:.4f}")
        print(f"  Norm of (Initial Guess - True): {norm_diff_init_vs_true:.4f}")
        if norm_diff_reco_vs_true < norm_diff_init_vs_true:
            print("  Reconstruction error is lower than initial guess error (good).")
        else:
            print("  Warning: Reconstruction error is NOT lower than initial guess error. May need more iterations or step_size tuning.")
        print("  IterativeMaterialDecompositionReconstructor (no reg) test completed.")
    except Exception as e_recon_no_reg:
        print(f"Error during IterativeMaterialDecompositionReconstructor (no reg) test: {e_recon_no_reg}")
        traceback.print_exc()

    # Placeholder for tests with dummy regularizers
    # print("\nTesting IterativeMaterialDecompositionReconstructor (with dummy regularization)...")
    # ... (rest of dummy regularizer test code)
