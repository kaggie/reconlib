import torch
from reconlib.operators import Operator
from reconlib.modalities.pcct.operators import PCCTProjectorOperator
from typing import List, Dict, Union, Optional, Callable

from reconlib.reconstructors.proximal_gradient_reconstructor import ProximalGradientReconstructor
try:
    from reconlib.regularizers.base import RegularizerBase
except ImportError:
    print("Warning: Could not import RegularizerBase from reconlib.regularizers.base. Using a dummy placeholder for type hinting.")
    class RegularizerBase:
        def proximal_operator(self, x: torch.Tensor, step_size: float) -> torch.Tensor:
            raise NotImplementedError("This is a dummy RegularizerBase.")

import traceback
import numpy as np # For np.isclose in dot product test comparison & __main__

__all__ = ['MaterialDecompositionForwardOperator', 'IterativeMaterialDecompositionReconstructor']

class MaterialDecompositionForwardOperator(Operator):
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
                 verbose: bool = False,
                 enforce_non_negativity: bool = False): # New parameter
        self.iterations = iterations
        self.step_size = step_size
        self.regularizers = regularizers if regularizers is not None else []
        self.verbose = verbose
        self.enforce_non_negativity = enforce_non_negativity # Store new parameter

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

        if self.enforce_non_negativity:
            reconstructed_stack = torch.clamp(reconstructed_stack, min=0.0)

        return reconstructed_stack

if __name__ == '__main__':
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- Running MaterialDecompositionForwardOperator Tests on {dev} ---")

    # Common PCCT parameters for tests
    img_s_fordward_op_test = (16, 16)
    n_angles_fordward_op_test = 10
    n_dets_fordward_op_test = 20
    energy_bins_fordward_op_test = [(20, 50), (50, 80)]
    n_bins_fordward_op_test = len(energy_bins_fordward_op_test)
    I0_fordward_op_test = torch.tensor([1e4] * n_bins_fordward_op_test, device=dev, dtype=torch.float32)

    pcct_op_inst_fordward_op_test = PCCTProjectorOperator(
        image_shape=img_s_fordward_op_test,
        num_angles=n_angles_fordward_op_test,
        num_detector_pixels=n_dets_fordward_op_test,
        energy_bins_keV=energy_bins_fordward_op_test,
        source_photons_per_bin=I0_fordward_op_test,
        device=dev,
        add_poisson_noise=False, spectral_resolution_keV=None,
        pileup_parameters=None, charge_sharing_kernel=None, k_escape_probabilities=None
    )
    print("\nPCCTProjectorOperator instance for ForwardOp tests created.")

    print("\nPerforming Dot Product Test for PCCTProjectorOperator itself (ForwardOp test setup)...")
    try:
        x_pcct_dp = torch.rand(img_s_fordward_op_test, device=dev, dtype=torch.float32)
        y_pcct_dp = torch.rand(pcct_op_inst_fordward_op_test.measurement_shape, device=dev, dtype=torch.float32)
        Ax_pcct = pcct_op_inst_fordward_op_test.op(x_pcct_dp.to(pcct_op_inst_fordward_op_test.device))
        Aty_pcct = pcct_op_inst_fordward_op_test.op_adj(y_pcct_dp.to(pcct_op_inst_fordward_op_test.device))
        lhs_pcct = torch.sum(Ax_pcct.to(dev) * y_pcct_dp)
        rhs_pcct = torch.sum(x_pcct_dp * Aty_pcct.to(dev))
        print(f"  PCCT LHS (sum(Ax*y)): {lhs_pcct.item():.6f}")
        print(f"  PCCT RHS (sum(x*Aty)): {rhs_pcct.item():.6f}")
        diff_pcct = abs(lhs_pcct.item() - rhs_pcct.item())
        print(f"  PCCT Absolute Difference: {diff_pcct:.6f}")
        if torch.allclose(lhs_pcct, rhs_pcct, rtol=0.35): # Existing tolerance
            print("  PCCTProjectorOperator dot product test passed or within tolerance (rtol=0.35).")
        else:
            print("  PCCTProjectorOperator dot product test FAILED (rtol=0.35). This is expected due to non-linearities and approximate Radon adjoint.")
    except Exception as e_pcct_dp:
        print(f"Exception during PCCTProjectorOperator dot product test: {e_pcct_dp}")

    mat_ref_att_2mat = {'water': 0.2, 'bone': 0.5}
    basis_names_2mat = ['water', 'bone']

    try:
        mat_decomp_op_2mat = MaterialDecompositionForwardOperator(
            material_reference_attenuations=mat_ref_att_2mat,
            pcct_projector=pcct_op_inst_fordward_op_test, # Use the 16x16 projector
            basis_material_names=basis_names_2mat,
            device=dev
        )
        print("\nMaterialDecompositionForwardOperator instance (N=2) created.")

        print("\nTesting op method (N=2)...")
        basis_images_tensor_2mat = torch.rand(len(basis_names_2mat), img_s_fordward_op_test[0], img_s_fordward_op_test[1], device=dev, dtype=torch.float32)
        sinograms_2mat = mat_decomp_op_2mat.op(basis_images_tensor_2mat)
        assert sinograms_2mat.shape == pcct_op_inst_fordward_op_test.measurement_shape
        assert sinograms_2mat.device == dev
        print("  op method basic test (N=2) passed.")

        print("\nTesting op_adj method (N=2)...")
        adjoint_input_sinograms_2mat = torch.rand_like(sinograms_2mat)
        adj_basis_images_2mat = mat_decomp_op_2mat.op_adj(adjoint_input_sinograms_2mat)
        assert adj_basis_images_2mat.shape == basis_images_tensor_2mat.shape
        assert adj_basis_images_2mat.device == dev
        print("  op_adj method basic test (N=2) passed.")

        print("\nPerforming Dot Product Test (N=2)...")
        x_dp_2mat = torch.rand(len(basis_names_2mat), img_s_fordward_op_test[0], img_s_fordward_op_test[1], device=dev, dtype=torch.float32)
        y_dp_2mat = torch.rand(pcct_op_inst_fordward_op_test.measurement_shape, device=dev, dtype=torch.float32)
        Ax_2mat = mat_decomp_op_2mat.op(x_dp_2mat)
        Aty_2mat = mat_decomp_op_2mat.op_adj(y_dp_2mat)
        lhs_2mat = torch.sum(Ax_2mat * y_dp_2mat)
        rhs_2mat = torch.sum(x_dp_2mat * Aty_2mat)
        print(f"  LHS (sum(Ax*y)) (N=2): {lhs_2mat.item():.6f}")
        print(f"  RHS (sum(x*Aty)) (N=2): {rhs_2mat.item():.6f}")
        diff_2mat = abs(lhs_2mat.item() - rhs_2mat.item())
        print(f"  Absolute Difference (N=2): {diff_2mat:.6f}")
        if torch.allclose(lhs_2mat, rhs_2mat, rtol=0.35): # Existing tolerance
             print("  MaterialDecompositionForwardOperator Dot product test (N=2) passed (rtol=0.35).")
        else:
            print("  MaterialDecompositionForwardOperator Dot product test (N=2) FAILED (rtol=0.35). This is expected.")
        print("\nMaterialDecompositionForwardOperator (N=2) basic functionality tests completed.")
    except Exception as e:
        print(f"\nError during MaterialDecompositionForwardOperator (N=2) tests: {e}")
        traceback.print_exc()

    # --- Tests for IterativeMaterialDecompositionReconstructor (N=2, original test) ---
    print("\n--- Testing IterativeMaterialDecompositionReconstructor (N=2, no regularization) ---")
    # Using mat_decomp_op_2mat and pcct_op_inst_fordward_op_test (16x16)
    true_basis_images_2mat_recon = torch.zeros(len(basis_names_2mat), img_s_fordward_op_test[0], img_s_fordward_op_test[1], device=dev, dtype=torch.float32)
    true_basis_images_2mat_recon[0, img_s_fordward_op_test[0]//4 : img_s_fordward_op_test[0]*3//4, img_s_fordward_op_test[1]//4 : img_s_fordward_op_test[1]*3//4] = 1.0
    true_basis_images_2mat_recon[1, img_s_fordward_op_test[0]//2 - 2 : img_s_fordward_op_test[0]//2 + 2, img_s_fordward_op_test[1]//2 - 2 : img_s_fordward_op_test[1]//2 + 2] = 0.8
    measured_sinograms_2mat_recon = mat_decomp_op_2mat.op(true_basis_images_2mat_recon)
    try:
        material_reconstructor_no_reg_2mat = IterativeMaterialDecompositionReconstructor(
            iterations=20, step_size=1e-5, regularizers=None, verbose=True
        )
        initial_guess_2mat_recon = mat_decomp_op_2mat.op_adj(measured_sinograms_2mat_recon)
        reconstructed_no_reg_2mat = material_reconstructor_no_reg_2mat.reconstruct(
            measured_data=measured_sinograms_2mat_recon,
            forward_operator=mat_decomp_op_2mat,
            initial_estimate=initial_guess_2mat_recon
        )
        assert reconstructed_no_reg_2mat.shape == true_basis_images_2mat_recon.shape
        norm_diff_reco_vs_true_2mat = torch.norm(reconstructed_no_reg_2mat - true_basis_images_2mat_recon).item()
        norm_diff_init_vs_true_2mat = torch.norm(initial_guess_2mat_recon - true_basis_images_2mat_recon).item()
        print(f"  N=2 Norm of (Reconstructed - True): {norm_diff_reco_vs_true_2mat:.4f}")
        print(f"  N=2 Norm of (Initial Guess - True): {norm_diff_init_vs_true_2mat:.4f}")
        if norm_diff_reco_vs_true_2mat < norm_diff_init_vs_true_2mat:
            print("  N=2 Reconstruction error is lower than initial guess error (good).")
        else:
            print("  N=2 Warning: Reconstruction error is NOT lower than initial guess error.")
        print("  IterativeMaterialDecompositionReconstructor (N=2, no reg) test completed.")
    except Exception as e_recon_no_reg_2mat:
        print(f"Error during IterativeMaterialDecompositionReconstructor (N=2, no reg) test: {e_recon_no_reg_2mat}")
        traceback.print_exc()

    # --- Test IterativeMaterialDecompositionReconstructor with N=3 materials (K-Edge Demo) ---
    print("\n--- Testing IterativeMaterialDecompositionReconstructor (N=3 materials, K-Edge Demo, non-negativity) ---")
    img_s_kedge = (32, 32)

    # PCCTProjectorOperator Configuration for K-Edge Test
    # Hypothetical K-edge at 50 keV
    energy_bins_keV_kedge = [(30.0, 48.0), (48.0, 52.0), (52.0, 70.0)] # Bin straddling K-edge
    num_bins_kedge = len(energy_bins_keV_kedge)
    # Energy scaling factors for a reference material (e.g., water) across these bins.
    # Attenuation typically decreases with energy, except across an edge.
    # For simplicity, assume reference material (water) has decreasing attenuation.
    # The K-edge material's specific behavior is captured by its basis image and reference mu.
    energy_scaling_factors_kedge = torch.tensor([1.2, 1.0, 0.8], device=dev, dtype=torch.float32)
    source_photons_per_bin_kedge = torch.tensor([1e5] * num_bins_kedge, device=dev, dtype=torch.float32)

    pcct_op_inst_kedge_test = PCCTProjectorOperator(
        image_shape=img_s_kedge,
        num_angles=n_angles_fordward_op_test, # Using n_angles from previous N=2 test setup
        num_detector_pixels=n_dets_fordward_op_test, # Using n_dets from previous N=2 test setup
        energy_bins_keV=energy_bins_keV_kedge,
        source_photons_per_bin=source_photons_per_bin_kedge,
        energy_scaling_factors=energy_scaling_factors_kedge, # Key for K-edge differentiation
        device=dev,
        add_poisson_noise=False, spectral_resolution_keV=None,
        pileup_parameters=None, charge_sharing_kernel=None, k_escape_probabilities=None
    )
    print("\nPCCTProjectorOperator instance for K-Edge test created (32x32 image, 3 bins).")

    # Material Definitions for K-Edge Test
    mat_ref_att_kedge = {'water': 0.20, 'soft_tissue': 0.19, 'contrast_agent': 0.80} # mu at reference energy for energy_scaling_factors
    basis_names_kedge = ['water', 'soft_tissue', 'contrast_agent']

    mat_decomp_op_kedge = MaterialDecompositionForwardOperator(
        material_reference_attenuations=mat_ref_att_kedge,
        pcct_projector=pcct_op_inst_kedge_test,
        basis_material_names=basis_names_kedge,
        device=dev
    )
    print("MaterialDecompositionForwardOperator for K-Edge test created.")

    # True Basis Images for K-Edge Test
    true_basis_images_kedge = torch.zeros(len(basis_names_kedge), img_s_kedge[0], img_s_kedge[1], device=dev, dtype=torch.float32)
    # Material 0 (water): Background
    true_basis_images_kedge[0, :, :] = 1.0
    # Material 1 (soft_tissue): Large central square, displacing some water
    h, w = img_s_kedge
    true_basis_images_kedge[1, h//4 : h*3//4, w//4 : w*3//4] = 0.7
    true_basis_images_kedge[0, h//4 : h*3//4, w//4 : w*3//4] -= 0.7 # Assume densities sum to 1 in this region
    # Material 2 (contrast_agent): Smaller circle within the soft tissue
    radius_contrast_kedge = h // 6
    cx_contrast_kedge, cy_contrast_kedge = w // 2, h // 2
    yy_k, xx_k = torch.meshgrid(torch.arange(h, device=dev), torch.arange(w, device=dev), indexing='ij')
    mask_contrast_circle_kedge = (xx_k - cx_contrast_kedge)**2 + (yy_k - cy_contrast_kedge)**2 < radius_contrast_kedge**2
    true_basis_images_kedge[2, mask_contrast_circle_kedge] = 0.5 # Density of contrast agent
    true_basis_images_kedge[1, mask_contrast_circle_kedge] -= 0.5 # Displace soft tissue where contrast is present
    true_basis_images_kedge = torch.clamp(true_basis_images_kedge, min=0.0) # Ensure non-negativity from subtractions

    print(f"Created true_basis_images_kedge with shape: {true_basis_images_kedge.shape}")
    for i, name in enumerate(basis_names_kedge):
        print(f"  Norm of true basis image '{name}': {torch.norm(true_basis_images_kedge[i]).item():.2f}")


    measured_sinograms_kedge = mat_decomp_op_kedge.op(true_basis_images_kedge)
    print(f"Generated measured_sinograms_kedge with shape: {measured_sinograms_kedge.shape}")

    try:
        material_reconstructor_kedge = IterativeMaterialDecompositionReconstructor(
            iterations=50,      # Increased iterations
            step_size=1e-6,     # Adjusted step size
            regularizers=None,
            verbose=True,
            enforce_non_negativity=True
        )

        initial_guess_kedge = mat_decomp_op_kedge.op_adj(measured_sinograms_kedge)
        initial_guess_kedge = torch.clamp(initial_guess_kedge, min=0.0)
        print(f"Initial guess (K-Edge) shape: {initial_guess_kedge.shape}")

        reconstructed_kedge_images = material_reconstructor_kedge.reconstruct(
            measured_data=measured_sinograms_kedge,
            forward_operator=mat_decomp_op_kedge,
            initial_estimate=initial_guess_kedge
        )

        assert reconstructed_kedge_images.shape == true_basis_images_kedge.shape, \
            f"Reconstructed (K-Edge) shape mismatch. Expected {true_basis_images_kedge.shape}, Got {reconstructed_kedge_images.shape}"

        norm_diff_reco_kedge_vs_true = torch.norm(reconstructed_kedge_images - true_basis_images_kedge).item()
        norm_diff_init_kedge_vs_true = torch.norm(initial_guess_kedge - true_basis_images_kedge).item()

        print(f"  K-Edge Norm of (Reconstructed Stack - True Stack): {norm_diff_reco_kedge_vs_true:.4f}")
        print(f"  K-Edge Norm of (Initial Guess Stack - True Stack): {norm_diff_init_kedge_vs_true:.4f}")

        if norm_diff_reco_kedge_vs_true < norm_diff_init_kedge_vs_true:
            print("  K-Edge Reconstruction error (stack) is lower than initial guess error (good).")
        else:
            print("  K-Edge Warning: Reconstruction error (stack) is NOT lower than initial guess error. May need more iterations or step_size tuning.")

        # Check contrast agent map specifically
        norm_diff_reco_contrast_agent = torch.norm(reconstructed_kedge_images[2] - true_basis_images_kedge[2]).item()
        norm_diff_init_contrast_agent = torch.norm(initial_guess_kedge[2] - true_basis_images_kedge[2]).item()
        print(f"  K-Edge Norm of (Recon Contrast Agent - True): {norm_diff_reco_contrast_agent:.4f}")
        print(f"  K-Edge Norm of (Initial Contrast Agent - True): {norm_diff_init_contrast_agent:.4f}")
        if norm_diff_reco_contrast_agent < norm_diff_init_contrast_agent:
             print("  K-Edge Contrast Agent map reconstruction improved over initial guess.")
        else:
            print("  K-Edge Warning: Contrast Agent map reconstruction did NOT improve over initial guess.")


        if material_reconstructor_kedge.enforce_non_negativity:
            assert torch.all(reconstructed_kedge_images >= -1e-6), "Non-negativity constraint failed (K-Edge)."
            print("  Non-negativity constraint effectively verified (K-Edge).")

        print("  IterativeMaterialDecompositionReconstructor (N=3 materials, K-Edge demo) test completed.")

    except Exception as e_recon_kedge:
        print(f"Error during IterativeMaterialDecompositionReconstructor (K-Edge demo) test: {e_recon_kedge}")
        traceback.print_exc()
