import torch
from reconlib.operators import Operator
import numpy as np

class XRayDiffractionOperator(Operator):
    """
    Placeholder Forward and Adjoint Operator for X-ray Diffraction Imaging.

    Models a simplified far-field diffraction scenario where the measured data
    is the magnitude of the Fourier Transform of the object's structure.
    The phase information is lost in this simplified model of detection.

    Forward: object (real-space) -> |FT(object)| (diffraction pattern magnitudes)
    Adjoint (simplified): diffraction_magnitudes -> IFT(magnitudes * estimated_phase)
    """
    def __init__(self,
                 image_shape: tuple[int, int], # (Ny, Nx) - real-space object representation
                 add_random_phase_to_adjoint: bool = False, # For basic phase retrieval attempts
                 device: str | torch.device = 'cpu'):
        super().__init__()
        self.image_shape = image_shape
        self.device = torch.device(device)
        self.add_random_phase_to_adjoint = add_random_phase_to_adjoint

        # Measurement shape will be the same as image_shape (for FT magnitudes)
        self.measurement_shape = image_shape

        print(f"XRayDiffractionOperator (Placeholder) initialized.")
        print(f"  Image Shape: {self.image_shape}")
        print(f"  Adjoint will use {'random phase' if add_random_phase_to_adjoint else 'zero phase'} for IFT if no phase provided.")

    def op(self, object_map: torch.Tensor) -> torch.Tensor:
        """
        Forward: Real-space object map to diffraction pattern magnitudes.
        Output: Magnitude of the 2D Fourier Transform.
        """
        if object_map.shape != self.image_shape:
            raise ValueError(f"Input object_map shape {object_map.shape} must match {self.image_shape}.")
        object_map = object_map.to(self.device)

        if object_map.is_complex():
            print("Warning: XRayDiffractionOperator expects a real object_map. Taking real part.")
            object_map = object_map.real

        # Compute Fourier Transform
        diffraction_pattern_complex = torch.fft.fft2(object_map, norm='ortho')
        # Keep only the magnitude (simulating intensity measurement)
        diffraction_magnitudes = torch.abs(diffraction_pattern_complex)

        return diffraction_magnitudes

    def op_adj(self, diffraction_magnitudes: torch.Tensor,
               phase_estimate: torch.Tensor | None = None) -> torch.Tensor:
        """
        Adjoint (simplified): Diffraction pattern magnitudes to real-space object estimate.
        Combines magnitudes with an estimated (or zero/random) phase and performs IFT.

        Args:
            diffraction_magnitudes (torch.Tensor): Measured diffraction magnitudes. Shape: self.image_shape.
            phase_estimate (torch.Tensor | None, optional): An estimate of the phase.
                                                            Shape: self.image_shape. If None, uses zero or random phase.
        Returns:
            torch.Tensor: Reconstructed real-space object (should be real).
        """
        if diffraction_magnitudes.shape != self.measurement_shape: # measurement_shape is image_shape here
            raise ValueError(f"Input magnitudes shape {diffraction_magnitudes.shape} must match {self.measurement_shape}.")
        diffraction_magnitudes = diffraction_magnitudes.to(self.device)

        if phase_estimate is not None:
            if phase_estimate.shape != self.image_shape:
                raise ValueError(f"Phase estimate shape {phase_estimate.shape} must match {self.image_shape}.")
            phase_estimate = phase_estimate.to(self.device)
            k_space_estimate = diffraction_magnitudes * torch.exp(1j * phase_estimate)
        elif self.add_random_phase_to_adjoint:
            random_phases = torch.rand(self.image_shape, device=self.device) * 2 * np.pi
            k_space_estimate = diffraction_magnitudes * torch.exp(1j * random_phases)
        else: # Zero phase
            k_space_estimate = diffraction_magnitudes.to(torch.complex64) # Make it complex with zero imag part

        # Inverse Fourier Transform
        object_estimate_complex = torch.fft.ifft2(k_space_estimate, norm='ortho')

        # Result should ideally be real if the original object was real and phase was correct
        return object_estimate_complex.real

    # Standard dot product test is not directly applicable here because op() is not linear (due to abs()).
    # Phase retrieval algorithms test consistency with magnitude constraints instead.
    # We can test op_adj separately for its IFT property if needed.

if __name__ == '__main__':
    print("\nRunning basic XRayDiffractionOperator (Placeholder) checks...")
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_s = (64, 64)

    try:
        xrd_op = XRayDiffractionOperator(image_shape=img_s, device=dev)
        print("XRayDiffractionOperator instantiated.")

        phantom_object = torch.zeros(img_s, device=dev)
        phantom_object[img_s[0]//4:img_s[0]*3//4, img_s[1]//4:img_s[1]*3//4] = 1.0
        phantom_object[10:20,10:20]=0.5 # Some internal structure

        # Test forward op
        magnitudes = xrd_op.op(phantom_object)
        print(f"Forward op output shape (magnitudes): {magnitudes.shape}")
        assert magnitudes.shape == img_s
        assert not magnitudes.is_complex()

        # Test adjoint op (with zero phase)
        recon_adj_zero_phase = xrd_op.op_adj(magnitudes)
        print(f"Adjoint op (zero phase) output shape: {recon_adj_zero_phase.shape}")
        assert recon_adj_zero_phase.shape == img_s
        assert not recon_adj_zero_phase.is_complex()

        # Test adjoint op (with random phase)
        xrd_op_rand_phase = XRayDiffractionOperator(image_shape=img_s, add_random_phase_to_adjoint=True, device=dev)
        recon_adj_rand_phase = xrd_op_rand_phase.op_adj(magnitudes)
        print(f"Adjoint op (random phase) output shape: {recon_adj_rand_phase.shape}")
        assert recon_adj_rand_phase.shape == img_s

        # Test adjoint op (with true phase - for ideal scenario check)
        true_complex_pattern = torch.fft.fft2(phantom_object, norm='ortho')
        true_phase = torch.angle(true_complex_pattern)
        recon_adj_true_phase = xrd_op.op_adj(magnitudes, phase_estimate=true_phase)

        # With true phase, reconstruction should be close to original
        # print(f"Max diff with true phase recon: {torch.max(torch.abs(recon_adj_true_phase - phantom_object))}")
        assert torch.allclose(recon_adj_true_phase, phantom_object, atol=1e-5), \
            "Adjoint with true phase should reconstruct the object closely."

        print("XRayDiffractionOperator __main__ checks completed (op, op_adj with different phases).")
        print("Note: Standard dot product test is not applicable due to non-linear forward op (|FT|).")

    except Exception as e:
        print(f"Error in XRayDiffractionOperator __main__ checks: {e}")
        import traceback; traceback.print_exc()
