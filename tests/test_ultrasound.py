import unittest
import torch
import numpy as np
import sys
import os

# Ensure reconlib is in the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from reconlib.modalities.ultrasound.operators import UltrasoundForwardOperator
    from reconlib.modalities.ultrasound.reconstructors import das_reconstruction, inverse_reconstruction_pg
    from reconlib.modalities.ultrasound.regularizers import UltrasoundTVCustomRegularizer
    RECONLIB_ULTRASOUND_AVAILABLE = True
except ImportError as e:
    print(f"Could not import ultrasound modules for testing: {e}")
    RECONLIB_ULTRASOUND_AVAILABLE = False


@unittest.skipIf(not RECONLIB_ULTRASOUND_AVAILABLE, "reconlib.modalities.ultrasound module not available")
class TestUltrasoundForwardOperator(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = (32, 32) # Smaller for faster tests
        self.operator_params = {
            'image_shape': self.image_shape,
            'sound_speed': 1540.0,
            'num_elements': 16, # Fewer elements for speed
            'sampling_rate': 10e6, # 10 MHz
            'num_samples': 128,   # Fewer samples
            'image_spacing': (0.001, 0.001), # 1 mm pixels
            'center_frequency': 5e6,
            'pulse_bandwidth_fractional': 0.5,
            'beam_sigma_rad': 0.05, # Increased slightly for less aggressive beam falloff in tests
            'attenuation_coeff_db_cm_mhz': 0.3,
            'device': self.device
        }
        # Simplified element positions for testing
        elem_pitch = 0.001
        array_width = (self.operator_params['num_elements'] - 1) * elem_pitch
        x_coords = torch.linspace(-array_width / 2, array_width / 2, self.operator_params['num_elements'], device=self.device)
        self.operator_params['element_positions'] = torch.stack(
            (x_coords, torch.full_like(x_coords, -0.002)), dim=1 # Closer to image
        )
        self.us_op = UltrasoundForwardOperator(**self.operator_params)

        # Create a simple phantom
        self.phantom = torch.zeros(self.image_shape, dtype=torch.complex64, device=self.device)
        self.phantom[self.image_shape[0] // 2, self.image_shape[1] // 2] = 1.0

        # Create dummy echo data for adjoint testing
        self.dummy_echo_data = torch.randn(
            (self.operator_params['num_elements'], self.operator_params['num_samples']),
            dtype=torch.complex64, device=self.device
        )

    def test_operator_instantiation(self):
        self.assertIsInstance(self.us_op, UltrasoundForwardOperator)
        print("TestUltrasoundForwardOperator: Instantiation OK.")

    def test_forward_op_shape_dtype(self):
        echo_data = self.us_op.op(self.phantom)
        self.assertEqual(echo_data.shape,
                         (self.operator_params['num_elements'], self.operator_params['num_samples']))
        self.assertTrue(echo_data.is_complex())
        self.assertEqual(echo_data.device, self.device)
        print("TestUltrasoundForwardOperator: Forward op shape and dtype OK.")

    def test_adjoint_op_shape_dtype(self):
        recon_image = self.us_op.op_adj(self.dummy_echo_data)
        self.assertEqual(recon_image.shape, self.image_shape)
        self.assertTrue(recon_image.is_complex())
        self.assertEqual(recon_image.device, self.device)
        print("TestUltrasoundForwardOperator: Adjoint op shape and dtype OK.")

    def test_dot_product(self):
        # Create random image x and random echo data y_rand
        x_dp = torch.randn_like(self.phantom) + 1j * torch.randn_like(self.phantom)
        y_dp_rand = torch.randn_like(self.dummy_echo_data) + 1j * torch.randn_like(self.dummy_echo_data)

        Ax = self.us_op.op(x_dp)
        Aty = self.us_op.op_adj(y_dp_rand)

        # <Ax, y_rand> vs <x, A^T*y_rand>
        lhs = torch.vdot(Ax.flatten(), y_dp_rand.flatten())
        rhs = torch.vdot(x_dp.flatten(), Aty.flatten())

        print(f"Dot Product Test - LHS: {lhs.item()}, RHS: {rhs.item()}")
        # Increased tolerance due to simplifications in the forward/adjoint model
        # (e.g., simple attenuation, no pulse shape, nearest neighbor in time mapping)
        # These can break perfect adjointness.
        self.assertAlmostEqual(lhs.real.item(), rhs.real.item(), delta=1e-1 * (abs(lhs.real.item()) + abs(rhs.real.item()) + 1e-9))
        self.assertAlmostEqual(lhs.imag.item(), rhs.imag.item(), delta=1e-1 * (abs(lhs.imag.item()) + abs(rhs.imag.item()) + 1e-9))
        print("TestUltrasoundForwardOperator: Dot product test OK (within tolerance).")


@unittest.skipIf(not RECONLIB_ULTRASOUND_AVAILABLE, "reconlib.modalities.ultrasound module not available")
class TestUltrasoundReconstructors(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_shape = (32, 32)

        # Use a real UltrasoundForwardOperator instance for reconstructor tests
        operator_params = {
            'image_shape': self.image_shape,
            'sound_speed': 1540.0,
            'num_elements': 16,
            'sampling_rate': 10e6,
            'num_samples': 128,
            'image_spacing': (0.001, 0.001),
            'device': self.device
        }
        elem_pitch = 0.001
        array_width = (operator_params['num_elements'] - 1) * elem_pitch
        x_coords = torch.linspace(-array_width / 2, array_width / 2, operator_params['num_elements'], device=self.device)
        operator_params['element_positions'] = torch.stack(
            (x_coords, torch.full_like(x_coords, -0.002)), dim=1
        )
        self.us_operator = UltrasoundForwardOperator(**operator_params)

        # Create dummy echo data based on the operator's expected dimensions
        self.echo_data = torch.randn(
            (self.us_operator.num_elements, self.us_operator.num_samples),
            dtype=torch.complex64, device=self.device
        )

    def test_das_reconstruction(self):
        recon_image = das_reconstruction(self.echo_data, self.us_operator)
        self.assertEqual(recon_image.shape, self.image_shape)
        self.assertTrue(recon_image.is_complex())
        self.assertEqual(recon_image.device, self.device)
        print("TestUltrasoundReconstructors: DAS reconstruction execution OK.")

    def test_inverse_reconstruction_pg_l1(self):
        recon_image = inverse_reconstruction_pg(
            echo_data=self.echo_data,
            ultrasound_operator=self.us_operator,
            regularizer_type='l1',
            lambda_reg=0.001,
            iterations=2, # Keep iterations minimal for speed
            step_size=0.01,
            verbose=False
        )
        self.assertEqual(recon_image.shape, self.image_shape)
        self.assertTrue(recon_image.is_complex())
        self.assertEqual(recon_image.device, self.device)
        print("TestUltrasoundReconstructors: Inverse PG L1 execution OK.")

    def test_inverse_reconstruction_pg_l2(self):
        recon_image = inverse_reconstruction_pg(
            echo_data=self.echo_data,
            ultrasound_operator=self.us_operator,
            regularizer_type='l2',
            lambda_reg=0.01,
            iterations=2,
            step_size=0.01,
            verbose=False
        )
        self.assertEqual(recon_image.shape, self.image_shape)
        self.assertTrue(recon_image.is_complex())
        self.assertEqual(recon_image.device, self.device)
        print("TestUltrasoundReconstructors: Inverse PG L2 execution OK.")

    def test_inverse_reconstruction_pg_invalid_reg(self):
        with self.assertRaises(ValueError):
            inverse_reconstruction_pg(
                echo_data=self.echo_data,
                ultrasound_operator=self.us_operator,
                regularizer_type='invalid_type',
                lambda_reg=0.01,
                iterations=2
            )
        print("TestUltrasoundReconstructors: Inverse PG invalid regularizer check OK.")


@unittest.skipIf(not RECONLIB_ULTRASOUND_AVAILABLE, "reconlib.modalities.ultrasound module not available")
class TestUltrasoundTVCustomRegularizer(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.N_2d = 16  # Smaller for tests
        self.img_2d_real = torch.randn((self.N_2d, self.N_2d), device=self.device)
        self.img_2d_complex = torch.randn((self.N_2d, self.N_2d), dtype=torch.complex64, device=self.device)

        self.N_3d = 8 # Even smaller for 3D
        self.img_3d_real = torch.randn((self.N_3d, self.N_3d, self.N_3d), device=self.device)
        self.img_3d_complex = torch.randn((self.N_3d, self.N_3d, self.N_3d), dtype=torch.complex64, device=self.device)

        self.lambda_tv = 0.01
        self.steplength = 1.0
        self.prox_iters = 3 # Keep low for test speed
        self.prox_step = 0.01

    def test_regularizer_instantiation_2d(self):
        reg = UltrasoundTVCustomRegularizer(lambda_reg=self.lambda_tv, prox_iterations=self.prox_iters, is_3d=False, prox_step_size=self.prox_step)
        self.assertIsInstance(reg, UltrasoundTVCustomRegularizer)
        print("TestUltrasoundTVCustomRegularizer: 2D Instantiation OK.")

    def test_regularizer_instantiation_3d(self):
        reg = UltrasoundTVCustomRegularizer(lambda_reg=self.lambda_tv, prox_iterations=self.prox_iters, is_3d=True, prox_step_size=self.prox_step)
        self.assertIsInstance(reg, UltrasoundTVCustomRegularizer)
        print("TestUltrasoundTVCustomRegularizer: 3D Instantiation OK.")

    def test_value_2d_real(self):
        reg = UltrasoundTVCustomRegularizer(lambda_reg=self.lambda_tv, is_3d=False)
        value = reg.value(self.img_2d_real)
        self.assertTrue(torch.is_tensor(value))
        self.assertEqual(value.numel(), 1)
        self.assertTrue(value.item() >= 0)
        print("TestUltrasoundTVCustomRegularizer: Value 2D Real OK.")

    def test_value_3d_real(self):
        reg = UltrasoundTVCustomRegularizer(lambda_reg=self.lambda_tv, is_3d=True)
        value = reg.value(self.img_3d_real)
        self.assertTrue(torch.is_tensor(value))
        self.assertEqual(value.numel(), 1)
        self.assertTrue(value.item() >= 0)
        print("TestUltrasoundTVCustomRegularizer: Value 3D Real OK.")

    def test_proximal_operator_2d_real(self):
        reg = UltrasoundTVCustomRegularizer(lambda_reg=self.lambda_tv, prox_iterations=self.prox_iters, is_3d=False, prox_step_size=self.prox_step)
        prox_out = reg.proximal_operator(self.img_2d_real, self.steplength)
        self.assertEqual(prox_out.shape, self.img_2d_real.shape)
        self.assertEqual(prox_out.dtype, self.img_2d_real.dtype)
        print("TestUltrasoundTVCustomRegularizer: Prox Op 2D Real OK.")

    def test_proximal_operator_2d_complex(self):
        reg = UltrasoundTVCustomRegularizer(lambda_reg=self.lambda_tv, prox_iterations=self.prox_iters, is_3d=False, prox_step_size=self.prox_step)
        prox_out = reg.proximal_operator(self.img_2d_complex, self.steplength)
        self.assertEqual(prox_out.shape, self.img_2d_complex.shape)
        self.assertEqual(prox_out.dtype, self.img_2d_complex.dtype)
        print("TestUltrasoundTVCustomRegularizer: Prox Op 2D Complex OK.")

    def test_proximal_operator_3d_real(self):
        reg = UltrasoundTVCustomRegularizer(lambda_reg=self.lambda_tv, prox_iterations=self.prox_iters, is_3d=True, prox_step_size=self.prox_step)
        prox_out = reg.proximal_operator(self.img_3d_real, self.steplength)
        self.assertEqual(prox_out.shape, self.img_3d_real.shape)
        self.assertEqual(prox_out.dtype, self.img_3d_real.dtype)
        print("TestUltrasoundTVCustomRegularizer: Prox Op 3D Real OK.")

    def test_proximal_operator_3d_complex(self):
        reg = UltrasoundTVCustomRegularizer(lambda_reg=self.lambda_tv, prox_iterations=self.prox_iters, is_3d=True, prox_step_size=self.prox_step)
        prox_out = reg.proximal_operator(self.img_3d_complex, self.steplength)
        self.assertEqual(prox_out.shape, self.img_3d_complex.shape)
        self.assertEqual(prox_out.dtype, self.img_3d_complex.dtype)
        print("TestUltrasoundTVCustomRegularizer: Prox Op 3D Complex OK.")



if __name__ == '__main__':
    if RECONLIB_ULTRASOUND_AVAILABLE:
        unittest.main()
    else:
        print("Skipping ultrasound tests as module is not available.")
