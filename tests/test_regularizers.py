import unittest
import torch
from reconlib.regularizers.base import Regularizer
from reconlib.regularizers.common import L1Regularizer, L2Regularizer, TVRegularizer, HuberRegularizer, CharbonnierRegularizer
from reconlib.regularizers.functional import l1_norm, l2_norm_squared, total_variation, huber_penalty, charbonnier_penalty

# It's good practice to have a common device for tests
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TestL1Regularizer(unittest.TestCase):
    def setUp(self):
        self.lambda_reg = 0.1
        self.l1_reg = L1Regularizer(lambda_reg=self.lambda_reg).to(DEVICE)
        self.x_real = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], device=DEVICE)
        self.x_cplx = torch.tensor([-2+1j, -0.5-0.5j, 0+0j, 0.5+1j, 2-0.5j], dtype=torch.complex64, device=DEVICE)

    def test_instantiation(self):
        self.assertIsInstance(self.l1_reg, Regularizer)
        self.assertEqual(self.l1_reg.lambda_reg, self.lambda_reg)
        with self.assertRaises(ValueError):
            L1Regularizer(lambda_reg=-0.1)

    def test_value_real(self):
        expected_value = self.lambda_reg * torch.sum(torch.abs(self.x_real))
        torch.testing.assert_close(self.l1_reg.value(self.x_real), expected_value)

    def test_value_complex(self):
        expected_value = self.lambda_reg * torch.sum(torch.abs(self.x_cplx))
        torch.testing.assert_close(self.l1_reg.value(self.x_cplx), expected_value)

    def test_proximal_operator_real(self):
        steplength = 1.0
        threshold = self.lambda_reg * steplength
        expected_prox = torch.sign(self.x_real) * torch.maximum(torch.abs(self.x_real) - threshold, torch.zeros_like(self.x_real))
        torch.testing.assert_close(self.l1_reg.proximal_operator(self.x_real, steplength), expected_prox)

    def test_proximal_operator_complex(self):
        steplength = 0.5
        threshold = self.lambda_reg * steplength
        
        abs_x = torch.abs(self.x_cplx)
        shrinkage = torch.maximum(abs_x - threshold, torch.zeros_like(abs_x))
        expected_prox = torch.sgn(self.x_cplx) * shrinkage
        
        torch.testing.assert_close(self.l1_reg.proximal_operator(self.x_cplx, steplength), expected_prox)

    def test_proximal_operator_steplength_tensor(self):
        steplength = torch.tensor([0.5, 1.0, 0.1, 2.0, 1.5], device=DEVICE)
        threshold = self.lambda_reg * steplength
        expected_prox = torch.sign(self.x_real) * torch.maximum(torch.abs(self.x_real) - threshold, torch.zeros_like(self.x_real))
        torch.testing.assert_close(self.l1_reg.proximal_operator(self.x_real, steplength), expected_prox)


class TestL2Regularizer(unittest.TestCase):
    def setUp(self):
        self.lambda_reg = 0.2
        self.l2_reg = L2Regularizer(lambda_reg=self.lambda_reg).to(DEVICE)
        self.x_real = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0], device=DEVICE)
        self.x_cplx = torch.tensor([-2+1j, -0.5-0.5j, 0+0j, 0.5+1j, 2-0.5j], dtype=torch.complex64, device=DEVICE)

    def test_instantiation(self):
        self.assertIsInstance(self.l2_reg, Regularizer)
        self.assertEqual(self.l2_reg.lambda_reg, self.lambda_reg)
        with self.assertRaises(ValueError):
            L2Regularizer(lambda_reg=-0.1)

    def test_value_real(self):
        expected_value = 0.5 * self.lambda_reg * torch.sum(self.x_real**2)
        torch.testing.assert_close(self.l2_reg.value(self.x_real), expected_value)

    def test_value_complex(self):
        expected_value = 0.5 * self.lambda_reg * torch.sum(torch.abs(self.x_cplx)**2)
        torch.testing.assert_close(self.l2_reg.value(self.x_cplx), expected_value)

    def test_proximal_operator_real(self):
        steplength = 1.0
        expected_prox = self.x_real / (1 + self.lambda_reg * steplength)
        torch.testing.assert_close(self.l2_reg.proximal_operator(self.x_real, steplength), expected_prox)

    def test_proximal_operator_complex(self):
        steplength = 2.0
        expected_prox = self.x_cplx / (1 + self.lambda_reg * steplength)
        torch.testing.assert_close(self.l2_reg.proximal_operator(self.x_cplx, steplength), expected_prox)

# Placeholder for TV, Huber, Charbonnier tests - to be expanded
class TestTVRegularizer(unittest.TestCase):
    def setUp(self):
        self.lambda_param = 0.05
        self.tv_reg = TVRegularizer(lambda_param=self.lambda_param, max_chambolle_iter=20).to(DEVICE)
        self.img_2d = torch.zeros(16, 16, device=DEVICE)
        self.img_2d[4:12, 4:12] = 1.0
        self.img_3d = torch.zeros(8, 16, 16, device=DEVICE)
        self.img_3d[2:6, 4:12, 4:12] = 1.0
        self.img_cplx_2d = torch.complex(torch.randn(16,16, device=DEVICE), torch.randn(16,16, device=DEVICE))


    def test_instantiation(self):
        self.assertIsInstance(self.tv_reg, Regularizer)
        with self.assertRaises(ValueError):
            TVRegularizer(lambda_param=-0.1)

    def test_value_2d(self):
        expected_value = self.lambda_param * total_variation(self.img_2d, isotropic=True)
        torch.testing.assert_close(self.tv_reg.value(self.img_2d), expected_value)

    def test_value_3d(self):
        expected_value = self.lambda_param * total_variation(self.img_3d, isotropic=True)
        torch.testing.assert_close(self.tv_reg.value(self.img_3d), expected_value)
        
    def test_prox_2d_isotropic(self):
        # Test that prox reduces TV or keeps it smooth
        noisy_img = self.img_2d + 0.2 * torch.randn_like(self.img_2d)
        denoised_img = self.tv_reg.proximal_operator(noisy_img, steplength=1.0)
        self.assertEqual(denoised_img.shape, noisy_img.shape)
        # A simple check: norm should decrease or not increase too much if already smooth
        self.assertTrue(torch.norm(denoised_img) <= torch.norm(noisy_img) + 1e-1) # Allow slight increase for noise structure
        # More specific checks would require known input/output pairs or more sophisticated analysis

    def test_prox_3d_isotropic(self):
        noisy_img = self.img_3d + 0.2 * torch.randn_like(self.img_3d)
        denoised_img = self.tv_reg.proximal_operator(noisy_img, steplength=1.0)
        self.assertEqual(denoised_img.shape, noisy_img.shape)
        self.assertTrue(torch.norm(denoised_img) <= torch.norm(noisy_img) + 1e-1)
        
    def test_prox_complex_2d(self):
        noisy_cplx_img = self.img_cplx_2d + 0.2 * torch.complex(torch.randn_like(self.img_cplx_2d.real), torch.randn_like(self.img_cplx_2d.imag))
        denoised_cplx = self.tv_reg.proximal_operator(noisy_cplx_img, steplength=1.0)
        self.assertEqual(denoised_cplx.shape, noisy_cplx_img.shape)
        self.assertTrue(denoised_cplx.is_complex())


class TestHuberRegularizer(unittest.TestCase):
    def setUp(self):
        self.lambda_reg = 0.1
        self.delta = 0.5
        self.huber_reg = HuberRegularizer(lambda_reg=self.lambda_reg, delta=self.delta).to(DEVICE)
        self.x = torch.tensor([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0], device=DEVICE)

    def test_instantiation(self):
        self.assertIsInstance(self.huber_reg, Regularizer)
        with self.assertRaises(ValueError): HuberRegularizer(lambda_reg=-0.1, delta=0.5)
        with self.assertRaises(ValueError): HuberRegularizer(lambda_reg=0.1, delta=-0.1)

    def test_value(self):
        expected_val = self.lambda_reg * huber_penalty(self.x, self.delta)
        torch.testing.assert_close(self.huber_reg.value(self.x), expected_val)

    def test_proximal_operator(self):
        steplength = 1.0
        gamma_eff = self.lambda_reg * steplength
        
        # Expected values based on the prox logic:
        # u = x / (1+gamma_eff) if |x/(1+gamma_eff)| <= delta
        # u = x - gamma_eff * delta * sign(x) if |x/(1+gamma_eff)| > delta
        # Let's manually compute for self.x = torch.tensor([-1.0, -0.6, -0.3, 0.0, 0.3, 0.6, 1.0])
        # gamma_eff = 0.1, delta = 0.5
        # For x = -1.0: u_case1 = -1.0/1.1 = -0.909. abs(u_case1)=0.909 > delta=0.5. So case 2.
        #   u = -1.0 - 0.1 * 0.5 * (-1) = -1.0 + 0.05 = -0.95
        # For x = -0.6: u_case1 = -0.6/1.1 = -0.545. abs(u_case1)=0.545 > delta=0.5. So case 2.
        #   u = -0.6 - 0.1 * 0.5 * (-1) = -0.6 + 0.05 = -0.55
        # For x = -0.3: u_case1 = -0.3/1.1 = -0.272. abs(u_case1)=0.272 <= delta=0.5. So case 1.
        #   u = -0.2727...
        # For x = 0.0: u_case1 = 0. abs(u_case1)=0 <= delta=0.5. So case 1. u = 0.
        expected = torch.tensor([-0.95, -0.55, -0.3/1.1, 0.0, 0.3/1.1, 0.55, 0.95], device=DEVICE)
        
        # Need to recheck the prox logic in HuberRegularizer implementation against these manual values
        # The worker used:
        # u_case1 = x / (1 + gamma_eff) -> active if abs(u_case1) <= delta
        # u_case2_pos = x - gamma_eff * delta -> active if u_case2_pos > delta AND not case1
        # u_case2_neg = x + gamma_eff * delta -> active if u_case2_neg < -delta AND not case1
        # This logic seems correct.
        
        prox_out = self.huber_reg.proximal_operator(self.x, steplength)
        torch.testing.assert_close(prox_out, expected, rtol=1e-4, atol=1e-4)


class TestCharbonnierRegularizer(unittest.TestCase):
    def setUp(self):
        self.lambda_reg = 0.1
        self.epsilon = 0.01
        self.char_reg = CharbonnierRegularizer(lambda_reg=self.lambda_reg, epsilon=self.epsilon, newton_iter=10).to(DEVICE)
        self.x = torch.tensor([-1.0, -0.1, 0.0, 0.1, 1.0], device=DEVICE)
        self.x_cplx = torch.complex(self.x, self.x * 0.5)

    def test_instantiation(self):
        self.assertIsInstance(self.char_reg, Regularizer)
        with self.assertRaises(ValueError): CharbonnierRegularizer(lambda_reg=-0.1, epsilon=0.01)
        with self.assertRaises(ValueError): CharbonnierRegularizer(lambda_reg=0.1, epsilon=-0.01)

    def test_value(self):
        expected_val = self.lambda_reg * charbonnier_penalty(self.x, self.epsilon)
        torch.testing.assert_close(self.char_reg.value(self.x), expected_val)

    def test_proximal_operator_real_scalar_solver(self):
        # Test the internal scalar solver with a known case
        # Solve y * (1 + gamma_eff / sqrt(y^2 + eps^2)) = v_abs
        v_abs = torch.tensor(0.1, device=DEVICE)
        gamma_eff = self.lambda_reg * 1.0 # steplength=1.0
        
        # If y is small, y * (1 + gamma_eff/eps) = v_abs => y = v_abs / (1+gamma_eff/eps)
        # y_approx_small = 0.1 / (1 + 0.1/0.01) = 0.1 / (1+10) = 0.1/11 = 0.00909
        # If y is large, y * (1 + 0) = v_abs => y = v_abs
        # y_approx_large = 0.1
        # True value should be between these.
        
        solved_y = self.char_reg._solve_charbonnier_prox_scalar(v_abs, gamma_eff)
        # Check if g(solved_y) is close to 0
        g_solved_y = solved_y * (1 + gamma_eff / torch.sqrt(solved_y**2 + self.epsilon**2)) - v_abs
        self.assertAlmostEqual(g_solved_y.item(), 0.0, delta=1e-5)

    def test_proximal_operator_real(self):
        steplength = 1.0
        prox_out = self.char_reg.proximal_operator(self.x, steplength)
        self.assertEqual(prox_out.shape, self.x.shape)
        # For x=0, prox should be 0
        self.assertAlmostEqual(prox_out[2].item(), 0.0, delta=1e-7)
        # Prox result should be smaller in magnitude than x for non-zero x
        self.assertTrue(torch.all(torch.abs(prox_out) <= torch.abs(self.x)))

    def test_proximal_operator_complex(self):
        steplength = 1.0
        prox_out_cplx = self.char_reg.proximal_operator(self.x_cplx, steplength)
        self.assertEqual(prox_out_cplx.shape, self.x_cplx.shape)
        self.assertTrue(prox_out_cplx.is_complex())
        # For x=0, prox should be 0
        self.assertTrue(torch.abs(prox_out_cplx[2]) < 1e-7)
        self.assertTrue(torch.all(torch.abs(prox_out_cplx) <= torch.abs(self.x_cplx) + 1e-7)) # Add tolerance for floating point


if __name__ == '__main__':
    unittest.main()
