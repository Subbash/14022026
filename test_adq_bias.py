import unittest

from adq_bias import compute_bias_codes


class TestComputeBiasCodes(unittest.TestCase):
    def test_example_minus_100mV_at_500mVpp(self):
        self.assertEqual(compute_bias_codes(-100.0, 500.0), -13107)

    def test_zero_bias(self):
        self.assertEqual(compute_bias_codes(0.0, 500.0), 0)

    def test_positive_bias(self):
        self.assertEqual(compute_bias_codes(125.0, 500.0), 16384)

    def test_clamps_to_int16(self):
        self.assertEqual(compute_bias_codes(10000.0, 500.0), 32767)
        self.assertEqual(compute_bias_codes(-10000.0, 500.0), -32768)

    def test_invalid_range(self):
        with self.assertRaises(ValueError):
            compute_bias_codes(10.0, 0.0)


if __name__ == "__main__":
    unittest.main()
