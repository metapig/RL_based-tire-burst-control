import unittest

import numpy as np

from src.utils.generate_burst_force_tables import (
    corrected_full_burst_mz,
    interpolate_tables,
    recover_full_burst_table,
)


class BurstTableGenerationTests(unittest.TestCase):
    def setUp(self):
        self.normal = np.asarray(
            [[0.0, 100.0, 200.0], [1.0, 10.0, 20.0]],
            dtype=np.float64,
        )
        self.full_burst = np.asarray(
            [[0.0, 100.0, 200.0], [1.0, 3.0, 6.0]],
            dtype=np.float64,
        )

    def test_recovers_full_burst_from_intermediate_table(self):
        intermediate = interpolate_tables(
            self.normal, self.full_burst, severity=0.75
        )
        recovered = recover_full_burst_table(
            self.normal, intermediate, intermediate_severity=0.75
        )
        np.testing.assert_allclose(recovered, self.full_burst)

    def test_corrected_mz_preserves_axes_and_scales_values(self):
        corrected = corrected_full_burst_mz(self.normal, scale=0.25)
        np.testing.assert_allclose(corrected[0, :], self.normal[0, :])
        np.testing.assert_allclose(corrected[:, 0], self.normal[:, 0])
        np.testing.assert_allclose(
            corrected[1:, 1:], 0.25 * self.normal[1:, 1:]
        )


if __name__ == "__main__":
    unittest.main()
