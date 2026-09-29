"""Both CLOUDY brightness cases must export the same full-shell photon flux."""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from iongrid.cloudy.spectra import compile_node


class CloudyEmissionUnitsTest(unittest.TestCase):
    def test_intensity_and_luminosity_cases_agree(self):
        radius_cm = 1e16
        energy = np.array([0.5, 1.0, 2.0])
        observed = []
        with tempfile.TemporaryDirectory() as tmp:
            for mode, scale in (("intensity", 1.0), ("luminosity", 4 * np.pi * radius_cm**2)):
                point_id = "xi3p000_vt0000_nh21p000"
                directory = Path(tmp) / mode / point_id
                directory.mkdir(parents=True)
                command = "intensity 1 range 1 to 1000 Ryd" if mode == "intensity" else "table agn"
                (directory / f"{point_id}.in").write_text(
                    f"{command}\nhden 10 log\nradius 16.0\n", encoding="utf-8"
                )
                unit_label = "Intensity (erg/s/cm^2)." if mode == "intensity" else "Luminosity (erg/s) emitted by a shell with full coverage."
                (directory / f"{point_id}.out").write_text(
                    f"Emission Line Spectrum.\n    {unit_label}\nCloudy exited OK\n", encoding="utf-8"
                )
                (directory / f"{point_id}.ovr").write_text("1 100000\n", encoding="utf-8")
                (directory / f"{point_id}.opt").write_text("0.0007 0 0 0\n102.5 0 0 0\n", encoding="utf-8")
                (directory / f"{point_id}.con").write_text(
                    "\n".join(f"{e} 1 1 {e * scale} 0 0 0" for e in (0.0007, 102.5)) + "\n",
                    encoding="utf-8",
                )
                (directory / f"{point_id}.raw").write_text(
                    "0.0007 0 0 0 0 0 0\n102.5 0 0 0 0 0 0\n", encoding="utf-8"
                )
                compile_node(directory, energy_grid=energy)
                with np.load(directory / f"{point_id}.npz") as node:
                    observed.append(node["emt_continuum"].copy())
                    if mode == "luminosity":
                        self.assertEqual(node["incident"].dtype, np.float64)
        np.testing.assert_allclose(observed[0], observed[1], rtol=1e-6)
        self.assertGreater(observed[0].sum(), 0)


if __name__ == "__main__":
    unittest.main()
