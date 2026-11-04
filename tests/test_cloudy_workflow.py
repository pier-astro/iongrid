"""CLOUDY readers, numerical binning, and collection into a FITS table."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iongrid import Grid
from iongrid.cloudy.spectra import read_continuum, read_optical_depth, read_slab_continuum


class CloudyUtilsTest(unittest.TestCase):
    def test_read_optical_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            opt = Path(tmp) / "sample.opt"
            opt.write_text(
                "#energy/keV total absorption scattering\n"
                "1.0 1.5 1.0 0.5\n"
                "2.0 3.0 2.0 1.0\n",
                encoding="utf-8"
            )
            energy, total, abs_depth, scat_depth = read_optical_depth(opt)
            np.testing.assert_allclose(energy, [1.0, 2.0])
            np.testing.assert_allclose(total, [1.5, 3.0])
            np.testing.assert_allclose(abs_depth, [1.0, 2.0])
            np.testing.assert_allclose(scat_depth, [0.5, 1.0])

    def test_read_slab_continuum(self):
        with tempfile.TemporaryDirectory() as tmp:
            con = Path(tmp) / "sample.con"
            con.write_text(
                "#Cont  nu\tincident\ttrans\tDiffOut\tDiffInward\treflc\ttotal\treflin\toutlin\tlineID\tcont\tnLine\n"
                "1.0\t100.0\t50.0\t5.0\t3.0\t5.0\t10.0\t1.0\t2.0\tHe 2\t\t300.0\n"
                "2.0\t100.0\t50.0\t20.0\t6.0\t10.0\t20.0\t2.0\t8.0\tH  1\t\t600.0\n",
                encoding="utf-8"
            )
            raw = Path(tmp) / "sample.raw"
            raw.write_text(
                "#Cont nu incident trans DiffOut DiffInward ConRefIncid ConEmitReflec\n"
                "1.0\t100.0\t50.0\t5.0\t3.0\t0.0\t1.0\n"
                "2.0\t100.0\t50.0\t20.0\t6.0\t1.0\t1.0\n",
                encoding="utf-8"
            )
            energy, continuum = read_slab_continuum(con, raw)
            np.testing.assert_allclose(energy, [1.0, 2.0])
            np.testing.assert_allclose(continuum, [10.0, 12.5])

    def test_read_continuum(self):
        with tempfile.TemporaryDirectory() as tmp:
            con_file = Path(tmp) / "sample.con"
            con_file.write_text(
                "#Cont nu incident trans DiffOut net trans reflc total reflin outlin lineID cont nLine\n"
                "1.0 10.0 5.0\n"
                "2.0 20.0 10.0\n",
                encoding="utf-8"
            )
            energy, incident, transmitted = read_continuum(con_file)
            np.testing.assert_allclose(energy, [1.0, 2.0])
            np.testing.assert_allclose(incident, [10.0, 20.0])
            np.testing.assert_allclose(transmitted, [5.0, 10.0])

    def test_downsampling_and_32bit_fits(self):
        # 1. Verify ENERGY_GRID properties
        from iongrid.cloudy.spectra import ENERGY_GRID
        self.assertEqual(ENERGY_GRID.dtype, np.float32)
        self.assertEqual(len(ENERGY_GRID), 7720)
        self.assertAlmostEqual(ENERGY_GRID[0], 0.001, places=6)
        self.assertAlmostEqual(ENERGY_GRID[-1], 100.0, places=6)

        # 2. Verify bin_average conservation
        from iongrid.cloudy.spectra import bin_average, bin_edges
        x_src = np.linspace(0.1, 10.0, 1000)
        # Create a sharp box signal (representing a line or feature)
        y_src = np.zeros_like(x_src)
        y_src[(x_src >= 4.0) & (x_src <= 6.0)] = 5.0

        total_src = np.sum(y_src * np.diff(bin_edges(x_src)))

        x_dst = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        y_dst = bin_average(x_src, y_src, x_dst)

        total_dst = np.sum(y_dst * np.diff(bin_edges(x_dst)))

        # Total flux must be conserved within numeric precision
        self.assertAlmostEqual(total_dst, total_src, places=4)

        # Grid-density transitions must not change a constant spectrum.
        uneven_src = np.array([0.5, 0.7, 0.9, 1.0, 1.01, 1.02, 2.0, 4.0])
        uneven_dst = np.array([0.6, 0.95, 1.005, 1.015, 1.5, 3.0])
        averaged = bin_average(uneven_src, np.full_like(uneven_src, 3.0), uneven_dst)
        np.testing.assert_allclose(averaged, 3.0, rtol=0.0, atol=1e-12)

        # 3. Verify 32-bit FITS table writing
        with tempfile.TemporaryDirectory() as tmp:
            grid = Grid.create(tmp, engine="cloudy", params=[
                dict(logxi=3, vturb=100, lognh=21, logdensity=10, logluminosity=48)
            ])
            pid = "p000000001"
            p_dir = grid.directory(pid)

            # Mock files
            val = 3.0
            (p_dir / f"{pid}.out").write_text("ExecTime(s) 1.0\nCloudy exited OK\n", encoding="utf-8")
            (p_dir / f"{pid}.ovr").write_text(
                "#depth Te Htot hden eden\n1e3 5e5 1.0 1e12 1e12\n", encoding="utf-8"
            )
            (p_dir / f"{pid}.opt").write_text(
                f"#energy total absorption scattering\n0.0007 {1.5*val} {val} {0.5*val}\n102.5 {1.5*val} {val} {0.5*val}\n",
                encoding="utf-8"
            )
            (p_dir / f"{pid}.fopt").write_text(
                f"#energy fine\n0.0007 {val}\n102.5 {val}\n", encoding="utf-8"
            )
            total1 = 2.0 * val
            reflin1 = val
            outlin1 = val
            trans1 = 100.0 * np.exp(-1.5 * val)
            con_text = (
                f"#Cont nu incident trans DiffOut DiffInward reflc total reflin outlin lineID cont nLine\n"
                f"0.0007\t100.0\t{trans1:.12e}\t{total1*0.0007:.12e}\t{total1:.6f}\t0.0\t0.0\t{reflin1:.6f}\t{outlin1*0.0007:.12e}\t\t\t\n"
                f"102.5\t100.0\t{trans1:.12e}\t{total1*102.5:.12e}\t{total1:.6f}\t0.0\t0.0\t{reflin1:.6f}\t{outlin1*102.5:.12e}\t\t\t\n"
            )
            (p_dir / f"{pid}.con").write_text(con_text, encoding="utf-8")
            raw_text = f"#Cont nu incident trans DiffOut DiffInward ConRefIncid ConEmitReflec\n0.0007\t100.0\t0.0\t0.0\t0.0\t0.0\t0.0\t1.0\n102.5\t100.0\t0.0\t0.0\t0.0\t0.0\t0.0\t1.0\n"
            (p_dir / f"{pid}.raw").write_text(raw_text, encoding="utf-8")
            (p_dir / f"{pid}.llab").write_text("#index\tlabel\tcomment\n", encoding="utf-8")

            # Compile
            (p_dir / "run.json").write_text('{"status": "completed"}')
            self.assertEqual(grid.collect(), {})
            with np.load(p_dir / f"{pid}.npz") as data:
                self.assertEqual(int(data["schema_version"]), 1)

            # Write FITS table
            fits_path = Path(tmp) / "test_cube.fits"
            grid.pack(fits_path, key="tau_absorption", quantity="absorption", unit="dimensionless")

            # Read and verify BITPIX and data type
            from astropy.io import fits
            with fits.open(fits_path) as hdul:
                self.assertIn("SPECTRA", hdul)
                spectra_hdu = hdul["SPECTRA"]
                self.assertEqual(spectra_hdu.header["BITPIX"], -32)
                self.assertTrue(np.issubdtype(spectra_hdu.data.dtype, np.float32))
                self.assertEqual(spectra_hdu.data.shape, (1, 7720))
