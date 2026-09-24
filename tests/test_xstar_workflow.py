"""Focused XSTAR workflow tests that never invoke the real simulator."""

import os
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iongrid.xstar.grid import read_manifest, refresh_grid, run_grid, setup_grid, pack_grid_npz
from iongrid.xstar.htcondor import write_condor_submission
from iongrid.xstar.cube import write_fits_cube
from iongrid.ogip import write_ogip_table


def make_fixture(root: Path):
    params = fits.BinTableHDU.from_columns([
        fits.Column(name="parameter", format="16A", array=["rlrad38"]),
        fits.Column(name="value", format="D", array=[2.0]),
    ], name="PARAMETERS")
    energy = np.array([500.0, 1000.0, 2000.0])
    for filename, emission in (("xout_spect1.fits", [3.0, 5.0, 7.0]),
                                ("xout_cont1.fits", [2.0, 5.1, 4.0])):
        spectrum = fits.BinTableHDU.from_columns([
            fits.Column(name="energy", format="D", unit="eV", array=energy),
            fits.Column(name="incident", format="D", array=[10.0, 10.0, 10.0]),
            fits.Column(name="transmitted", format="D", array=[5.0, 5.0, 5.0]),
            fits.Column(name="emit_inward", format="D", array=emission),
            fits.Column(name="emit_outward", format="D", array=emission),
        ], name="XSTAR_SPECTRA")
        fits.HDUList([fits.PrimaryHDU(), params, spectrum]).writeto(root / filename)


class XstarWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture = self.root / "fixture"
        fixture.mkdir()
        make_fixture(fixture)
        self.fixture = fixture
        script = self.root / "fake-xstar"
        script.write_text("#!/bin/sh\nset -eu\nprintf '%s\\n' \"$@\" > args.txt\ncp \"$IONGRID_FIXTURE/xout_spect1.fits\" .\ncp \"$IONGRID_FIXTURE/xout_cont1.fits\" .\n")
        script.chmod(0o755)
        self.binary = str(script)
        self.old_fixture = os.environ.get("IONGRID_FIXTURE")
        os.environ["IONGRID_FIXTURE"] = str(fixture)
        self.addCleanup(self._restore_env)
        self.options = dict(logxi_values=[3.0, 4.0], vturb_values=[100.0],
                            log_nh_values=[22.0], log_density=10.0,
                            luminosity_38=2.0)

    def _restore_env(self):
        if self.old_fixture is None:
            os.environ.pop("IONGRID_FIXTURE", None)
        else:
            os.environ["IONGRID_FIXTURE"] = self.old_fixture

    def test_parallel_local_and_common_products(self):
        manifest = run_grid(self.root / "run", "run", workers=2,
                            xstar_path=self.binary, **self.options)
        points = read_manifest(manifest)
        self.assertEqual([point.status for point in points], ["ok", "ok"])
        for point in points:
            args = (point.dirpath(manifest.parent) / "args.txt").read_text()
            self.assertIn(f"rlogxi={point.logxi}", args)
            self.assertIn(f"column={10**point.log_nh}", args)
        pack_grid_npz(manifest.parent)
        with np.load(points[0].dirpath(manifest.parent) / f"{points[0].point_id}.npz") as data:
            np.testing.assert_allclose(data["energy"], [0.5, 1.0, 2.0])
            np.testing.assert_allclose(data["tau_total"], np.log(2))
            np.testing.assert_allclose(data["emt_total"], data["emt_lines"] + data["emt_continuum"])
            self.assertTrue(np.all(data["emt_lines"] >= 0))
            self.assertEqual(int(data["schema_version"]), 1)
        cube = write_fits_cube(manifest, self.root / "tau.fits", npz_key="tau_total",
                               quantity="TOTAL OPTICAL DEPTH", unit="dimensionless")
        with fits.open(cube) as hdul:
            self.assertEqual(hdul[0].header["ENGINE"], "XSTAR")
            self.assertEqual(hdul["SPECTRA"].data.shape, (2, 1, 1, 3))
        table = write_ogip_table(cube, self.root / "tau.emod", model_name="tau", model_type="exponential")
        with fits.open(table) as hdul:
            self.assertEqual(hdul[0].header["REFDIST"], 1.0)
            self.assertEqual(hdul[0].header["REFLUM"], 1e38)

    def test_htcondor_wrapper_and_refresh(self):
        manifest = setup_grid(self.root / "run", **self.options)
        submit = write_condor_submission(manifest.parent, xstar_path=self.binary,
                                         python_path=sys.executable, batch_size=1)
        self.assertIn("queue 2", submit.read_text())
        wrapper = submit.parent / "xstar_condor_wrapper.sh"
        env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        subprocess.run([str(wrapper), "0"], env=env, check=True, capture_output=True)
        statuses = [pt.status for pt in read_manifest(refresh_grid(manifest.parent))]
        self.assertEqual(statuses.count("ok"), 1)
        self.assertEqual(statuses.count("ready"), 1)
        self.assertTrue((submit.parent / "results" / "batch_000000.csv").exists())

    def test_existing_xstar_outputs_can_be_imported(self):
        manifest = setup_grid(self.root / "run", **self.options)
        pt = read_manifest(manifest)[0]
        for name in ("xout_spect1.fits", "xout_cont1.fits"):
            shutil.copy2(self.fixture / name, pt.dirpath(manifest.parent) / name)
        self.assertEqual(read_manifest(refresh_grid(manifest.parent))[0].status, "ok")
        pack_grid_npz(manifest.parent)

    def test_custom_sed_is_copied_and_recorded(self):
        sed = self.root / "sed.dat"
        sed.write_text("2\n100 1\n200 2\n")
        manifest = setup_grid(self.root / "run", sed_file=sed, **self.options)
        point = read_manifest(manifest)[0]
        directory = point.dirpath(manifest.parent)
        params = json.loads((directory / "xstar-params.json").read_text())
        self.assertEqual(params["spectrum"], "file")
        self.assertEqual(params["spectrum_file"], "sed.dat")
        self.assertEqual(params["spectun"], 0)
        self.assertEqual((directory / "sed.dat").read_text(), sed.read_text())
        sed.write_text("2\n100 3\n200 4\n")
        setup_grid(self.root / "run", sed_file=sed, **self.options)
        self.assertEqual((directory / "sed.dat").read_text(), sed.read_text())
        run_grid(self.root / "run", "run", xstar_path=self.binary,
                 sed_file=sed, **self.options)
        args = (directory / "args.txt").read_text()
        self.assertIn("spectrum=file", args)
        self.assertIn("spectrum_file=sed.dat", args)
        self.assertIn("spectun=0", args)
        sed.write_text("2\n100 5\n200 6\n")
        with self.assertRaisesRegex(ValueError, "SED changed"):
            setup_grid(self.root / "run", sed_file=sed, **self.options)
