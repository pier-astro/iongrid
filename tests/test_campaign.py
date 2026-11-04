"""Exercise the shared campaign entry points and native input preparation."""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iongrid import Grid, condor
from iongrid.cloudy import CloudyPoint
from iongrid.inspect import TableInspector
from iongrid.ogip import write_ogip_table
from iongrid.sampling import cartesian, lhs
from iongrid.sed import table
from astropy.io import fits


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


class CampaignTest(unittest.TestCase):
    def test_sampling_uses_exact_values_and_includes_anchors(self):
        self.assertEqual(cartesian(lognh=[22, 23], vturb=100),
                         [{"lognh": 22, "vturb": 100}, {"lognh": 23, "vturb": 100}])
        bounds = {"vturb": (100, 1000), "logxi": (3, 4)}
        rows = lhs(bounds, samples=4, seed=1, logarithmic=("vturb",))
        self.assertEqual(len(rows), 4)
        for name, limits in bounds.items():
            self.assertEqual((min(row[name] for row in rows), max(row[name] for row in rows)), limits)
        self.assertEqual(len(lhs(bounds, samples=1, anchors=False)), 1)

    def test_cloudy_callable_sed_and_extension(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(lognh=22, logxi=3, vturb=100, logdensity=10, logluminosity=44, gamma=1.8)
            energy = np.array([0.01, 1., 100.])
            grid = Grid.create(root, engine="cloudy", params=[row, row],
                               sed=table(lambda gamma: (energy, energy**-gamma), axes=("gamma",)))
            grid = Grid.open(root)
            grid.extend([row, dict(row, logxi=4)])
            self.assertEqual(grid.status()["total"], 2)
            text = (grid.directory("p000000001") / "p000000001.in").read_text()
            self.assertIn('table SED "incident.sed"', text)
            self.assertNotIn("power law", text)
            self.assertNotIn("", text.splitlines())
            self.assertIn('save overview last "p000000001.ovr"', text)
            self.assertNotIn("# pending", grid.manifest.read_text())
            with self.assertRaisesRegex(ValueError, "SED changed"):
                grid.extend([dict(row, logxi=5)], sed=table(lambda gamma: (energy, energy), axes=("gamma",)))
            self.assertEqual(grid.status()["total"], 2)
            check = root / "standalone"
            CloudyPoint().prepare(check, text="hden 10 log\n")
            self.assertEqual((check / "standalone.in").read_text(), "hden 10 log\n")

    def test_xstar_cluster_retry_collect_and_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "fixture"
            fixture.mkdir()
            make_fixture(fixture)
            binary = root / "xstar"
            binary.write_text(
                '#!/bin/sh\nset -eu\ntest -f "${PFILES%%;*}/xstar.par"\n'
                'printf "%s\\n" "$@" > args.txt\ncp "$FIXTURE"/xout_*.fits .\n'
            )
            binary.chmod(0o755)
            rows = cartesian(lognh=21, logxi=[3, 4], vturb=100, logdensity=10,
                             logluminosity=44, gamma=2, feabund=.8)
            grid = Grid.create(root / "run", engine="xstar", params=rows)
            self.assertEqual(grid.run(executable="/missing-xstar")["failed"], 2)
            submit = condor.write(grid, batch=2, select="failed", executable=binary, python=sys.executable,
                                  text="request_memory = 3G")
            self.assertIn("request_memory = 3G\nqueue 1", submit.read_text())
            source = str(Path(__file__).resolve().parents[1] / "src")
            with patch.dict(os.environ, FIXTURE=str(fixture), PYTHONPATH=source):
                subprocess.run([str(submit.parent / "worker.sh"), "0"], cwd=submit.parent, check=True)
            self.assertEqual(grid.refresh()["completed"], 2)
            self.assertEqual(grid.collect(), {})
            cache = grid.directory("p000000001") / "p000000001.npz"
            with np.load(cache) as data:
                np.testing.assert_allclose(data["energy"], [0.5, 1.0, 2.0])
                np.testing.assert_allclose(data["emt_total"], data["emt_continuum"] + data["emt_lines"])
                self.assertTrue(np.all(data["emt_lines"] >= 0))
            args = (grid.directory("p000000001") / "args.txt").read_text()
            self.assertIn("trad=-1.0", args)
            self.assertIn("feabund=0.8", args)
            packed = grid.pack(root / "tau.fits", key="tau_total", quantity="Optical depth", unit="")
            np.testing.assert_allclose(TableInspector(packed).node(**rows[0]), np.log(2))
            table_path = write_ogip_table(packed, root / "tau.emod", model_name="tau", model_type="exponential")
            with fits.open(table_path) as hdul:
                self.assertEqual(hdul[0].header["REFDIST"], 1.0)
                self.assertEqual(hdul[0].header["REFLUM"], 1e38)
                self.assertEqual(hdul["PARAMETERS"].data["NAME"].tolist(), ["logxi"])
                self.assertEqual(hdul["SPECTRA"].data["PARAMVAL"].shape, (2,))
            grid = Grid.open(grid.root)
            grid.extend([rows[0], dict(rows[0], gamma=1.8)])
            self.assertEqual(grid.status()["total"], 3)
            self.assertEqual(grid.status()["collected"], 2)
            partial = grid.pack(root / "partial.fits", key="tau_total", quantity="Optical depth", unit="")
            with self.assertRaisesRegex(ValueError, "all points"):
                write_ogip_table(partial, root / "partial.emod", model_name="tau", model_type="exponential")
