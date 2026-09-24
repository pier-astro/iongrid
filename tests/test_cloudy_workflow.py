import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iongrid.cloudy.grid import (
    CloudyPoint,
    cloud_exited_ok,
    expected_files,
    make_local_points,
    parse_exec_time,
    point_id,
    read_manifest,
    write_grid,
    run_grid,
    refresh_grid,
    pack_grid_npz,
)


from iongrid.cloudy.cube import write_fits_cube
from iongrid.cloudy.htcondor import run_condor_batch, write_condor_submission
from iongrid.cloudy.spectra import read_continuum, read_optical_depth, read_slab_continuum
from iongrid.ogip import write_ogip_table

class CloudyUtilsTest(unittest.TestCase):
    def test_cube_rejects_unknown_node_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            point = read_manifest(manifest)[0]
            point.filepath("out", tmp).write_text("Cloudy exited OK\n", encoding="utf-8")
            np.savez_compressed(
                point.filepath("npz", tmp),
                schema_version=2,
                energy=np.array([0.5, 1.0]),
                tau_absorption=np.array([0.1, 0.2]),
            )
            refresh_grid(tmp)
            with self.assertRaisesRegex(ValueError, "Unsupported iongrid schema version"):
                write_fits_cube(
                    manifest, Path(tmp) / "tau.fits",
                    npz_key="tau_absorption", quantity="optical depth", unit="dimensionless",
                )

    def test_write_grid_manifest_and_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            self.assertTrue(Path(manifest).exists())
            points = read_manifest(manifest)
            self.assertEqual(len(points), 1)

            # Read manifest CSV
            with open(manifest, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["logxi"]), 3.0)
            self.assertEqual(float(rows[0]["vturb"]), 100.0)
            self.assertEqual(float(rows[0]["log_nh"]), 21.0)

            first = points[0]
            first_input = expected_files(first, Path(tmp))["input"]
            self.assertTrue(first_input.exists())
            self.assertEqual(first_input.parent.name, rows[0]["point_id"])
            text = first_input.read_text(encoding="utf-8")
            self.assertIn("xi 3 log", text)
            self.assertIn("radius 16", text)
            self.assertNotIn("thickness 7 log", text)
            self.assertIn("stop column density 21 log", text)
            self.assertIn('save optical depths last "', text)
            self.assertNotIn("stop zone 1", text)
            self.assertNotIn("set dr -5", text)
            self.assertNotIn("set save resolving power 3000 absorption", text)
            self.assertIn("set continuum resolution 4", text)
            files = expected_files(first, Path(tmp))
            self.assertEqual(set(files), {"input", "stdout", "overview", "species_abundance", "line_list", "line_labels", "continuum", "raw", "optical_depth", "fine_optical_depth"})
            self.assertTrue((Path(tmp) / "cloudy-template.in").exists())

    def test_write_grid_accepts_custom_template(self):
        with tempfile.TemporaryDirectory() as tmp:
            template_path = Path(tmp) / "custom-template.in"
            template_path.write_text(
                "title {title}\n"
                "xi {logxi} log\n"
                "hden {log_density} log\n"
                "stop column density {log_nh} log\n"
                "save continuum last \"{continuum_file}\" units kev\n"
                "save optical depths last \"{optical_depth_file}\" units kev\n",
                encoding="utf-8"
            )
            manifest = write_grid(
                tmp,
                logxi_values=[3.0],
                vturb_values=[100.0],
                log_nh_values=[21.0],
                template_path=template_path,
            )
            first = read_manifest(manifest)[0]
            first_input = expected_files(first, Path(tmp))["input"]
            text = first_input.read_text(encoding="utf-8")
            self.assertIn(f'save optical depths last "{first.point_id}.opt" units kev', text)

    def test_write_grid_custom_sed(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Custom sed_slope
            manifest = write_grid(
                tmp,
                logxi_values=[3.0],
                vturb_values=[100.0],
                log_nh_values=[21.0],
                sed_slope=-1.2,
            )
            first = read_manifest(manifest)[0]
            first_input = expected_files(first, Path(tmp))["input"]
            text = first_input.read_text(encoding="utf-8")
            self.assertIn("power law, slope=-1.2 hi cut 1000 Ryd low cut 1 Ryd", text)

        with tempfile.TemporaryDirectory() as tmp:
            # 2. Custom sed_file
            sed_file = Path(tmp) / "custom_sed.sed"
            sed_file.write_text("0.1 1.0\n1.0 1.0\n", encoding="utf-8")
            manifest = write_grid(
                tmp,
                logxi_values=[3.0],
                vturb_values=[100.0],
                log_nh_values=[21.0],
                sed_file=sed_file,
            )
            first = read_manifest(manifest)[0]
            first_input = expected_files(first, Path(tmp))["input"]
            text = first_input.read_text(encoding="utf-8")
            self.assertIn('table SED "custom_sed.sed"', text)

        with tempfile.TemporaryDirectory() as tmp:
            # 3. Both sed_slope and sed_file provided (sed_file should override)
            sed_file = Path(tmp) / "custom_sed.sed"
            sed_file.write_text("0.1 1.0\n1.0 1.0\n", encoding="utf-8")
            manifest = write_grid(
                tmp,
                logxi_values=[3.0],
                vturb_values=[100.0],
                log_nh_values=[21.0],
                sed_slope=-1.2,
                sed_file=sed_file,
            )
            first = read_manifest(manifest)[0]
            first_input = expected_files(first, Path(tmp))["input"]
            text = first_input.read_text(encoding="utf-8")
            self.assertIn('table SED "custom_sed.sed"', text)
            self.assertNotIn("power law", text)

    def test_point_id_is_compact(self):
        self.assertEqual(point_id(3.875, 325.0, 22.5), "xi3p875_vt0325_nh22p500")

    def test_default_axes_are_5x5x4(self):
        points = make_local_points()
        logxi = sorted({point.logxi for point in points})
        vturb = sorted({point.vturb for point in points})
        log_nh = sorted({point.log_nh for point in points})
        self.assertEqual(len(logxi), 5)
        self.assertEqual(len(vturb), 5)
        self.assertEqual(len(log_nh), 4)

    def test_cloud_exited_ok_handles_various_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test.out"

            # 1. Success case
            f.write_text("ExecTime(s) 12.34\nCloudy exited OK\n", encoding="utf-8")
            self.assertTrue(cloud_exited_ok(f))

            # 2. Fail case
            f.write_text("An error occurred during convergence...\n", encoding="utf-8")
            self.assertFalse(cloud_exited_ok(f))

    def test_parse_exec_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "test.out"

            # 1. Matching case
            f.write_text("ExecTime(s) 12.3\nCloudy exited OK\n", encoding="utf-8")
            self.assertAlmostEqual(parse_exec_time(f), 12.3)

            # 2. Non-matching case
            f.write_text("No timing here", encoding="utf-8")
            self.assertIsNone(parse_exec_time(f))

    def test_run_grid_in_dry_run_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0, 4.0], vturb_values=[100.0], log_nh_values=[21.0, 22.0], log_density=11.0)
            self.assertTrue(Path(manifest).exists())
            self.assertTrue((Path(tmp) / "cloudy-template.in").exists())
            points = read_manifest(manifest)
            self.assertEqual(len(points), 4)
            self.assertEqual(points[0].log_density, 11.0)
            self.assertEqual(points[0].log_nh, 21.0)

            manifest2 = run_grid(tmp, action="refresh")
            self.assertEqual(len(read_manifest(manifest2)), 4)

    def test_write_condor_submission_batches_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0, 4.0], vturb_values=[100.0], log_nh_values=[21.0, 22.0])
            before_ids = [pt.point_id for pt in read_manifest(manifest)]
            submit_file = write_condor_submission(tmp, cloudy_path="/bin/cloudy", batch_size=2, seed=7)
            after_ids = [pt.point_id for pt in read_manifest(manifest)]

            self.assertTrue(submit_file.exists())
            submit_text = submit_file.read_text(encoding="utf-8")
            self.assertIn("queue 2", submit_text)
            self.assertIn("request_memory = 3G", submit_text)
            self.assertEqual(after_ids, before_ids)

            batch_files = sorted((Path(tmp) / "htc_submit" / "batches").glob("batch_*.txt"))
            self.assertEqual(len(batch_files), 2)
            batched_ids = []
            for batch_file in batch_files:
                batched_ids.extend(batch_file.read_text(encoding="utf-8").splitlines())
            manifest_ids = [pt.point_id for pt in read_manifest(manifest)]
            self.assertEqual(sorted(batched_ids), sorted(manifest_ids))

            wrapper = Path(tmp) / "htc_submit" / "cloudy_condor_wrapper.sh"
            self.assertTrue(wrapper.exists())
            self.assertIn("from iongrid.cloudy.htcondor import run_condor_batch", wrapper.read_text(encoding="utf-8"))

    def test_write_condor_submission_resolves_python_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            bin_dir = Path(tmp) / "bin"
            bin_dir.mkdir()
            fake_python = bin_dir / "python3"
            fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
            fake_python.chmod(0o755)

            submit_file = write_condor_submission(tmp, cloudy_path="/bin/cloudy", batch_size=1, seed=7, python_path=str(bin_dir))
            wrapper = Path(tmp) / "htc_submit" / "cloudy_condor_wrapper.sh"
            wrapper_text = wrapper.read_text(encoding="utf-8")

            self.assertTrue(submit_file.exists())
            self.assertIn(f"PYTHON_BIN={str(fake_python)!r}", wrapper_text)

    def test_run_condor_batch_runs_points_without_shared_manifest_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0, 4.0], vturb_values=[100.0], log_nh_values=[21.0])
            points = read_manifest(manifest)
            batch_file = Path(tmp) / "batch.txt"
            batch_file.write_text("\n".join(pt.point_id for pt in points) + "\n", encoding="utf-8")

            fake_cloudy = Path(tmp) / "fake_cloudy.sh"
            fake_cloudy.write_text(
                "#!/bin/sh\n"
                "cat >/dev/null\n"
                "echo 'ExecTime(s) 1.25'\n"
                "echo 'Cloudy exited OK'\n",
                encoding="utf-8",
            )
            fake_cloudy.chmod(0o755)

            results = run_condor_batch(tmp, batch_file, cloudy_path=str(fake_cloudy), make_binaries=False)
            self.assertEqual(Path(results).parent, Path(tmp) / "results")
            result_points = read_manifest(results)
            self.assertEqual([pt.status for pt in result_points], ["ok", "ok"])
            self.assertAlmostEqual(result_points[0].exec_time_s, 1.25)
            self.assertEqual([pt.status for pt in read_manifest(manifest)], ["ready", "ready"])
            for pt in points:
                self.assertTrue(pt.filepath("out", tmp).exists())

    def test_write_fits_cubes_from_compiled_nodes(self):
        from astropy.io import fits

        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0, 4.0], vturb_values=[100.0], log_nh_values=[21.0, 22.0])
            points = read_manifest(manifest)
            energy = np.array([0.5, 1.0, 2.0])

            for pt in points:
                pt.filepath("out", tmp).write_text("ExecTime(s) 1.0\nCloudy exited OK\n", encoding="utf-8")
                value = pt.logxi + pt.log_nh + pt.vturb / 1000.0
                np.savez_compressed(
                    pt.dirpath(tmp) / f"{pt.point_id}.npz",
                    energy=energy,
                    tau_absorption=np.full_like(energy, value),
                    tau_scattering=np.full_like(energy, 0.1 * value),
                    tau_total=np.full_like(energy, 1.1 * value),
                    emt_continuum=np.full_like(energy, 2.0 * value),
                    emt_lines=np.full_like(energy, 3.0 * value),
                    emt_total=np.full_like(energy, 5.0 * value),
                )
            refresh_grid(tmp, make_binaries=False)

            absorption_path = write_fits_cube(
                manifest,
                Path(tmp) / "abs.fits",
                npz_key="tau_scattering",
                quantity="scattering optical depth",
                unit="dimensionless",
            )
            emission_path = write_fits_cube(
                manifest,
                Path(tmp) / "emt.fits",
                npz_key="emt_lines",
                quantity="emission-line spectrum before geometry convolution",
                unit="ph cm-2 s-1",
            )

            with fits.open(absorption_path) as hdul:
                self.assertEqual(hdul[0].header["IGFMTVER"], 1)
                self.assertEqual(hdul[0].header["ENGINE"], "CLOUDY")
                self.assertEqual(hdul[0].header["NPZKEY"], "tau_scattering")
                self.assertEqual(hdul[0].header["QUANTITY"], "scattering optical depth")
                self.assertEqual([hdu.name for hdu in hdul], ["PRIMARY", "SPECTRA", "GRID"])
                self.assertEqual(hdul["SPECTRA"].data.shape, (2, 1, 2, 3))
                self.assertEqual(hdul["SPECTRA"].header["BUNIT"], "dimensionless")
                np.testing.assert_allclose(hdul["GRID"].data["ENERGY"][0], energy)
                np.testing.assert_allclose(hdul["GRID"].data["LOGXI"][0], [3.0, 4.0])

            with fits.open(emission_path) as hdul:
                self.assertEqual(hdul[0].header["NPZKEY"], "emt_lines")
                self.assertEqual(hdul["SPECTRA"].data.shape, (2, 1, 2, 3))
                self.assertEqual(hdul["SPECTRA"].header["BUNIT"], "ph cm-2 s-1")

    def test_write_ogip_table_from_fits_cube(self):
        from astropy.io import fits

        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0, 4.0], vturb_values=[100.0], log_nh_values=[21.0])
            points = read_manifest(manifest)
            energy = np.array([0.5, 1.0, 2.0])

            for pt in points:
                pt.filepath("out", tmp).write_text("ExecTime(s) 1.0\nCloudy exited OK\n", encoding="utf-8")
                value = pt.logxi + pt.vturb / 1000.0 + pt.log_nh
                np.savez_compressed(
                    pt.dirpath(tmp) / f"{pt.point_id}.npz",
                    energy=energy,
                    tau_total=np.full_like(energy, value),
                )
            refresh_grid(tmp, make_binaries=False)

            cube_path = write_fits_cube(
                manifest,
                Path(tmp) / "tau.fits",
                npz_key="tau_total",
                quantity="total optical depth",
                unit="dimensionless",
            )
            table_path = write_ogip_table(
                cube_path,
                Path(tmp) / "tau.mod",
                model_name="iongrid_tau",
                model_type="exponential",
            )

            with fits.open(table_path) as hdul:
                self.assertEqual(hdul[0].header["IGFMTVER"], 1)
                self.assertEqual([hdu.name for hdu in hdul], ["PRIMARY", "PARAMETERS", "ENERGIES", "SPECTRA"])
                self.assertEqual(hdul[0].header["MODLNAME"], "iongrid_tau")
                self.assertFalse(hdul[0].header["ADDMODEL"])
                self.assertEqual(hdul[0].header["XFXP0001"], "exponential")
                self.assertEqual(hdul["PARAMETERS"].header["NINTPARM"], 1)
                self.assertEqual(len(hdul["PARAMETERS"].data), 1)
                self.assertEqual(hdul["PARAMETERS"].data["NAME"][0].strip(), "logxi")
                self.assertEqual(hdul["PARAMETERS"].data["METHOD"][0], 0)
                self.assertEqual(len(hdul["ENERGIES"].data), 3)
                self.assertEqual(hdul["SPECTRA"].data["PARAMVAL"].shape, (2,))
                self.assertEqual(hdul["SPECTRA"].data["INTPSPEC"].shape, (2, 3))

            table_path_mult = write_ogip_table(
                cube_path,
                Path(tmp) / "trans.mod",
                model_name="iongrid_trans",
                model_type="multiplicative",
            )

            with fits.open(table_path_mult) as hdul:
                self.assertEqual([hdu.name for hdu in hdul], ["PRIMARY", "PARAMETERS", "ENERGIES", "SPECTRA"])
                self.assertEqual(hdul[0].header["MODLNAME"], "iongrid_trans")
                self.assertFalse(hdul[0].header["ADDMODEL"])
                self.assertEqual(hdul[0].header["XFXP0001"], "multiplicative")
                self.assertEqual(hdul["PARAMETERS"].header["NINTPARM"], 1)
                self.assertEqual(len(hdul["PARAMETERS"].data), 1)
                self.assertEqual(hdul["PARAMETERS"].data["NAME"][0].strip(), "logxi")
                self.assertEqual(hdul["PARAMETERS"].data["METHOD"][0], 0)
                self.assertEqual(len(hdul["ENERGIES"].data), 3)
                self.assertEqual(hdul["SPECTRA"].data["PARAMVAL"].shape, (2,))
                self.assertEqual(hdul["SPECTRA"].data["INTPSPEC"].shape, (2, 3))
                # First point tau was: logxi + vturb/1000 + log_nh = 3.0 + 0.1 + 21.0 = 24.1
                # Second point tau was: 4.0 + 0.1 + 21.0 = 25.1
                np.testing.assert_allclose(
                    hdul["SPECTRA"].data["INTPSPEC"][0],
                    np.exp(-24.1),
                    rtol=1e-6
                )
                np.testing.assert_allclose(
                    hdul["SPECTRA"].data["INTPSPEC"][1],
                    np.exp(-25.1),
                    rtol=1e-6
                )

    def test_read_optical_depth(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            point = read_manifest(manifest)[0]
            opt = expected_files(point, Path(tmp))["optical_depth"]
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
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            point = read_manifest(manifest)[0]
            con = expected_files(point, Path(tmp))["continuum"]
            con.write_text(
                "#Cont  nu\tincident\ttrans\tDiffOut\tDiffInward\treflc\ttotal\treflin\toutlin\tlineID\tcont\tnLine\n"
                "1.0\t100.0\t50.0\t5.0\t3.0\t5.0\t10.0\t1.0\t2.0\tHe 2\t\t300.0\n"
                "2.0\t100.0\t50.0\t20.0\t6.0\t10.0\t20.0\t2.0\t8.0\tH  1\t\t600.0\n",
                encoding="utf-8"
            )
            raw = expected_files(point, Path(tmp))["raw"]
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
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            point = read_manifest(manifest)[0]
            expected_files(point, Path(tmp))["input"].parent.mkdir(parents=True, exist_ok=True)
            con_file = expected_files(point, Path(tmp))["input"].parent / f"{point.point_id}.con"
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

        # 3. Verify 32-bit FITS cube writing
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            from iongrid.cloudy.grid import read_manifest
            point = read_manifest(manifest)[0]
            p_dir = expected_files(point, Path(tmp))["input"].parent
            p_dir.mkdir(parents=True, exist_ok=True)

            # Mock files
            val = 3.0
            point.filepath("out", tmp).write_text("ExecTime(s) 1.0\nCloudy exited OK\n", encoding="utf-8")
            expected_files(point, Path(tmp))["overview"].write_text(
                "#depth Te Htot hden eden\n1e3 5e5 1.0 1e12 1e12\n", encoding="utf-8"
            )
            expected_files(point, Path(tmp))["optical_depth"].write_text(
                f"#energy total absorption scattering\n0.0007 {1.5*val} {val} {0.5*val}\n102.5 {1.5*val} {val} {0.5*val}\n",
                encoding="utf-8"
            )
            expected_files(point, Path(tmp))["fine_optical_depth"].write_text(
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
            expected_files(point, Path(tmp))["continuum"].write_text(con_text, encoding="utf-8")
            raw_text = f"#Cont nu incident trans DiffOut DiffInward ConRefIncid ConEmitReflec\n0.0007\t100.0\t0.0\t0.0\t0.0\t0.0\t0.0\t1.0\n102.5\t100.0\t0.0\t0.0\t0.0\t0.0\t0.0\t1.0\n"
            expected_files(point, Path(tmp))["raw"].write_text(raw_text, encoding="utf-8")
            expected_files(point, Path(tmp))["line_labels"].write_text("#index\tlabel\tcomment\n", encoding="utf-8")

            # Compile
            from iongrid.cloudy.spectra import compile_node
            compile_node(p_dir)
            with np.load(p_dir / f"{point.point_id}.npz") as data:
                self.assertEqual(int(data["schema_version"]), 1)
            refresh_grid(tmp, make_binaries=False)

            # Write FITS cube
            fits_path = Path(tmp) / "test_cube.fits"
            write_fits_cube(manifest, fits_path, npz_key="tau_absorption", quantity="absorption", unit="dimensionless")

            # Read and verify BITPIX and data type
            from astropy.io import fits
            with fits.open(fits_path) as hdul:
                self.assertIn("SPECTRA", hdul)
                spectra_hdu = hdul["SPECTRA"]
                self.assertEqual(spectra_hdu.header["BITPIX"], -32)
                self.assertTrue(np.issubdtype(spectra_hdu.data.dtype, np.float32))
                self.assertEqual(spectra_hdu.data.shape, (1, 1, 1, 7720))

    def test_grid_extension_and_remapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 1. Setup a grid with a single point
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            points = read_manifest(manifest)
            self.assertEqual(len(points), 1)

            point = points[0]
            p_dir = expected_files(point, Path(tmp))["input"].parent
            p_dir.mkdir(parents=True, exist_ok=True)
            point.filepath("out", tmp).write_text("ExecTime(s) 1.2\nCloudy exited OK\n", encoding="utf-8")

            # Make sure refresh marks it as ok
            refresh_grid(tmp, make_binaries=False)

            # Verify status in manifest is 'ok'
            points = read_manifest(manifest)
            self.assertEqual(points[0].status, "ok")

            # 2. Now extend the grid by adding a new logxi_value
            # We keep the old point logxi=3.0, and add logxi=4.0
            manifest = write_grid(tmp, logxi_values=[3.0, 4.0], vturb_values=[100.0], log_nh_values=[21.0])
            points = read_manifest(manifest)
            self.assertEqual(len(points), 2)

            # Verify that the first point has kept its 'ok' status and exec time, while the new one is 'ready'
            pt3, pt4 = (points[0], points[1]) if points[0].logxi == 3.0 else (points[1], points[0])
            self.assertEqual(pt3.status, "ok")
            self.assertAlmostEqual(pt3.exec_time_s, 1.2)
            self.assertEqual(pt4.status, "ready")

            # 3. Now let's test point remapping/renaming
            # Clear standard directory if it exists to test renaming path
            std_dir = Path(tmp) / "points" / "xi3p000_vt0100_nh21p000"
            if std_dir.exists():
                import shutil
                shutil.rmtree(std_dir)

            # We will manually create a mock old manifest with a different ID style for the same point
            # Say we have a point with the same parameters (3.0, 100.0, 21.0), but with ID 'old_xi3_vt100_nh21'
            old_pid = "old_xi3_vt100_nh21"
            old_dir = Path(tmp) / "points" / old_pid
            old_dir.mkdir(parents=True, exist_ok=True)

            # Mock the stdout file for the old point
            old_out = old_dir / f"{old_pid}.out"
            old_out.write_text("ExecTime(s) 2.5\nCloudy exited OK\n", encoding="utf-8")

            # Update the manifest manually to contain only this old point
            mock_point = CloudyPoint(
                point_id=old_pid,
                logxi=3.0,
                vturb=100.0,
                log_density=10.0,
                log_nh=21.0,
                status="ok",
                exec_time_s=2.5
            )
            from iongrid.cloudy.grid import write_manifest
            write_manifest([mock_point], manifest)

            # Verify mock setup
            points = read_manifest(manifest)
            self.assertEqual(points[0].point_id, old_pid)
            self.assertEqual(points[0].status, "ok")

            # Now run write_grid again, which should generate the standard name 'xi3p000_vt0100_nh21p000'
            # and it should match the old point physically, rename the directory and the files, and update the status
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            points = read_manifest(manifest)
            self.assertEqual(len(points), 1)

            # The point ID should be renamed to standard
            std_pid = "xi3p000_vt0100_nh21p000"
            self.assertEqual(points[0].point_id, std_pid)
            self.assertEqual(points[0].status, "ok")
            self.assertAlmostEqual(points[0].exec_time_s, 2.5)

            # Verify the directory and files were renamed on disk
            new_dir = Path(tmp) / "points" / std_pid
            self.assertTrue(new_dir.exists())
            self.assertFalse(old_dir.exists())

            # The stdout file inside should also be renamed
            self.assertTrue((new_dir / f"{std_pid}.out").exists())
            self.assertFalse((new_dir / f"{old_pid}.out").exists())

    def test_refresh_and_packnpz_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = write_grid(tmp, logxi_values=[3.0], vturb_values=[100.0], log_nh_values=[21.0])
            point = read_manifest(manifest)[0]
            out_file = expected_files(point, Path(tmp))["stdout"]
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("ExecTime(s) 1.5\nCloudy exited OK\n", encoding="utf-8")

            # Mock optical depth file needed for compile_node
            expected_files(point, Path(tmp))["overview"].write_text("#depth Te Htot hden eden\n1e3 5e5 1.0 1e12 1e12\n", encoding="utf-8")
            expected_files(point, Path(tmp))["optical_depth"].write_text("#energy total absorption scattering\n0.0007 1.5 1.0 0.5\n102.5 1.5 1.0 0.5\n", encoding="utf-8")
            expected_files(point, Path(tmp))["fine_optical_depth"].write_text("#energy fine\n0.0007 1.0\n102.5 1.0\n", encoding="utf-8")
            expected_files(point, Path(tmp))["continuum"].write_text(
                "#Cont nu incident trans DiffOut DiffInward reflc total reflin outlin lineID cont nLine\n"
                "0.0007\t100.0\t1.0\t1.0\t1.0\t0.0\t0.0\t1.0\t1.0\t\t\t\n"
                "102.5\t100.0\t1.0\t1.0\t1.0\t0.0\t0.0\t1.0\t1.0\t\t\t\n",
                encoding="utf-8"
            )
            expected_files(point, Path(tmp))["raw"].write_text(
                "#Cont nu incident trans DiffOut DiffInward ConRefIncid ConEmitReflec\n"
                "0.0007\t100.0\t0.0\t0.0\t0.0\t0.0\t0.0\t1.0\n"
                "102.5\t100.0\t0.0\t0.0\t0.0\t0.0\t0.0\t1.0\n",
                encoding="utf-8"
            )
            expected_files(point, Path(tmp))["line_labels"].write_text("#index\tlabel\tcomment\n", encoding="utf-8")

            npz_file = point.dirpath(tmp) / f"{point.point_id}.npz"
            self.assertFalse(npz_file.exists())

            # 1. Refresh grid should update status in manifest but NOT create npz file
            refresh_grid(tmp, make_binaries=False)
            pts = read_manifest(manifest)
            self.assertEqual(pts[0].status, "ok")
            self.assertAlmostEqual(pts[0].exec_time_s, 1.5)
            self.assertFalse(npz_file.exists())

            # 2. Pack npz should create npz file
            pack_grid_npz(tmp)
            self.assertTrue(npz_file.exists())

            # 3. Test run_grid action="refresh" and action="packnpz"
            npz_file.unlink()
            self.assertFalse(npz_file.exists())

            run_grid(tmp, action="refresh")
            self.assertFalse(npz_file.exists())

            run_grid(tmp, action="packnpz")
            self.assertTrue(npz_file.exists())


if __name__ == "__main__":
    unittest.main()
