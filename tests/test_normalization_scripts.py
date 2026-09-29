"""Check the two editable example recipes against their table conventions."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def run_recipe(engine, replacements=()):
    path = EXAMPLES / f"{engine}-sim" / "calc-norm.py"
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "calc-norm.py"
        source = path.read_text(encoding="utf-8")
        for old, new in replacements:
            source = source.replace(old, new)
        script.write_text(source, encoding="utf-8")
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True)


class NormalizationScriptsTest(unittest.TestCase):
    def test_cloudy_luminosity_and_node_scaling(self):
        result = run_recipe("cloudy", (("REDSHIFT = 0.184", "REDSHIFT = None"),
                                       ("DISTANCE_MPC = None", "DISTANCE_MPC = 1000")))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CLOUDY L_ion,sim = 1.000000e+48", result.stdout)
        self.assertIn("1.000000e-13", result.stdout)

        luminosity_case = run_recipe("cloudy", (("L_ION_SIM_INPUT = 1e48", "L_ION_SIM_INPUT = 2e45"),
                                               ("REDSHIFT = 0.184", "REDSHIFT = None"),
                                               ("DISTANCE_MPC = None", "DISTANCE_MPC = 1000")))
        self.assertEqual(luminosity_case.returncode, 0, luminosity_case.stderr)
        self.assertIn("CLOUDY L_ion,sim = 2.000000e+45", luminosity_case.stdout)
        self.assertIn("5.000000e-11", luminosity_case.stdout)

        xi_case = run_recipe("cloudy", (("CLOUDY_INPUT = \"luminosity\"", "CLOUDY_INPUT = \"xi\""),
                                       ("LOGXI = None", "LOGXI = 3.0"),
                                       ("LOG_DENSITY = None", "LOG_DENSITY = 10.0"),
                                       ("RADIUS_CM = None", "RADIUS_CM = 1e16"),
                                       ("REDSHIFT = 0.184", "REDSHIFT = None"),
                                       ("DISTANCE_MPC = None", "DISTANCE_MPC = 1000")))
        self.assertEqual(xi_case.returncode, 0, xi_case.stderr)
        self.assertIn("CLOUDY L_ion,sim = 1.000000e+45", xi_case.stdout)
        self.assertIn("1.000000e-10", xi_case.stdout)

        missing_xi_inputs = run_recipe("cloudy", (("CLOUDY_INPUT = \"luminosity\"", "CLOUDY_INPUT = \"xi\""),))
        self.assertNotEqual(missing_xi_inputs.returncode, 0)
        self.assertIn("Set LOGXI, LOG_DENSITY and RADIUS_CM", missing_xi_inputs.stderr)

    def test_xstar_input_luminosity_cancels(self):
        first = run_recipe("xstar")
        changed = run_recipe("xstar", (("L_ION_SIM = 1e38", "L_ION_SIM = 2e38"),))
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(changed.returncode, 0, changed.stderr)
        self.assertNotEqual(first.stdout.splitlines()[0], changed.stdout.splitlines()[0])
        self.assertEqual(first.stdout.splitlines()[1], changed.stdout.splitlines()[1])


if __name__ == "__main__":
    unittest.main()
