"""Check that the inspector reads both sides of the cube-to-OGIP workflow."""

from pathlib import Path
import runpy

import numpy as np
import pytest

from iongrid.inspect import TableInspector


def test_cube_and_ogip_nodes():
    pytest.importorskip("astropy")
    recipe = Path(__file__).resolve().parents[1] / "examples/inspect-table/make_demo.py"
    runpy.run_path(str(recipe))
    output = recipe.parent / "products"

    cube = TableInspector(output / "demo-tau.fits")
    table = TableInspector(output / "demo-transmission.mmod")
    assert cube.names == ("logxi", "vturb", "log_nh")
    assert table.names == ("logxi", "vturb", "lognH")
    assert cube.values.shape == table.values.shape == (3, 2, 2, 160)
    optical_depth = cube.node(logxi=4.0, vturb=2000, log_nh=23.0)
    transmission = table.node(logxi=4.0, vturb=2000, lognH=23.0)
    np.testing.assert_allclose(transmission, np.exp(-optical_depth), rtol=1e-6)
    with pytest.raises(ValueError, match="not a stored grid coordinate"):
        table.node(logxi=4.2, vturb=2000, lognH=23.0)


def test_slider_selects_another_stored_node(monkeypatch):
    pytest.importorskip("ipywidgets")
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from IPython import display
    import matplotlib.pyplot as plt

    recipe = Path(__file__).resolve().parents[1] / "examples/inspect-table/make_demo.py"
    runpy.run_path(str(recipe))
    inspector = TableInspector(recipe.parent / "products/demo-tau.fits")
    monkeypatch.setattr(display, "display", lambda value: None)
    fig, sliders = inspector.show(emin=4, emax=10)
    before = np.asarray(fig.axes[0].lines[0].get_ydata()).copy()
    sliders[0].value = 2
    after = np.asarray(fig.axes[0].lines[0].get_ydata())
    assert not np.array_equal(before, after)
    plt.close(fig)
