# Inspect a spectral table

This example makes a small synthetic optical-depth cube and its derived OGIP
transmission table. It does not run CLOUDY or XSTAR. Generate the tables with:

```bash
python make_demo.py
```

Open `inspect_tables.ipynb` in Jupyter and run the cells. The notebook uses
one slider per tabulated parameter; it shows stored nodes without spectral
interpolation. Select `%matplotlib widget` for live plot updates. The generated
`products/` directory is ignored by Git.

For a real table, replace the path passed to `TableInspector` with an iongrid
cube (`.fits`) or an OGIP table (`.amod`, `.mmod`, `.emod`). An OGIP additive
table stores flux per energy bin, while an iongrid cube retains its original
quantity and unit. See [the output schema](../../docs/schema.md).
