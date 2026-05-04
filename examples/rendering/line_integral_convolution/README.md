# Line Integral Convolution Rendering

This example computes a line integral convolution field from a rotating dipole
vector field and writes a focused animation of the center slice. It starts from
fixed random noise, advects the texture along the dipole field, and uses the
LIC result to modulate a jet-colored field-magnitude image.

Run it with:

```bash
python render_lic.py
```

To render the LIC field as a 3D RGBA volume with a rotating cube overlay, run:

```bash
python render_lic_volume.py
```

The script writes PNG frames and an animated GIF to
`outputs_line_integral_convolution/`.
