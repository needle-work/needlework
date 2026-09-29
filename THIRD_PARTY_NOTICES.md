# Third-party notices

needlework is released under the MIT License (`LICENSE`). It includes code adapted from
the projects below, whose licenses require their notices to be kept.

## Included code

### iPhUMI (Patel et al., 2026)

UMI-related code and data collection workflows draw on
[iPhUMI](https://github.com/real-stanford/iPhUMI) (MIT; copyright 2026 Austin Patel).
See the [upstream license](https://github.com/real-stanford/iPhUMI/blob/main/LICENSE).

### Diffusion Policy (Chi et al., RSS 2023)

`src/needlework/models/unet.py` (the conditional 1-D U-Net, its convolution blocks and
the sinusoidal step embedding) is adapted from Diffusion Policy. The EMA warmup schedule
in `src/needlework/models/ema.py` follows its `EMAModel`.

```text
MIT License

Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Diffuser (Janner et al., ICML 2022)

Diffusion Policy's convolution blocks and sinusoidal embedding, adapted in
`src/needlework/models/unet.py`, come from Diffuser.

```text
MIT License

Copyright (c) 2020 Phil Wang
Copyright (c) 2022 Michael Janner and Yilun Du

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Not included

These are fetched by `install_deps.sh` or by you, under their own licenses; nothing
from them is redistributed here.

- **DINOv3** (Meta). `install_deps.sh` checks out its source at a pinned commit; you
  download the ViT-B/16 weights from Meta after accepting the DINOv3 License. Both are
  governed by the DINOv3 License, not by this repository's license.
- **robomimic** and **robosuite** (MIT), **MuJoCo** (Apache-2.0), and the other Python
  packages in `pyproject.toml` / `uv.lock` are installed from their own distributions
  under their own licenses.
- **Robomimic v1.5 datasets.** `python -m needlework.data.download` fetches the
  low-dimensional HDF5 files from the robomimic dataset repository under its terms.
