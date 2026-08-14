# Environments

The main training environment is never globally upgraded or downgraded.

`.venv-reference` is a local `--system-site-packages` environment used only for AgenticIR official reference parity. Create it with the pinned packages in `reference-requirements.txt`. Before importing BasicSR/pyiqa under torchvision 0.18, the parity entrypoint injects an in-memory compatibility module named `torchvision.transforms.functional_tensor` exposing `rgb_to_grayscale` from `torchvision.transforms.functional`.

Validated reference versions:

```text
Python 3.12
Torch 2.3.0+cu121 (inherited, training environment unchanged)
NumPy 1.26.4
SciPy 1.14.1
OpenCV 4.9.0
BasicSR 1.4.2
pyiqa 0.1.10
```
