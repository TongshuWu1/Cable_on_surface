# Cable On Surface Project

This folder contains the current cable tracking code, tests, config, tools, and reference material.

Run the main pipeline from this folder:

```bash
../.venv/bin/python main.py
```

Run the hardware-free tests from this folder:

```bash
../.venv/bin/python -m unittest discover -s tests
```

If your virtual environment is already active in the terminal or PyCharm, `python main.py` works too.

The live pipeline depends on the Stereolabs ZED SDK Python module (`pyzed.sl`), which is not installed from PyPI.
