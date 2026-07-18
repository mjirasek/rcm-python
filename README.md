# RCM Python

Pure Python/NumPy implementation of the Ring Current Model, with a FastAPI
backend and a static Plotly frontend. It contains no C++, pybind11, OpenMP, or
compiled extension.

The implementation includes the MATLAB-compatible legacy and Chem. Sci. path
builders, RCGF calculation, raw and interior-positive sign conventions, NMR
fitting, field grids, and the `rung12` MATLAB regression case.

## Repository layout

```text
src/rcm_python/     calculation engine
rcm_api.py          FastAPI application
rcm_backend_core.py Plotly figures and example case
frontend/rcm/       static browser interface
examples/rung12/    regression data
tests/              calculation and API tests
deploy/oracle/      Oracle VM service configuration
```

The frontend source is kept here so API and interface changes remain together.
For production, `frontend/rcm/` is published at `michaeljirasek.com/rcm`, while
Oracle runs only the Python API.

## Local backend

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e ".[api,dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m uvicorn rcm_api:app --host 127.0.0.1 --port 8070
```

Open `frontend/rcm/index.html` through a local static server. Its local default
API address is `http://127.0.0.1:8070`.

## API

- `GET /health`
- `GET /api/config`
- `POST /api/model`

The backend identifier returned by the API is always `python`.

## Deployment

Oracle deployment files are in `deploy/oracle/`. GitHub Actions runs the test
suite on each push. The manual `Deploy Oracle` action becomes usable after
these repository secrets are configured:

- `ORACLE_HOST`: public IP or hostname.
- `ORACLE_USER`: normally `ubuntu`.
- `ORACLE_SSH_KEY`: private deployment key.
- `ORACLE_KNOWN_HOSTS`: verified SSH host-key entry for the VM.

The deployment action is manual, so pushing code cannot deploy accidentally.
