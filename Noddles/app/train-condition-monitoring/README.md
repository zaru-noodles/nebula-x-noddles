# Noddles train condition monitor

Local web application for running the Door, ACV, Rail Corrugation, and SHM
models in `Noddles/Optional_Items`. Uploaded files are processed in isolated
temporary directories and removed after each request.

## Run locally

Install the repository-root Python requirements, then start the web app:

```powershell
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). The inference API detects
the bundled Codex Python runtime when it is available. To select another
interpreter explicitly:

```powershell
$env:PYTHON_EXECUTABLE = "C:\path\to\python.exe"
npm run dev
```

The selected interpreter must provide NumPy, pandas, and openpyxl. The app is
intended to run locally because its route handler launches the frozen Python
models as child processes.

Each subsystem accepts its native input format and returns the exact required
submission schema. Use **Download CSV** on the result screen to save it.
