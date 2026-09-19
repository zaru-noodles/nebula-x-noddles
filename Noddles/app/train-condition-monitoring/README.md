# Noddles train condition monitor

Local web application for running the Door, ACV, Rail Corrugation, and SHM
models in `Noddles/Optional_Items`. Uploaded files are processed in isolated
temporary directories and removed after each request.

## Run locally

Install the Python requirements stored beside the web-app directory, then start the web app:

```powershell
python -m pip install -r ..\requirements.txt
```

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

The selected interpreter must provide NumPy, pandas, and openpyxl. The route
handler launches the frozen Python models as child processes. The repository
Docker image includes both the Node.js application and its Python runtime, so
the same behavior is available locally and on Cloud Run.

Each subsystem accepts its native input format and returns the exact required
submission schema. Use **Download CSV** on the result screen to save it.

## Deploy to Google Cloud Run

The deployment bundle is in `Noddles/app`: `Dockerfile`, `cloudbuild.yaml`,
`requirements.txt`, ignore rules, and the deployment script. It builds a
standalone Next.js server, installs the frozen Python dependencies, copies the
four model packages, listens on Cloud Run's `PORT`, and runs as a non-root user.

Prerequisites:

- A GCP project with billing enabled
- The [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- Permission to enable APIs, run Cloud Build, and deploy Cloud Run services

Authenticate without sharing a password or service-account key:

```powershell
gcloud auth login
gcloud auth application-default login
```

From `Noddles/app`, deploy to Singapore with:

```powershell
.\scripts\deploy-gcp.ps1 -ProjectId "your-gcp-project-id"
```

Override the region or service name when needed:

```powershell
.\scripts\deploy-gcp.ps1 `
  -ProjectId "your-gcp-project-id" `
  -Region "asia-southeast1" `
  -ServiceName "train-condition-monitoring"
```

The script enables Cloud Run, Cloud Build, and Artifact Registry, creates a
Docker image repository when needed, builds and pushes the image, then deploys
it as a public Cloud Run service with 2 vCPU, 2 GiB memory, a 10-minute request
timeout, and concurrency of one. The low concurrency prevents multiple CPU- and
memory-heavy model runs from competing inside one instance. Autoscaling is
capped at eight instances to fit the default 16-vCPU regional quota.

After deployment, verify `https://SERVICE_URL/api/health` returns
`{"status":"ok"}`.

### Upload-size limitation

Cloud Run limits HTTP/1 requests to 32 MiB. The current UI sends selected files
in one multipart request, so keep the combined upload below that limit. For
larger production datasets, upload the source files directly to Cloud Storage
and pass object references to a background inference job instead of proxying
the bytes through this service.
