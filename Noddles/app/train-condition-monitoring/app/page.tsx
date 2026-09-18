"use client";

import {
  ChangeEvent,
  DragEvent,
  ReactNode,
  useCallback,
  useRef,
  useState,
} from "react";

type SubsystemId = "door" | "acv" | "rail" | "shm";

type Subsystem = {
  id: SubsystemId;
  name: string;
  shortName: string;
  task: string;
  description: string;
  accepts: string[];
  signal: string;
  color: string;
};

type PredictionResponse = {
  subsystem: SubsystemId;
  outputFilename: string;
  columns: string[];
  rows: Record<string, string | number>[];
};

const subsystems: Subsystem[] = [
  {
    id: "door",
    name: "Passenger door",
    shortName: "Door",
    task: "Detect resistance",
    description:
      "Find each opening and closing cycle, then identify abnormal mechanical resistance.",
    accepts: [".csv"],
    signal: "Motor current, voltage, back-EMF and position",
    color: "#1368e8",
  },
  {
    id: "acv",
    name: "Air-conditioning",
    shortName: "ACV",
    task: "Locate refrigerant leak",
    description:
      "Rank every car from most to least likely to have a refrigerant leak.",
    accepts: [".xlsx"],
    signal: "Temperature and control-mode telemetry",
    color: "#08a6a6",
  },
  {
    id: "rail",
    name: "Rail corrugation",
    shortName: "Rail",
    task: "Classify rail wear",
    description:
      "Classify vibration records as Normal, Side I or Side II corrugation.",
    accepts: [".csv"],
    signal: "Multi-channel axle-box vibration and shock",
    color: "#7857d6",
  },
  {
    id: "shm",
    name: "Structural health",
    shortName: "SHM",
    task: "Estimate fatigue damage",
    description:
      "Estimate cumulative fatigue damage from dynamic stress time-series data.",
    accepts: [".csv"],
    signal: "Dynamic stress time series",
    color: "#df6b32",
  },
];

function Icon({ name, size = 20 }: { name: string; size?: number }) {
  const paths: Record<string, ReactNode> = {
    train: (
      <>
        <rect x="5" y="3" width="14" height="15" rx="3" />
        <path d="M8 21l2-3m6 3-2-3M8 7h8M8 11h8" />
        <circle cx="9" cy="15" r="1" fill="currentColor" stroke="none" />
        <circle cx="15" cy="15" r="1" fill="currentColor" stroke="none" />
      </>
    ),
    door: <><rect x="5" y="3" width="14" height="18" rx="2" /><path d="M12 3v18m-3-9h.01m6 0h.01" /></>,
    acv: <><path d="M12 3v18M4.2 7.5l15.6 9M4.2 16.5l15.6-9" /><circle cx="12" cy="12" r="2" /></>,
    rail: <><path d="M7 3l2 18m8-18-2 18M8 7h8M8.5 12h7M9 17h6" /></>,
    shm: <><path d="M3 13h4l2-7 4 12 2-6h6" /></>,
    upload: <><path d="M12 16V4m0 0L7 9m5-5 5 5" /><path d="M5 15v4h14v-4" /></>,
    file: <><path d="M6 2h8l4 4v16H6z" /><path d="M14 2v5h5M9 13h6m-6 4h6" /></>,
    close: <><path d="M6 6l12 12M18 6L6 18" /></>,
    check: <path d="M5 12.5l4.2 4.2L19 7" />,
    download: <><path d="M12 3v12m0 0 4-4m-4 4-4-4" /><path d="M5 19h14" /></>,
    arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
    shield: <><path d="M12 3l7 3v5c0 4.7-2.8 8.2-7 10-4.2-1.8-7-5.3-7-10V6z" /><path d="M9 12l2 2 4-5" /></>,
    reset: <><path d="M4 11a8 8 0 1 0 2-5.3L4 8" /><path d="M4 3v5h5" /></>,
  };
  return (
    <svg
      aria-hidden="true"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      {paths[name]}
    </svg>
  );
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function toCsv(result: PredictionResponse) {
  const escape = (value: string | number) => {
    const text = String(value);
    return /[",\n]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
  };
  return [
    result.columns.join(","),
    ...result.rows.map((row) =>
      result.columns.map((column) => escape(row[column] ?? "")).join(","),
    ),
  ].join("\n");
}

export default function Home() {
  const [selectedId, setSelectedId] = useState<SubsystemId>("door");
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [status, setStatus] = useState<"idle" | "processing" | "complete">("idle");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const selected = subsystems.find((item) => item.id === selectedId)!;

  const resetRun = useCallback(() => {
    setFiles([]);
    setResult(null);
    setStatus("idle");
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  }, []);

  const validateAndSetFiles = (incoming: File[]) => {
    setError("");
    const valid = incoming.filter((file) =>
      selected.accepts.some((extension) => file.name.toLowerCase().endsWith(extension)),
    );
    if (!valid.length) {
      setError(`Choose ${selected.accepts.join(" or ")} files for ${selected.shortName}.`);
      return;
    }
    const maxFiles = selectedId === "door" ? 1 : 20;
    const nextFiles = valid.slice(0, maxFiles);
    setFiles(nextFiles);
    setResult(null);
    setStatus("idle");
    if (valid.length > maxFiles) {
      setError(
        selectedId === "door"
          ? "Door analysis accepts one continuous-stream file. The first file was selected."
          : "A maximum of 20 files can be analysed at once.",
      );
    } else if (valid.length !== incoming.length) {
      setError("Some unsupported files were left out.");
    }
  };

  const onInputChange = (event: ChangeEvent<HTMLInputElement>) => {
    validateAndSetFiles(Array.from(event.target.files ?? []));
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    validateAndSetFiles(Array.from(event.dataTransfer.files));
  };

  const removeFile = (index: number) => {
    setFiles((current) => current.filter((_, fileIndex) => fileIndex !== index));
    setResult(null);
    setStatus("idle");
  };

  const runAnalysis = async () => {
    if (!files.length) return;
    setStatus("processing");
    setError("");
    const formData = new FormData();
    formData.set("subsystem", selectedId);
    files.forEach((file) => formData.append("files", file));

    try {
      const [response] = await Promise.all([
        fetch("/api/predict", { method: "POST", body: formData }),
        new Promise((resolve) => setTimeout(resolve, 1050)),
      ]);
      const responseBody = await response.json() as PredictionResponse & { error?: string };
      if (!response.ok) {
        throw new Error(responseBody.error || "The analysis service could not process this file.");
      }
      setResult(responseBody);
      setStatus("complete");
    } catch (requestError) {
      setStatus("idle");
      setError(
        requestError instanceof Error
          ? requestError.message
          : "The analysis could not be completed. Try again.",
      );
    }
  };

  const downloadResult = () => {
    if (!result) return;
    const blob = new Blob([toCsv(result)], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = result.outputFilename;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark"><Icon name="train" size={22} /></span>
          <span>Noddles</span>
          <span className="brand-divider" />
          <span className="product-name">Condition monitor</span>
        </div>
        <div className="system-state"><span /> Online</div>
      </header>

      <div className="workspace">
        <aside className="subsystem-panel" aria-label="Subsystem selection">
          <div className="panel-heading">
            <p>Choose subsystem</p>
          </div>
          <div className="route-list">
            <span className="route-line" aria-hidden="true" />
            {subsystems.map((subsystem) => (
              <button
                key={subsystem.id}
                className={`subsystem-button ${selectedId === subsystem.id ? "active" : ""}`}
                onClick={() => {
                  setSelectedId(subsystem.id);
                  resetRun();
                }}
                aria-pressed={selectedId === subsystem.id}
                style={{ "--subsystem-color": subsystem.color } as React.CSSProperties}
              >
                <span className="route-stop"><span /></span>
                <span className="subsystem-icon"><Icon name={subsystem.id} /></span>
                <span className="subsystem-label">
                  <strong>{subsystem.name}</strong>
                  <small>{subsystem.task}</small>
                </span>
                {selectedId === subsystem.id && <Icon name="arrow" size={17} />}
              </button>
            ))}
          </div>
        </aside>

        <section className="work-panel">
          <div className="work-heading">
            <div>
              <h1><span className="context-icon" style={{ color: selected.color }}><Icon name={selected.id} /></span>{selected.task}</h1>
              <p>{selected.description}</p>
            </div>
          </div>

          <div className="progress-track" aria-label={result ? "Result ready" : "Ready for upload"}>
            {["Subsystem", "Upload", "Result"].map((label, index) => {
              const currentStep = result ? 2 : 1;
              const state = index < currentStep ? "complete" : index === currentStep ? "current" : "";
              return (
                <div className={`progress-step ${state}`} key={label}>
                  <span className="progress-line" />
                  <span className="progress-label">
                    {state === "complete" ? <Icon name="check" size={13} /> : <i>{index + 1}</i>}
                    {label}
                  </span>
                </div>
              );
            })}
          </div>

          {!result ? (
            <div className="upload-stage">
              <input
                ref={inputRef}
                className="upload-input"
                type="file"
                multiple={selectedId !== "door"}
                accept={selected.accepts.join(",")}
                onChange={onInputChange}
              />

              {selectedId === "door" && files.length ? (
                <div
                  className={`selected-file-panel ${isDragging ? "dragging" : ""}`}
                  onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={onDrop}
                >
                  <span className="file-icon"><Icon name="file" size={21} /></span>
                  <div className="selected-file-details">
                    <strong>{files[0].name}</strong>
                    <span>{formatBytes(files[0].size)}</span>
                  </div>
                  <button className="replace-file" onClick={() => inputRef.current?.click()}>Replace</button>
                  <button className="remove-file" aria-label={`Remove ${files[0].name}`} onClick={() => removeFile(0)}><Icon name="close" size={18} /></button>
                </div>
              ) : (
                <div
                  className={`dropzone ${isDragging ? "dragging" : ""} ${files.length ? "has-files" : ""}`}
                  onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
                  onDragOver={(event) => event.preventDefault()}
                  onDragLeave={() => setIsDragging(false)}
                  onDrop={onDrop}
                  onClick={() => inputRef.current?.click()}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" || event.key === " ") inputRef.current?.click();
                  }}
                  role="button"
                  tabIndex={0}
                >
                  <span className="upload-orbit"><Icon name="upload" size={25} /></span>
                  <div>
                    <h2>{isDragging ? "Drop to upload" : "Drop sensor data here"}</h2>
                    {!isDragging && <p>or <span>browse files</span></p>}
                  </div>
                  <small>{selected.accepts.join(", ").toUpperCase()} · {selectedId === "door" ? "1 file" : "20 files max"}</small>
                </div>
              )}

              {error && <div className="error-message" role="alert">{error}</div>}

              {files.length > 0 && selectedId !== "door" && (
                <div className="file-queue">
                  <div className="queue-heading">
                    <div><strong>{files.length} {files.length === 1 ? "file" : "files"}</strong></div>
                    <button onClick={() => inputRef.current?.click()}>Choose files</button>
                  </div>
                  <div className="file-list">
                    {files.map((file, index) => (
                      <div className="file-row" key={`${file.name}-${file.lastModified}`}>
                        <span className="file-icon"><Icon name="file" size={19} /></span>
                        <div><strong>{file.name}</strong><span>{formatBytes(file.size)}</span></div>
                        <button aria-label={`Remove ${file.name}`} onClick={() => removeFile(index)}><Icon name="close" size={17} /></button>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="action-row">
                <button className="primary-button" disabled={!files.length || status === "processing"} onClick={runAnalysis}>
                  {status === "processing" ? <><span className="spinner" />Analysing sensor data</> : <>Run analysis <Icon name="arrow" size={18} /></>}
                </button>
              </div>
            </div>
          ) : (
            <ResultPanel
              result={result}
              onDownload={downloadResult}
              onReset={resetRun}
            />
          )}
        </section>
      </div>
    </main>
  );
}

function ResultPanel({
  result,
  onDownload,
  onReset,
}: {
  result: PredictionResponse;
  onDownload: () => void;
  onReset: () => void;
}) {
  return (
    <div className="result-stage">
      <div className="success-banner">
        <span className="success-icon"><Icon name="check" size={23} /></span>
        <div><h2>Analysis complete</h2><p>{result.rows.length} {result.rows.length === 1 ? "prediction" : "predictions"} ready to download.</p></div>
      </div>

      <ResultSummary result={result} />

      <div className="results-table-wrap">
        <div className="table-heading"><div><h2>Prediction result</h2><p>{result.outputFilename}</p></div><span>{result.rows.length} rows</span></div>
        <div className="table-scroll">
          <table>
            <thead><tr>{result.columns.map((column) => <th key={column}>{column.replaceAll("_", " ")}</th>)}</tr></thead>
            <tbody>
              {result.rows.map((row, index) => (
                <tr key={index}>
                  {result.columns.map((column) => {
                    const value = row[column];
                    const isPrediction = column === "prediction";
                    const className = isPrediction
                      ? value === "Normal" ? "status-normal" : typeof value === "string" ? "status-alert" : ""
                      : "";
                    return <td key={column}><span className={className}>{value}</span></td>;
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="result-actions">
        <button className="secondary-button" onClick={onReset}><Icon name="reset" size={18} />New analysis</button>
        <button className="primary-button" onClick={onDownload}><Icon name="download" size={18} />Download CSV</button>
      </div>
    </div>
  );
}

function ResultSummary({ result }: { result: PredictionResponse }) {
  if (result.subsystem === "door") {
    const abnormal = result.rows.filter((row) => row.prediction === "Abnormal resistance").length;
    return (
      <div className="result-summary">
        <div className="summary-primary"><span>Detected cycles</span><strong>{result.rows.length}</strong><small>Timestamp-gap segmentation</small></div>
        <div className="summary-secondary"><span>Abnormal resistance</span><strong>{abnormal}</strong><small>{result.rows.length - abnormal} cycles classified normal</small></div>
      </div>
    );
  }
  if (result.subsystem === "acv") {
    const topCars = result.rows.map((row) => String(row.ranked_cars).split("|")[0]);
    return (
      <div className="result-summary">
        <div className="summary-primary"><span>Cases ranked</span><strong>{result.rows.length}</strong><small>Peer-relative anomaly analysis</small></div>
        <div className="summary-secondary"><span>Highest-risk car</span><strong>{topCars[0] ?? "—"}</strong><small>{result.rows.length === 1 ? "First in the complete car ranking" : "Shown for the first uploaded case"}</small></div>
      </div>
    );
  }
  if (result.subsystem === "rail") {
    const faults = result.rows.filter((row) => row.prediction !== "Normal").length;
    return (
      <div className="result-summary">
        <div className="summary-primary"><span>Recordings analysed</span><strong>{result.rows.length}</strong><small>Speed-normalised side comparison</small></div>
        <div className="summary-secondary"><span>Corrugation flags</span><strong>{faults}</strong><small>Side I or Side II predictions</small></div>
      </div>
    );
  }
  const values = result.rows.map((row) => Number(row.prediction)).filter(Number.isFinite);
  const maximum = values.length ? Math.max(...values) : 0;
  return (
    <div className="result-summary">
      <div className="summary-primary"><span>Stress records analysed</span><strong>{result.rows.length}</strong><small>Rainflow-informed fatigue model</small></div>
      <div className="summary-secondary"><span>Highest predicted damage</span><strong>{maximum.toPrecision(4)}</strong><small>Cumulative fatigue damage</small></div>
    </div>
  );
}
