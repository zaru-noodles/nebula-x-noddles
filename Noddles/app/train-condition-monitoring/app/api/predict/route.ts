import { execFile } from "node:child_process";
import { access, mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { homedir, tmpdir } from "node:os";
import path from "node:path";
import { promisify } from "node:util";

type SubsystemId = "door" | "acv" | "rail" | "shm";
type SubsystemConfig = {
  directory: string;
  outputFilename: string;
  columns: string[];
  extensions: string[];
  maxFiles: number;
  modelArgument?: string;
};

const runFile = promisify(execFile);
const subsystemConfigs: Record<SubsystemId, SubsystemConfig> = {
  door: { directory: "Door", outputFilename: "door_predictions.csv", columns: ["start_time", "end_time", "prediction"], extensions: [".csv"], maxFiles: 1 },
  acv: { directory: "ACV", outputFilename: "acv_predictions.csv", columns: ["file_id", "ranked_cars"], extensions: [".xlsx"], maxFiles: 20, modelArgument: "--weights" },
  rail: { directory: "Rail Corrugation", outputFilename: "rail_predictions.csv", columns: ["file_id", "prediction"], extensions: [".csv"], maxFiles: 20 },
  shm: { directory: "SHM", outputFilename: "shm_predictions.csv", columns: ["file_id", "prediction"], extensions: [".csv"], maxFiles: 20 },
};

type PythonCommand = { executable: string; prefix: string[] };
let pythonCommandPromise: Promise<PythonCommand> | undefined;

function candidatePythonCommands(): PythonCommand[] {
  const configured = process.env.PYTHON_EXECUTABLE;
  const bundled = path.join(homedir(), ".cache", "codex-runtimes", "codex-primary-runtime", "dependencies", "python", process.platform === "win32" ? "python.exe" : "bin/python");
  return [
    ...(configured ? [{ executable: configured, prefix: [] }] : []),
    { executable: bundled, prefix: [] },
    { executable: process.platform === "win32" ? "python.exe" : "python3", prefix: [] },
    ...(process.platform === "win32" ? [{ executable: "py.exe", prefix: ["-3"] }] : []),
  ];
}

async function resolvePython(): Promise<PythonCommand> {
  if (!pythonCommandPromise) {
    pythonCommandPromise = (async () => {
      for (const candidate of candidatePythonCommands()) {
        try {
          if (path.isAbsolute(candidate.executable)) await access(candidate.executable);
          await runFile(candidate.executable, [...candidate.prefix, "-c", "import numpy, pandas, openpyxl"], { timeout: 15_000, windowsHide: true });
          return candidate;
        } catch {
          // Try the next interpreter; the app needs all three model dependencies.
        }
      }
      throw new Error("No compatible Python runtime was found. Install requirements.txt or set PYTHON_EXECUTABLE.");
    })();
  }
  return pythonCommandPromise;
}

function isSubsystem(value: FormDataEntryValue | null): value is SubsystemId {
  return typeof value === "string" && Object.hasOwn(subsystemConfigs, value);
}

function parseCsvLine(line: string): string[] {
  const values: string[] = [];
  let value = "";
  let quoted = false;
  for (let index = 0; index < line.length; index += 1) {
    const character = line[index];
    if (character === '"') {
      if (quoted && line[index + 1] === '"') { value += '"'; index += 1; }
      else quoted = !quoted;
    } else if (character === "," && !quoted) { values.push(value); value = ""; }
    else value += character;
  }
  values.push(value);
  return values;
}

function parsePredictionCsv(text: string, expectedColumns: string[]) {
  const lines = text.replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
  if (!lines.length) throw new Error("The model produced an empty prediction file.");
  const columns = parseCsvLine(lines[0]);
  if (columns.join("\u0000") !== expectedColumns.join("\u0000")) throw new Error(`The model returned unexpected columns: ${columns.join(", ")}`);
  return lines.slice(1).map((line) => {
    const values = parseCsvLine(line);
    if (values.length !== columns.length) throw new Error("The model produced a malformed CSV row.");
    return Object.fromEntries(columns.map((column, index) => {
      const raw = values[index];
      const numeric = column === "prediction" && raw !== "" ? Number(raw) : Number.NaN;
      return [column, Number.isFinite(numeric) ? numeric : raw];
    }));
  });
}

function safeFilename(filename: string) {
  return path.basename(filename).replace(/[^a-zA-Z0-9._ -]/g, "_");
}

export async function POST(request: Request) {
  let workingDirectory: string | undefined;
  try {
    const formData = await request.formData();
    const subsystemValue = formData.get("subsystem");
    if (!isSubsystem(subsystemValue)) return Response.json({ error: "Choose a valid subsystem." }, { status: 400 });

    const config = subsystemConfigs[subsystemValue];
    const files = formData.getAll("files").filter((entry): entry is File => entry instanceof File && entry.size > 0);
    if (!files.length) return Response.json({ error: "Upload at least one data file." }, { status: 400 });
    if (files.length > config.maxFiles) return Response.json({ error: `${config.directory} accepts at most ${config.maxFiles} file${config.maxFiles === 1 ? "" : "s"} per run.` }, { status: 400 });

    const names = files.map((file) => safeFilename(file.name));
    if (new Set(names.map((name) => name.toLowerCase())).size !== names.length) return Response.json({ error: "Every uploaded file must have a unique filename." }, { status: 400 });
    const unsupported = names.find((name) => !config.extensions.includes(path.extname(name).toLowerCase()));
    if (unsupported) return Response.json({ error: `${unsupported} is not supported. Use ${config.extensions.join(" or ")} files.` }, { status: 400 });

    workingDirectory = await mkdtemp(path.join(tmpdir(), `noddles-${subsystemValue}-`));
    const inputDirectory = path.join(workingDirectory, "input");
    await mkdir(inputDirectory, { recursive: true });
    await Promise.all(files.map(async (file, index) => {
      await writeFile(path.join(inputDirectory, names[index]), Buffer.from(await file.arrayBuffer()));
    }));

    const modelRoot = path.resolve(process.cwd(), "..", "..", "Optional_Items", config.directory);
    const script = path.join(modelRoot, "code", "predict.py");
    const output = path.join(/*turbopackIgnore: true*/ workingDirectory, config.outputFilename);
    const input = subsystemValue === "door" ? path.join(inputDirectory, names[0]) : inputDirectory;
    const args = [script, "--input", input, "--output", output];
    if (config.modelArgument) args.push(config.modelArgument, path.join(modelRoot, "model", "model.json"));

    const python = await resolvePython();
    await runFile(python.executable, [...python.prefix, ...args], { cwd: path.dirname(script), timeout: 5 * 60_000, maxBuffer: 4 * 1024 * 1024, windowsHide: true });

    const rows = parsePredictionCsv(await readFile(/*turbopackIgnore: true*/ output, "utf8"), config.columns);
    if (!rows.length) throw new Error("The model did not produce any predictions.");
    return Response.json({ subsystem: subsystemValue, outputFilename: config.outputFilename, columns: config.columns, rows });
  } catch (error) {
    const message = error instanceof Error ? error.message : "The analysis could not be completed.";
    console.error("Prediction failed", error);
    return Response.json({ error: message }, { status: 500 });
  } finally {
    if (workingDirectory) await rm(workingDirectory, { recursive: true, force: true });
  }
}
