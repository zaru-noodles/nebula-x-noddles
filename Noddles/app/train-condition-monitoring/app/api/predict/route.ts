type SubsystemId = "door" | "acv" | "rail" | "shm";

const outputNames: Record<SubsystemId, string> = {
  door: "door_predictions.csv",
  acv: "acv_predictions.csv",
  rail: "rail_predictions.csv",
  shm: "shm_predictions.csv",
};

export async function POST(request: Request) {
  const formData = await request.formData();
  const subsystem = formData.get("subsystem") as SubsystemId | null;
  const files = formData
    .getAll("files")
    .filter((entry): entry is File => entry instanceof File && entry.size > 0);

  if (!subsystem || !Object.hasOwn(outputNames, subsystem)) {
    return Response.json({ error: "Choose a valid subsystem." }, { status: 400 });
  }
  if (!files.length) {
    return Response.json({ error: "Upload at least one data file." }, { status: 400 });
  }

  if (subsystem === "door") {
    return Response.json({
      subsystem,
      outputFilename: outputNames[subsystem],
      columns: ["start_time", "end_time", "prediction"],
      rows: [
        { start_time: "2023-7-5-0-0-0-0", end_time: "2023-7-5-0-0-4-500", prediction: "Normal" },
        { start_time: "2023-7-5-0-0-30-0", end_time: "2023-7-5-0-0-33-800", prediction: "Normal" },
        { start_time: "2023-7-5-0-1-10-0", end_time: "2023-7-5-0-1-13-200", prediction: "Abnormal resistance" },
        { start_time: "2023-7-5-0-1-45-0", end_time: "2023-7-5-0-1-48-600", prediction: "Normal" },
        { start_time: "2023-7-5-0-2-20-0", end_time: "2023-7-5-0-2-24-100", prediction: "Abnormal resistance" },
      ],
    });
  }

  if (subsystem === "acv") {
    return Response.json({
      subsystem,
      outputFilename: outputNames[subsystem],
      columns: ["file_id", "ranked_cars"],
      rows: files.map((file) => ({ file_id: file.name, ranked_cars: "03|01|05|02|04|06|07|08" })),
    });
  }

  if (subsystem === "rail") {
    const labels = ["Normal", "Side I", "Normal", "Side II", "Normal"];
    return Response.json({
      subsystem,
      outputFilename: outputNames[subsystem],
      columns: ["file_id", "prediction"],
      rows: files.map((file, index) => ({ file_id: file.name, prediction: labels[index % labels.length] })),
    });
  }

  const values = [0.6123, 0.0812, 0.3345, 0.9051, 0.0421];
  return Response.json({
    subsystem,
    outputFilename: outputNames[subsystem],
    columns: ["file_id", "prediction"],
    rows: files.map((file, index) => ({ file_id: file.name, prediction: values[index % values.length] })),
  });
}
