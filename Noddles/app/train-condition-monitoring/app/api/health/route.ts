import { access } from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

export async function GET() {
  const optionalItemsRoot = process.env.MODEL_ROOT
    ? path.resolve(process.env.MODEL_ROOT)
    : path.resolve(process.cwd(), "..", "..", "Optional_Items");

  try {
    await Promise.all([
      access(path.join(optionalItemsRoot, "Door", "code", "predict.py")),
      access(path.join(optionalItemsRoot, "ACV", "code", "predict.py")),
      access(path.join(optionalItemsRoot, "Rail Corrugation", "code", "predict.py")),
      access(path.join(optionalItemsRoot, "SHM", "code", "predict.py")),
    ]);
    return Response.json({ status: "ok" });
  } catch {
    return Response.json({ status: "unhealthy", error: "Model files are unavailable." }, { status: 503 });
  }
}
