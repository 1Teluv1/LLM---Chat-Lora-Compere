import { readdir } from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

type ArtifactOption = {
  value: string;
  label: string;
};

async function walkFiles(dir: string): Promise<string[]> {
  const entries = await readdir(dir, { withFileTypes: true });
  const files: string[] = [];
  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      files.push(...(await walkFiles(fullPath)));
      continue;
    }
    files.push(fullPath);
  }
  return files;
}

function toPosixRelative(root: string, fullPath: string): string {
  return path.relative(root, fullPath).split(path.sep).join("/");
}

export async function GET() {
  const repoRoot = path.resolve(process.cwd(), "..");
  const artifactsRoot = path.join(repoRoot, "artifacts");
  const baseRoot = path.join(artifactsRoot, "base");
  const loraRoot = path.join(artifactsRoot, "lora");

  try {
    const [baseFiles, loraFiles] = await Promise.all([
      walkFiles(baseRoot).catch(() => []),
      walkFiles(loraRoot).catch(() => [])
    ]);

    const base = baseFiles
      .filter((file) => file.endsWith(".gguf"))
      .map((file): ArtifactOption => {
        const relative = toPosixRelative(repoRoot, file);
        return { value: relative, label: path.basename(file) };
      });

    const loraSet = new Set<string>();
    for (const file of loraFiles) {
      if (!file.endsWith("adapter_model.gguf") && !file.endsWith("adapter_config.json")) {
        continue;
      }
      const dir = path.dirname(file);
      loraSet.add(toPosixRelative(repoRoot, dir));
    }
    const lora = Array.from(loraSet)
      .sort((a, b) => a.localeCompare(b))
      .map((dir): ArtifactOption => ({ value: dir, label: path.basename(dir) || dir }));

    return NextResponse.json({ base, lora });
  } catch (error) {
    return NextResponse.json(
      { message: `artifacts 스캔 실패: ${String(error)}` },
      { status: 500 }
    );
  }
}
