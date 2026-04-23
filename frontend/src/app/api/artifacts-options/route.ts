import { access, readdir } from "node:fs/promises";
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

async function existsDir(dir: string): Promise<boolean> {
  try {
    await access(dir);
    return true;
  } catch {
    return false;
  }
}

async function resolveRepoRoot(): Promise<string> {
  let current = process.cwd();
  for (let i = 0; i < 5; i += 1) {
    if (await existsDir(path.join(current, "artifacts"))) {
      return current;
    }
    const parent = path.dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return path.resolve(process.cwd(), "..");
}

function toPosixRelative(root: string, fullPath: string): string {
  return path.relative(root, fullPath).split(path.sep).join("/");
}

export async function GET() {
  const repoRoot = await resolveRepoRoot();
  const artifactsRoot = path.join(repoRoot, "artifacts");

  try {
    const artifactFiles = await walkFiles(artifactsRoot).catch(() => []);
    const artifactEntries = await readdir(artifactsRoot, { withFileTypes: true }).catch(() => []);

    const base = artifactFiles
      .filter((file) => {
        if (!file.endsWith(".gguf")) return false;
        const name = path.basename(file).toLowerCase();
        return name !== "adapter_model.gguf";
      })
      .map((file): ArtifactOption => {
        const relative = toPosixRelative(repoRoot, file);
        return { value: relative, label: relative };
      })
      .sort((a, b) => a.label.localeCompare(b.label));

    const artifactsRelative = toPosixRelative(repoRoot, artifactsRoot);
    const lora = artifactEntries
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name)
      .filter((name) => name !== "base")
      .map((name): ArtifactOption => ({
        value: `${artifactsRelative}/${name}`,
        label: `${artifactsRelative}/${name}`
      }))
      .sort((a, b) => a.label.localeCompare(b.label));

    return NextResponse.json({ base, lora });
  } catch (error) {
    return NextResponse.json(
      { message: `artifacts 스캔 실패: ${String(error)}` },
      { status: 500 }
    );
  }
}
