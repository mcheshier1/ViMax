import {execFile as execFileCallback} from 'node:child_process';
import {existsSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {promisify} from 'node:util';

const execFile = promisify(execFileCallback);
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const enginePath = path.join(repoRoot, 'utils', 'video_assembly.py');

/** Assemble clips through the shared geometry and conservative lighting correction engine. */
export async function assembleClips(clips, temporaryPath) {
  const localPython = path.join(repoRoot, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const python = process.env.VIMAX_ASSEMBLY_PYTHON || (existsSync(localPython) ? localPython : 'python3');
  const operation = execFile(python, [enginePath, '--output', temporaryPath], {maxBuffer: 8 * 1024 * 1024});
  let inputError;
  operation.child.stdin.on('error', (error) => { inputError = error; });
  operation.child.stdin.end(JSON.stringify({clips}));
  const {stdout} = await operation;
  if (inputError) throw inputError;
  const report = JSON.parse(stdout);
  if (report.version !== 4 || !Array.isArray(report.clips)) throw new Error('The assembly engine returned an incompatible correction report');
  return report;
}
 
/** Prepare the actual ending of a local clip through the shared video geometry engine. */
export async function prepareRenderedEnding(input) {
  const localPython = path.join(repoRoot, '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
  const python = process.env.VIMAX_ASSEMBLY_PYTHON || (existsSync(localPython) ? localPython : 'python3');
  const operation = execFile(python, [enginePath, '--rendered-ending', input.videoPath], {maxBuffer: 8 * 1024 * 1024});
  let inputError;
  operation.child.stdin.on('error', (error) => { inputError = error; });
  operation.child.stdin.end(JSON.stringify({
    first_frame: input.firstFrame ?? null,
    last_frame: input.lastFrame ?? null,
    width: input.width ?? 1280,
    height: input.height ?? 720,
  }));
  const {stdout} = await operation;
  if (inputError) throw inputError;
  const result = JSON.parse(stdout);
  if (result.normalization !== 'rendered-end-v1' || typeof result.path !== 'string'
    || typeof result.sha256 !== 'string' || typeof result.clip_sha256 !== 'string') {
    throw new Error('The assembly engine returned incompatible rendered-ending metadata');
  }
  return result;
}
