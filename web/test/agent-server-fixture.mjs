import {spawn} from 'node:child_process';
import {createServer} from 'node:net';
import {mkdtemp, mkdir, readdir, copyFile, symlink, writeFile, rm} from 'node:fs/promises';
import {once} from 'node:events';
import os from 'node:os';
import path from 'node:path';
import {fileURLToPath} from 'node:url';

export async function startAgentServer() {
  const webRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  const root = await mkdtemp(path.join(os.tmpdir(), 'vimax-agent-lifecycle-'));
  let child;
  try {
    const web = path.join(root, 'web');
    await mkdir(web);
    for (const file of await readdir(webRoot)) {
      if (file.endsWith('.mjs')) await copyFile(path.join(webRoot, file), path.join(web, file));
    }
    await symlink(path.join(webRoot, 'node_modules'), path.join(web, 'node_modules'), 'dir');
    await symlink(path.join(webRoot, 'dist'), path.join(web, 'dist'), 'dir');
    await mkdir(path.join(root, '.vimax'));
    const sessions = {};
    for (const id of ['original', 'other']) {
      sessions[id] = {session_id: id, project_name: id, working_dir: `.working_dir/${id}`, stage: 'initialized'};
      await mkdir(path.join(root, '.working_dir', id), {recursive: true});
    }
    await writeFile(path.join(root, '.vimax', 'sessions.json'), JSON.stringify({active_session_id: 'original', sessions}));
    const agent = path.join(root, 'controlled-agent.mjs');
    await writeFile(agent, `
      import readline from 'node:readline';
      import {existsSync, writeFileSync} from 'node:fs';
      import {setTimeout} from 'node:timers/promises';
      const emit = (event) => console.log(JSON.stringify(event));
      process.on('SIGTERM', () => {writeFileSync('terminated-' + process.pid, 'yes'); process.exit(0);});
      readline.createInterface({input: process.stdin}).on('line', async (text) => {
        const turn_id = 'turn-' + text;
        emit({type: 'turn', turn_id});
        writeFileSync('started-' + text, String(process.pid));
        while (!existsSync('release-' + text)) await setTimeout(10);
        writeFileSync('completed-' + text, 'yes');
        emit({type: 'done', turn_id, assistant: 'Completed', tool_results: []});
      });
    `);
    const reservation = createServer();
    reservation.listen(0, '127.0.0.1');
    await once(reservation, 'listening');
    const port = reservation.address().port;
    await new Promise((resolve, reject) => reservation.close((error) => error ? reject(error) : resolve()));
    child = spawn(process.execPath, [path.join(web, 'server.mjs')], {
      cwd: root,
      env: {...process.env, VIMAX_WEB_HOST: '127.0.0.1', VIMAX_WEB_PORT: String(port), VIMAX_AGENT_COMMAND: process.execPath, VIMAX_AGENT_ARGS: agent},
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    let logs = '';
    child.stderr.on('data', (chunk) => {logs += chunk;});
    await new Promise((resolve, reject) => {
      const deadline = setTimeout(() => reject(new Error(`Server startup timed out: ${logs}`)), 10000);
      child.on('error', (error) => {clearTimeout(deadline); reject(error);});
      child.on('exit', (code) => {clearTimeout(deadline); reject(new Error(`Server exited ${code}: ${logs}`));});
      child.stdout.on('data', (chunk) => {
        logs += chunk;
        if (logs.includes('ViMax Web:')) {clearTimeout(deadline); resolve();}
      });
    });
    const url = `http://127.0.0.1:${port}`;
    return {
      root, url,
      request: (endpoint, body) => fetch(url + endpoint, body === undefined ? {} : {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}),
      release: (text) => writeFile(path.join(root, 'release-' + text), 'yes'),
      async close() {
        const exited = once(child, 'exit');
        child.kill('SIGTERM');
        await exited;
        await rm(root, {recursive: true, force: true});
      },
    };
  } catch (error) {
    if (child && child.exitCode === null) {
      const exited = once(child, 'exit');
      child.kill('SIGTERM');
      await exited;
    }
    await rm(root, {recursive: true, force: true});
    throw error;
  }
}
