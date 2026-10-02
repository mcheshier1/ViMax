import {execFile as execFileCallback} from 'node:child_process';
import {promisify} from 'node:util';

const execFile = promisify(execFileCallback);

function executable(name) {
  return name === 'ffmpeg' ? (process.env.VIMAX_FFMPEG_CMD || 'ffmpeg') : (process.env.VIMAX_FFPROBE_CMD || 'ffprobe');
}

async function probe(file) {
  const {stdout} = await execFile(executable('ffprobe'), ['-v', 'error', '-protocol_whitelist', 'file,pipe', '-show_streams', '-show_format', '-of', 'json', file], {maxBuffer: 1024 * 1024});
  const info = JSON.parse(stdout);
  const video = info.streams?.find((stream) => stream.codec_type === 'video');
  const duration = Number(video?.duration || info.format?.duration);
  if (!video || !Number.isFinite(duration) || duration <= 0) throw new Error(`Cannot determine video duration: ${file}`);
  return {duration, audio: info.streams.some((stream) => stream.codec_type === 'audio')};
}

export async function assembleClips(clips, temporaryPath) {
  const metadata = await Promise.all(clips.map((file) => probe(file)));
  const args = ['-nostdin', '-hide_banner', '-loglevel', 'error', '-filter_complex_threads', '1', '-y'];
  for (const clip of clips) args.push('-threads', '1', '-protocol_whitelist', 'file,pipe', '-i', clip);
  const filters = [];
  const concat = [];
  metadata.forEach(({duration, audio}, index) => {
    filters.push(`[${index}:v:0]setpts=PTS-STARTPTS,scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps=30,format=yuv420p[v${index}]`);
    if (audio) {
      filters.push(`[${index}:a:0]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,apad,atrim=duration=${duration},asetpts=PTS-STARTPTS[a${index}]`);
    } else {
      filters.push(`anullsrc=channel_layout=stereo:sample_rate=48000,atrim=duration=${duration},asetpts=PTS-STARTPTS[a${index}]`);
    }
    concat.push(`[v${index}][a${index}]`);
  });
  filters.push(`${concat.join('')}concat=n=${clips.length}:v=1:a=1[outv][outa]`);
  args.push('-filter_complex', filters.join(';'), '-map', '[outv]', '-map', '[outa]', '-threads', '2', '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-c:a', 'aac', '-b:a', '192k', '-movflags', '+faststart', '-f', 'mp4', temporaryPath);
  await execFile(executable('ffmpeg'), args, {maxBuffer: 4 * 1024 * 1024});
  // Probe the completed temporary output before the caller atomically publishes it.
  await probe(temporaryPath);
  return metadata;
}
