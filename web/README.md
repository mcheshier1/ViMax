# ViMax Web

The Film workbench uses the existing ViMax agent loop and JSONL event stream. The browser makes review and generation scope explicit; the agent still executes planning and rendering.

From the `ViMax` repository root, create the private local agent configuration once:

```bash
cp configs/agent.example.yaml configs/agent.local.yaml
```

```bash
cd web
npm install
npm run dev
```

Or from the repository root:

```bash
./vimax web
```

The default address is `http://127.0.0.1:4173`. Override it with `VIMAX_WEB_HOST` and `VIMAX_WEB_PORT`.

Production mode:

```bash
cd web
npm run build
cd ..
./vimax web start
```

Agent credentials continue to come from ViMax environment variables or `configs/agent.local.yaml`.

## Film workbench

- **Film** is the default destination. One reel row represents one shot in playback order; select its first frame, last frame, or clip in the persistent viewer.
- **Assemble** stitches the existing accepted clips in camera playback order into `final_video.mp4` using local FFmpeg only, then opens **Final film** for review. Every active clip must still match its approval fingerprint; removed shots are excluded. Save or discard local film drafts first. Assembly does not call the assistant or a media provider, change render settings, overwrite clips, or alter clip/keyframe approvals. Inputs are normalized to 1280×720, 30fps H.264/AAC, with silence for clips without audio; unchanged ordered inputs reuse the existing assembly. Failed encoding or publication preserves the previous final and its approval. A new final film awaits its own approval. `ffmpeg` and `ffprobe` must be available on the server (`VIMAX_FFMPEG_CMD` and `VIMAX_FFPROBE_CMD` may point to alternate executables).
- The video player's **Download** saves the final film as `<project title>.mp4` (or `<session ID>.mp4` for an unnamed project). Unicode titles are supported; characters unsafe in filenames are replaced. Other artifacts retain their original filenames. Download naming does not rename stored files or change approvals.
- **Needs review**, **Needs changes**, and **Missing** narrow the reel. **Accept & next** advances only after approval is saved. Review requires the first frame for HeyGen Video 1 and both frames for the other presets; accepted output must be unlocked before regeneration.
- **Edit plan & feedback** keeps shot drafts per project, render root, and slot in browser storage. Navigation does not discard them. Saving a plan explicitly invalidates dependent approvals and outputs; existing feedback is retained as a protected draft. **Save & regenerate** saves guidance, then opens a generation decision rather than immediately starting a render.
- **Generate…** shows the phase, exact shot slots, selected model/provider, and estimated or unknown cost. Selected, missing, and needs-changes scopes do not include accepted outputs. Clip generation requires accepted frames. Global model changes require a new decision; video settings saved here affect future renders in every project. The workbench does not automatically retry failures.
- Submitting a new generation replaces historical failure/stall banners with **Generation request sent — waiting for the agent**. Once fresh render progress arrives, the indicator changes to **Rendering**. Errors from the new attempt remain visible; old records are not deleted from Activity. Waiting, failure, and stall notices include **Open Assistant** for direct access to the agent response.
- **Script coverage** can remain collapsed even when it needs attention; its warning summary stays visible. Reviewing coverage is an explicit language-model request, not media generation.
- **Assistant** and **Activity** open as drawers. Assistant requests can target the whole film or selected shot. Chat drafts and attachments stay with their project while switching projects. Opening or switching projects does not start the agent; sending a request does. Creating a project explicitly starts the agent to create its session.
- While the assistant works, the composer remains editable and **Queue message** submits follow-ups in order. Messages show **Queued**, **In progress**, then **Answered**; each captures its shot/film scope and attachments at submission. Failed sends preserve the draft and attachments without stopping the current request. Switching the viewed project does not interrupt work; sends to a different project are blocked until the current request finishes.
- **Script** displays story documents. **Assets** displays thumbnails and video posters; raw JSON is behind the Advanced disclosure. Project details, settings, uploads, and project deletion remain available.

The selected project, destination, shot, preview, filter, and reel position are remembered locally. Browser storage restrictions can prevent persistence; clearing browser storage removes unsaved shot drafts.

## Media and refresh behavior

`ffmpeg` must be available on the server for image thumbnails and video posters. Set `VIMAX_FFMPEG_CMD` to an alternate executable path if needed. Thumbnails are generated on demand with bounded worker concurrency and a disposable temporary cache; a thumbnail failure does not silently download the full original.

The selected preview loads the original media. Original delivery supports HTTP byte ranges and conditional requests, so clip playback can seek without downloading every clip in the reel.

The workspace shares one film snapshot across its views. Lightweight progress reads trigger fresh snapshots when the revision changes; external file changes are also reconciled periodically. Reads pause while the browser tab is hidden, pending reads are cancelled on project changes, and mutation refreshes wait for a post-save snapshot. Acceptance fingerprints remain request-scoped rather than relying on a persistent size/mtime cache.

After updating server code, restart the web server **when its agent is idle**. A frontend hot reload alone does not install the new film, progress, and thumbnail endpoints.

Agnes status-query HTTP 429 responses retry the **same video ID**, without submitting another video creation request. Consecutive rate limits use exponential backoff with jitter (a 3-second initial ceiling, doubling to 60 seconds); valid `Retry-After` seconds or HTTP dates set a minimum wait and may exceed that ceiling. A successful status query resets the backoff. Activity reports the rate limit and next check delay. The existing `VIMAX_VIDEO_QUERY_TIMEOUT_SECONDS` deadline (default 600 seconds; nonpositive disables it) includes retry waits, and no further status query is sent after it expires. Restart an idle agent to load Python adapter changes; do not resubmit a paid render merely to recover its status.

## OpenRouter video models

`heygen/heygen-video-1` is available under **OpenRouter** in Settings and the Film generation decision. It uses the existing OpenRouter API key and video adapter; adding the preset does not change the currently configured model or submit a render.

- Clips support integer durations from **5 to 15 seconds**, at **480p** or **768p**. Selecting HeyGen keeps compatible values and replaces incompatible ones with 480p and an 8-second duration.
- Switching to another OpenRouter preset replaces HeyGen's 480p/768p settings and incompatible durations with compatible values. Changing providers resets the clip duration to 8 seconds alongside the provider's default model and resolution. Custom model values remain editable.
- Only the **first frame** conditions generation. The pipeline does not generate an unused end keyframe, and the workbench permits review and approval of the first frame alone. Existing end frames are retained, but cannot constrain the clip's ending. Choose a model with last-frame support when exact final poses and cut continuity matter.
- OpenRouter describes synthesized dialogue, ambience, and effects, but exposes no `generate_audio` switch for this model. The audio control is hidden; the provider's native audio behavior applies.

The [OpenRouter video catalog](https://openrouter.ai/api/v1/videos/models) currently lists these USD-per-second SKUs:

| Resolution | Output-duration rate | Reference-duration rate used for conservative budgeting |
| --- | ---: | ---: |
| 480p | $0.02 | $0.04 |
| 768p | $0.03 | $0.06 |

For an 8-second clip, the advertised output rates are $0.16/$0.24; the workbench budgets $0.32/$0.48 respectively. The applicability of the higher reference-duration SKUs to this first-frame image workflow has not been confirmed with a paid generation. Missing or unknown resolutions budget at $0.06/second. Estimates are not guaranteed charges; prices and additional provider charges can change.

Capabilities are validated against the catalog before submission. Unsupported last-frame and audio-toggle parameters are omitted with progress notes; invalid duration, resolution, or aspect ratio fails before a job is created. OpenRouter origin-relative polling URLs retain their leading slash so the API prefix is not duplicated. See the [model page](https://openrouter.ai/heygen/heygen-video-1) for current availability and capabilities.

## Chained keyframe continuity

- A project opts in with `frame_continuity.json` beside `render_acceptance.json`, containing `{"mode":"chained_keyframes"}`. Playback follows numeric camera order and each camera's stored active-shot order, not numeric shot IDs.
- Within the same camera, the preceding last frame becomes the next first frame byte-for-byte. A camera change generates a new view of that ending at the same instant, using it as the mandatory first reference. Every new last frame uses its own first frame as the mandatory scene reference. Remaining references are approved portraits, bounded by the image provider's limit; newly entering characters take priority.
- Editing or redrawing a shot invalidates following frame/clip approvals and final-film approval without deleting their media. Timeline changes invalidate the affected playback suffix. Preserved files are evidence, not permission to reuse an invalidated predecessor.
- Chained reference selection follows explicit frame visibility, including entrances and exits; mentioning an offscreen person does not add their portrait. If the provider limit is unknown, selection conservatively uses the scene plus at most two portraits.
- Scoped rendering never silently expands into additional paid work. Missing, stale, or invalidated predecessors must be explicitly included; a multi-shot request cannot skip dependencies between its first and last target. Shots after the last requested target are not generated.
- Chained mode requires a video provider with last-frame bracketing. First-frame-only models such as HeyGen are rejected with an actionable error rather than pretending an unused last keyframe enforces clip continuity. Keyframe boundaries are exact within a camera; actual video endpoints can still drift from their conditioning images.

## Assistant responsiveness

- Agent startup registers planning and rendering tools without importing their heavy pipelines, provider SDKs, or embedding/reranking dependencies. Those dependencies load when the corresponding operation is first used; generation still incurs its provider and pipeline costs.
- Assistant history is streamed from the global log and retains the latest 120 displayed messages, preserving their ordering and IDs. The backend caches up to 16 session-history projections and shares concurrent reads. Append, truncation, replacement, or rewrite invalidates the cache; changed logs are fully rescanned, so a cold read of a large log can still take seconds. Approval fingerprints are not cached by this mechanism.
- Tool results are delivered as soon as execution completes. The agent waits for either progress or completion instead of polling every 100 ms, and drains final progress before publishing the result.
- The backend keeps one active request and a FIFO of pending messages, dispatching the next only when the active turn finishes. Page refresh restores active/pending rows without restarting the agent. **Stop**, agent replacement, or process failure marks affected messages **Cancelled**; pending requests are not persisted across a server restart. A planning operation that changes the active session cancels old-project follow-ups instead of sending them into the new project. Stopping locally does not guarantee cancellation of an already-submitted paid provider job.
- **Waiting for the queue-enabled bridge** prevents untracked submissions when the browser has reconnected to an older backend or has lost its event connection. Keep editing the draft; reconnect, or restart the web server after the current request finishes to load the updated bridge.
- Backend changes require a web-server restart after the current assistant request finishes. Python-only changes require a fresh agent process. Refreshing the browser does neither; do not stop an active planning or generation request to install an update.

## Verification

```bash
cd web
npx tsc --noEmit
npm test
npm run build
```
