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
- **Needs review**, **Needs changes**, and **Missing** narrow the reel. **Accept & next** advances only after approval is saved. Incomplete frame pairs cannot be accepted; accepted output must be unlocked before regeneration.
- **Edit plan & feedback** keeps shot drafts per project, render root, and slot in browser storage. Navigation does not discard them. Saving a plan explicitly invalidates dependent approvals and outputs; existing feedback is retained as a protected draft. **Save & regenerate** saves guidance, then opens a generation decision rather than immediately starting a render.
- **Generate…** shows the phase, exact shot slots, selected model/provider, and estimated or unknown cost. Selected, missing, and needs-changes scopes do not include accepted outputs. Clip generation requires accepted frames. Global model changes require a new decision; video settings saved here affect future renders in every project. The workbench does not automatically retry failures.
- Submitting a new generation replaces historical failure/stall banners with **Generation request sent — waiting for the agent**. Once fresh render progress arrives, the indicator changes to **Rendering**. Errors from the new attempt remain visible; old records are not deleted from Activity. Waiting, failure, and stall notices include **Open Assistant** for direct access to the agent response.
- **Script coverage** can remain collapsed even when it needs attention; its warning summary stays visible. Reviewing coverage is an explicit language-model request, not media generation.
- **Assistant** and **Activity** open as drawers. Assistant requests can target the whole film or selected shot. Chat drafts and attachments stay with their project while switching projects. Opening or switching projects does not start the agent; sending a request does. Creating a project explicitly starts the agent to create its session.
- **Script** displays story documents. **Assets** displays thumbnails and video posters; raw JSON is behind the Advanced disclosure. Project details, settings, uploads, and project deletion remain available.

The selected project, destination, shot, preview, filter, and reel position are remembered locally. Browser storage restrictions can prevent persistence; clearing browser storage removes unsaved shot drafts.

## Media and refresh behavior

`ffmpeg` must be available on the server for image thumbnails and video posters. Set `VIMAX_FFMPEG_CMD` to an alternate executable path if needed. Thumbnails are generated on demand with bounded worker concurrency and a disposable temporary cache; a thumbnail failure does not silently download the full original.

The selected preview loads the original media. Original delivery supports HTTP byte ranges and conditional requests, so clip playback can seek without downloading every clip in the reel.

The workspace shares one film snapshot across its views. Lightweight progress reads trigger fresh snapshots when the revision changes; external file changes are also reconciled periodically. Reads pause while the browser tab is hidden, pending reads are cancelled on project changes, and mutation refreshes wait for a post-save snapshot. Acceptance fingerprints remain request-scoped rather than relying on a persistent size/mtime cache.

After updating server code, restart the web server **when its agent is idle**. A frontend hot reload alone does not install the new film, progress, and thumbnail endpoints.

Agnes status-query HTTP 429 responses retry the **same video ID**, without submitting another video creation request. Consecutive rate limits use exponential backoff with jitter (a 3-second initial ceiling, doubling to 60 seconds); valid `Retry-After` seconds or HTTP dates set a minimum wait and may exceed that ceiling. A successful status query resets the backoff. Activity reports the rate limit and next check delay. The existing `VIMAX_VIDEO_QUERY_TIMEOUT_SECONDS` deadline (default 600 seconds; nonpositive disables it) includes retry waits, and no further status query is sent after it expires. Restart an idle agent to load Python adapter changes; do not resubmit a paid render merely to recover its status.

## Verification

```bash
cd web
npx tsc --noEmit
npm test
npm run build
```
