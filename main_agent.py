from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any, Iterable
from uuid import uuid4

ORIGINAL_STDOUT = sys.stdout


def event_stdout():
    if sys.stdout.__class__.__name__ == "_DiscardStream":
        return ORIGINAL_STDOUT
    return sys.stdout


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ViMax agent loop.")
    parser.add_argument("--session", default="", help="Existing session id to activate before the run starts.")
    parser.add_argument("--new-session", action="store_true", help="Create and activate a new empty session before the run starts.")
    parser.add_argument("--new-session-name", default="", help="Display name for a newly created session.")
    parser.add_argument("--new-session-style", default="", help="Visual style for a newly created session, applied to character portraits and inherited by later frames.")
    parser.add_argument("--jsonl", action="store_true", help="Print one JSON event per line.")
    parser.add_argument("--once", default="", help="Run a single prompt and exit. If omitted and stdin is not a TTY, stdin is consumed as one prompt.")
    parser.add_argument("--stdin-repl", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def load_runtime():
    from agent_runtime import build_runtime

    return build_runtime(".")


def load_session_index():
    from agent_runtime.session_index import SessionIndex

    return SessionIndex(".")


def print_event(event: dict[str, Any], *, jsonl: bool) -> None:
    out = event_stdout()
    if jsonl:
        print(json.dumps(event, ensure_ascii=False, default=str), file=out, flush=True)
        return
    event_type = event.get("type")
    if event_type == "turn":
        print(f"· turn: {event.get('turn_id', '')}", file=out, flush=True)
    elif event_type == "assistant_start":
        if event.get("round", 0):
            print("", file=out, flush=True)
    elif event_type == "assistant_end":
        outcome = event.get("outcome")
        if outcome in {"interrupted", "truncated"}:
            print(f"\n· assistant {outcome}: {event.get('finish_reason', 'unknown')}", file=out, flush=True)
        elif outcome == "tool_preamble":
            print("", file=out, flush=True)
    elif event_type == "token":
        print(event.get("delta", ""), end="", file=out, flush=True)
    elif event_type == "tool_start":
        tool = event.get("tool", {})
        print(f"\n· tool: {tool.get('name')} started", file=out, flush=True)
    elif event_type == "tool_progress":
        progress = event.get("progress", {})
        tool = event.get("tool", {})
        print(f"· tool: {tool.get('name')} {progress.get('stage', 'running')}: {progress.get('message', '')}", file=out, flush=True)
    elif event_type == "tool_result":
        result = event["tool_result"]
        status = "done" if result.get("ok") else "error"
        print(f"· tool: {result.get('name')} {status}", file=out, flush=True)
    elif event_type == "terminal":
        stream = event.get("stream", "stdout")
        print(f"· terminal[{stream}]: {event.get('line', '')}", file=out, flush=True)
    elif event_type == "status":
        print(f"· status: {event.get('phase')}: {event.get('message', '')}", file=out, flush=True)
    elif event_type == "session":
        session = (event.get("session") or {}).get("session") or {}
        if session:
            print(f"· session: {session.get('session_id')} {session.get('stage', '')}", file=out, flush=True)
    elif event_type == "done":
        print("", file=out, flush=True)
    elif event_type == "error":
        print(f"\nerror: {event.get('message', '')}", file=out, flush=True)


def prompt_inputs(args: argparse.Namespace) -> Iterable[str]:
    if args.once:
        yield args.once
        return
    if args.stdin_repl:
        for line in sys.stdin:
            user_input = line.strip()
            if user_input:
                yield user_input
        return
    if not sys.stdin.isatty():
        payload = sys.stdin.read().strip()
        if payload:
            yield payload
        return
    while True:
        try:
            user_input = input("› " if not args.jsonl else "")
        except EOFError:
            break
        if user_input.strip():
            yield user_input.strip()


async def amain(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.session and args.new_session:
        print("error: --session and --new-session cannot be used together", file=sys.stderr)
        return 2
    if args.new_session_name and not args.new_session:
        print("error: --new-session-name requires --new-session", file=sys.stderr)
        return 2
    if args.new_session_style and not args.new_session:
        print("error: --new-session-style requires --new-session", file=sys.stderr)
        return 2
    if args.session or args.new_session:
        try:
            session_index = load_session_index()
            if args.new_session:
                # A style chosen up front is inherited by every portrait and frame, so
                # the agent must neither invent one nor be unable to see it.
                session_index.create(project_name=args.new_session_name, style=args.new_session_style)
            else:
                session_index.set_active(args.session)
        except KeyError:
            print(f"error: unknown session id: {args.session}", file=sys.stderr)
            return 2
        except ValueError as exc:
            print(f"error: invalid session id: {exc}", file=sys.stderr)
            return 2
    runtime = load_runtime()
    interactive = sys.stdin.isatty() and not args.once
    if interactive and not args.jsonl:
        print("ViMax agent ready. Ctrl+C to exit.")
    for user_input in prompt_inputs(args):
        turn_id = ""
        terminal_seen = False
        try:
            async for event in runtime.stream_events(user_input):
                if event.get("type") == "turn":
                    turn_id = str(event.get("turn_id") or event.get("turn", {}).get("id") or "")
                if event.get("type") == "done":
                    terminal_seen = True
                print_event(event, jsonl=args.jsonl)
        except Exception as exc:
            if not turn_id:
                turn_id = f"turn-{uuid4().hex[:12]}"
                print_event({"type": "turn", "turn_id": turn_id, "turn": {"id": turn_id}}, jsonl=args.jsonl)
            if not terminal_seen:
                print_event({"type": "error", "turn_id": turn_id, "message": f"turn failed ({type(exc).__name__})"}, jsonl=args.jsonl)
                print_event({"type": "done", "turn_id": turn_id, "status": "failed", "assistant": "", "tool_results": []}, jsonl=args.jsonl)
    return 0


def main() -> None:
    try:
        raise SystemExit(asyncio.run(amain()))
    except KeyboardInterrupt:
        print("", file=sys.stderr)
        raise SystemExit(130)


if __name__ == "__main__":
    main()
