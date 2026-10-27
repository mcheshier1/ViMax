def resolve_duration_seconds(override, default, *, allowed=None, fixed=None) -> int:
    """Resolve and validate a per-request duration before provider submission."""
    seconds = default if override is None else override
    if not isinstance(seconds, int) or isinstance(seconds, bool) or seconds < 1:
        raise ValueError("Video duration must be a positive whole number of seconds")
    if fixed is not None and seconds != fixed:
        raise ValueError(f"This video provider supports fixed {fixed}-second clips only")
    if allowed is not None and seconds not in allowed:
        raise ValueError(f"Unsupported video duration {seconds}s; supported durations: {list(allowed)}")
    return seconds
