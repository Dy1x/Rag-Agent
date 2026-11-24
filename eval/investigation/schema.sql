PRAGMA foreign_keys = ON;

CREATE TABLE devices (
    device_id TEXT PRIMARY KEY NOT NULL,
    site_id TEXT NOT NULL,
    model TEXT NOT NULL
);

CREATE TABLE alarm_events (
    event_id TEXT PRIMARY KEY NOT NULL,
    device_id TEXT NOT NULL REFERENCES devices(device_id),
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);

-- Every row represents a successful rollout.
CREATE TABLE firmware_rollouts (
    rollout_id TEXT PRIMARY KEY NOT NULL,
    device_id TEXT NOT NULL REFERENCES devices(device_id),
    from_version TEXT NOT NULL,
    to_version TEXT NOT NULL,
    completed_at TEXT NOT NULL
);

CREATE TABLE site_events (
    event_id TEXT PRIMARY KEY NOT NULL,
    site_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    occurred_at TEXT NOT NULL
);
