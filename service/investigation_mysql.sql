CREATE TABLE IF NOT EXISTS ci_datasets (
    dataset VARCHAR(64) PRIMARY KEY,
    has_site_events BOOLEAN NOT NULL,
    content_sha256 CHAR(64)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS ci_devices (
    dataset VARCHAR(64) NOT NULL,
    device_id VARCHAR(191) NOT NULL,
    site_id VARCHAR(191) NOT NULL,
    model TEXT NOT NULL,
    PRIMARY KEY(dataset,device_id),
    INDEX(dataset,site_id,device_id),
    FOREIGN KEY(dataset) REFERENCES ci_datasets(dataset)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS ci_alarm_events (
    dataset VARCHAR(64) NOT NULL,
    event_id VARCHAR(191) NOT NULL,
    device_id VARCHAR(191) NOT NULL,
    event_type VARCHAR(128) NOT NULL,
    occurred_at VARCHAR(64) NOT NULL,
    instant_us BIGINT NOT NULL,
    PRIMARY KEY(dataset,event_id),
    INDEX(dataset,device_id,instant_us),
    FOREIGN KEY(dataset,device_id) REFERENCES ci_devices(dataset,device_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS ci_firmware_rollouts (
    dataset VARCHAR(64) NOT NULL,
    rollout_id VARCHAR(191) NOT NULL,
    device_id VARCHAR(191) NOT NULL,
    from_version TEXT NOT NULL,
    to_version TEXT NOT NULL,
    completed_at VARCHAR(64) NOT NULL,
    instant_us BIGINT NOT NULL,
    PRIMARY KEY(dataset,rollout_id),
    INDEX(dataset,device_id,instant_us),
    FOREIGN KEY(dataset,device_id) REFERENCES ci_devices(dataset,device_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS ci_site_events (
    dataset VARCHAR(64) NOT NULL,
    event_id VARCHAR(191) NOT NULL,
    site_id VARCHAR(191) NOT NULL,
    event_type VARCHAR(128) NOT NULL,
    occurred_at VARCHAR(64) NOT NULL,
    instant_us BIGINT NOT NULL,
    PRIMARY KEY(dataset,event_id),
    INDEX(dataset,site_id,instant_us),
    FOREIGN KEY(dataset) REFERENCES ci_datasets(dataset)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS ci_session_interruptions (
    dataset VARCHAR(64) NOT NULL,
    interruption_id VARCHAR(191) NOT NULL,
    device_id VARCHAR(191) NOT NULL,
    charger_id VARCHAR(32) NOT NULL,
    session_id VARCHAR(32) NOT NULL,
    site_event_id VARCHAR(191) NOT NULL,
    alarm_event_id VARCHAR(191) NOT NULL,
    cause VARCHAR(64) NOT NULL,
    occurred_at VARCHAR(64) NOT NULL,
    instant_us BIGINT NOT NULL,
    PRIMARY KEY(dataset,interruption_id),
    INDEX(dataset,site_event_id,cause,instant_us),
    FOREIGN KEY(dataset,site_event_id) REFERENCES ci_site_events(dataset,event_id),
    FOREIGN KEY(dataset,alarm_event_id) REFERENCES ci_alarm_events(dataset,event_id),
    FOREIGN KEY(dataset,device_id) REFERENCES ci_devices(dataset,device_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_bin;

CREATE TABLE IF NOT EXISTS ci_billing_children (
    session_id VARCHAR(32) NOT NULL PRIMARY KEY,
    child_case_id VARCHAR(32) NOT NULL UNIQUE,
    FOREIGN KEY(child_case_id) REFERENCES cb_dispute_cases(case_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS ci_case_links (
    parent_run_id VARCHAR(32) NOT NULL,
    child_case_id VARCHAR(32) NOT NULL,
    interruption_id VARCHAR(191) NOT NULL,
    PRIMARY KEY(parent_run_id,child_case_id),
    FOREIGN KEY(parent_run_id) REFERENCES cb_case_runs(run_id) ON DELETE CASCADE,
    FOREIGN KEY(child_case_id) REFERENCES cb_dispute_cases(case_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
