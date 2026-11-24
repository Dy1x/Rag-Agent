CREATE TABLE IF NOT EXISTS cb_customers (
  customer_id   VARCHAR(32)  NOT NULL PRIMARY KEY,
  nickname      VARCHAR(64)  NOT NULL,
  created_at    DATETIME(3)  NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 钱包投影：以流水为准，本表在同一事务中随流水更新，用于快速展示与加锁
CREATE TABLE IF NOT EXISTS cb_wallets (
  customer_id   VARCHAR(32)  NOT NULL PRIMARY KEY,
  balance_fen   BIGINT       NOT NULL DEFAULT 0,   -- 账面余额（含冻结部分）
  frozen_fen    BIGINT       NOT NULL DEFAULT 0,   -- 冻结中金额；可用余额 = balance - frozen
  version       INT          NOT NULL DEFAULT 0,
  updated_at    DATETIME(3)  NOT NULL,
  CONSTRAINT fk_wallet_customer FOREIGN KEY (customer_id) REFERENCES cb_customers(customer_id),
  CONSTRAINT chk_wallet_nonneg CHECK (balance_fen >= 0 AND frozen_fen >= 0)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_tariff_policies (
  policy_id               VARCHAR(32)  NOT NULL PRIMARY KEY,
  name                    VARCHAR(64)  NOT NULL,
  effective_from          DATETIME(3)  NOT NULL,
  effective_to            DATETIME(3)  NULL,          -- NULL 表示长期有效
  idle_grace_minutes      INT          NOT NULL,
  idle_fee_fen_per_minute INT          NOT NULL,
  idle_fee_cap_fen        INT          NULL,
  freeze_amount_fen       INT          NOT NULL,
  policy_text             TEXT         NOT NULL       -- 给车主/Agent 阅读的计费规则原文
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_tariff_periods (
  policy_id               VARCHAR(32)  NOT NULL,
  start_local             TIME         NOT NULL,      -- 本地时间，含
  end_local               TIME         NOT NULL,      -- 本地时间，不含；end <= start 表示跨零点
  period_type             ENUM('peak','flat','valley') NOT NULL,
  electricity_fen_per_kwh INT          NOT NULL,
  service_fen_per_kwh     INT          NOT NULL,
  PRIMARY KEY (policy_id, start_local),
  CONSTRAINT fk_period_policy FOREIGN KEY (policy_id) REFERENCES cb_tariff_policies(policy_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_chargers (
  charger_id    VARCHAR(32)  NOT NULL,
  connector_id  VARCHAR(8)   NOT NULL,
  station_name  VARCHAR(64)  NOT NULL,
  policy_id     VARCHAR(32)  NOT NULL,
  PRIMARY KEY (charger_id, connector_id),
  CONSTRAINT fk_charger_policy FOREIGN KEY (policy_id) REFERENCES cb_tariff_policies(policy_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_charging_attempts (
  attempt_id     VARCHAR(32)  NOT NULL PRIMARY KEY,
  customer_id    VARCHAR(32)  NOT NULL,
  charger_id     VARCHAR(32)  NOT NULL,
  connector_id   VARCHAR(8)   NOT NULL,
  requested_at   DATETIME(3)  NOT NULL,
  result         ENUM('started','timeout','rejected') NOT NULL,
  session_id     VARCHAR(32)  NULL,
  freeze_txn_id  VARCHAR(32)  NULL,
  INDEX idx_attempt_customer (customer_id, requested_at),
  INDEX idx_attempt_charger (charger_id, connector_id, requested_at),
  CONSTRAINT fk_attempt_customer FOREIGN KEY (customer_id) REFERENCES cb_customers(customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_charging_sessions (
  session_id     VARCHAR(32)  NOT NULL PRIMARY KEY,
  attempt_id     VARCHAR(32)  NOT NULL UNIQUE,
  charger_id     VARCHAR(32)  NOT NULL,
  connector_id   VARCHAR(8)   NOT NULL,
  started_at     DATETIME(3)  NOT NULL,
  charge_end_at  DATETIME(3)  NULL,       -- 停止充电时间
  unplugged_at   DATETIME(3)  NULL,       -- 拔枪时间（占位费依据）
  status         ENUM('charging','finished') NOT NULL,
  CONSTRAINT fk_session_attempt FOREIGN KEY (attempt_id) REFERENCES cb_charging_attempts(attempt_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_meter_readings (
  reading_id     VARCHAR(32)  NOT NULL PRIMARY KEY,
  session_id     VARCHAR(32)  NOT NULL,
  reading_type   ENUM('start','end') NOT NULL,
  energy_wh      BIGINT       NOT NULL,
  occurred_at    DATETIME(3)  NOT NULL,
  received_at    DATETIME(3)  NOT NULL,   -- 补传时 received_at 晚于 occurred_at
  UNIQUE KEY uk_session_reading (session_id, reading_type),
  CONSTRAINT fk_reading_session FOREIGN KEY (session_id) REFERENCES cb_charging_sessions(session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_orders (
  order_id            VARCHAR(32)  NOT NULL PRIMARY KEY,
  customer_id         VARCHAR(32)  NOT NULL,
  attempt_id          VARCHAR(32)  NOT NULL,
  session_id          VARCHAR(32)  NOT NULL UNIQUE,
  policy_id           VARCHAR(32)  NOT NULL,     -- 下单时的政策快照
  energy_wh           BIGINT       NULL,         -- 缺结束读数时为 NULL
  electricity_fee_fen BIGINT       NULL,
  service_fee_fen     BIGINT       NULL,
  idle_fee_fen        BIGINT       NULL,
  total_fen           BIGINT       NULL,         -- 订单应付快照（可能被故障注入写错）
  status              ENUM('pending_settlement','settled') NOT NULL,
  created_at          DATETIME(3)  NOT NULL,
  settled_at          DATETIME(3)  NULL,
  INDEX idx_order_customer (customer_id, created_at),
  CONSTRAINT fk_order_customer FOREIGN KEY (customer_id) REFERENCES cb_customers(customer_id),
  CONSTRAINT fk_order_session FOREIGN KEY (session_id) REFERENCES cb_charging_sessions(session_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 钱包流水：资金事实的唯一来源
CREATE TABLE IF NOT EXISTS cb_wallet_transactions (
  txn_id           VARCHAR(32)  NOT NULL PRIMARY KEY,
  customer_id      VARCHAR(32)  NOT NULL,
  txn_type         ENUM('recharge','freeze','unfreeze','deduct','refund') NOT NULL,
  status           ENUM('pending','succeeded','failed') NOT NULL,
  amount_fen       BIGINT       NOT NULL,
  order_id         VARCHAR(32)  NULL,
  attempt_id       VARCHAR(32)  NULL,
  related_txn_id   VARCHAR(32)  NULL,     -- unfreeze→freeze，refund→deduct
  external_txn_no  VARCHAR(64)  NULL,     -- 充值渠道单号
  idempotency_key  VARCHAR(96)  NULL,
  source           ENUM('simulator','ops_manual','fund_executor') NOT NULL,
  note             VARCHAR(255) NULL,
  created_at       DATETIME(3)  NOT NULL,
  finished_at      DATETIME(3)  NULL,
  UNIQUE KEY uk_external_txn (external_txn_no),
  UNIQUE KEY uk_idempotency (idempotency_key),
  INDEX idx_txn_customer (customer_id, created_at),
  INDEX idx_txn_order (order_id),
  INDEX idx_txn_related (related_txn_id),
  CONSTRAINT chk_txn_amount CHECK (amount_fen > 0),
  CONSTRAINT fk_txn_customer FOREIGN KEY (customer_id) REFERENCES cb_customers(customer_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_dispute_cases (
  case_id          VARCHAR(32)  NOT NULL PRIMARY KEY,
  handler          VARCHAR(32)  NOT NULL,          -- 'billing_dispute'，预留其他处理器
  customer_id      VARCHAR(32)  NOT NULL,
  chat_session_id  VARCHAR(64)  NULL,
  question         TEXT         NOT NULL,
  dispute_type     VARCHAR(32)  NULL,
  status           VARCHAR(32)  NOT NULL,          -- 取值见 6.6 报告状态
  latest_run_id    VARCHAR(32)  NULL,
  created_at       DATETIME(3)  NOT NULL,
  updated_at       DATETIME(3)  NOT NULL,
  INDEX idx_case_customer (customer_id, created_at),
  INDEX idx_case_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_case_runs (
  run_id       VARCHAR(32)  NOT NULL PRIMARY KEY,  -- 同时作为 LangGraph thread_id
  case_id      VARCHAR(32)  NOT NULL,
  planner      ENUM('offline','deepseek') NOT NULL,
  status       VARCHAR(32)  NOT NULL,
  report       JSON         NULL,
  started_at   DATETIME(3)  NOT NULL,
  finished_at  DATETIME(3)  NULL,
  INDEX idx_run_case (case_id),
  CONSTRAINT fk_run_case FOREIGN KEY (case_id) REFERENCES cb_dispute_cases(case_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_proposals (
  proposal_id  VARCHAR(32)  NOT NULL PRIMARY KEY,
  run_id       VARCHAR(32)  NOT NULL,
  version      INT          NOT NULL,
  payload      JSON         NOT NULL,              -- Proposal 序列化
  created_at   DATETIME(3)  NOT NULL,
  UNIQUE KEY uk_run_version (run_id, version)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_reviews (
  review_id          VARCHAR(32)  NOT NULL PRIMARY KEY,
  proposal_id        VARCHAR(32)  NOT NULL,
  hard_check_passed  BOOLEAN      NOT NULL,
  approved           BOOLEAN      NOT NULL,
  issues             JSON         NOT NULL,
  created_at         DATETIME(3)  NOT NULL,
  INDEX idx_review_proposal (proposal_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_approvals (
  approval_id      VARCHAR(32)  NOT NULL PRIMARY KEY,
  proposal_id      VARCHAR(32)  NOT NULL UNIQUE,
  approver         VARCHAR(64)  NOT NULL,
  decision         ENUM('approve','reject','approve_modified') NOT NULL,
  modified_actions JSON         NULL,             -- approve_modified 时的最终动作
  comment          VARCHAR(255) NULL,
  decided_at       DATETIME(3)  NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_fund_actions (
  customer_id      VARCHAR(32)  NOT NULL,
  action_id        VARCHAR(32)  NOT NULL PRIMARY KEY,
  proposal_id      VARCHAR(32)  NOT NULL,
  action_index     INT          NOT NULL,
  action_type      VARCHAR(32)  NOT NULL,         -- 'refund' | 'unfreeze'，可扩展
  idempotency_key  VARCHAR(96)  NOT NULL UNIQUE,
  target_txn_id    VARCHAR(32)  NOT NULL,
  amount_fen       BIGINT       NOT NULL,
  result_txn_id    VARCHAR(32)  NULL,
  status           ENUM('succeeded','rejected') NOT NULL,
  reject_reason    VARCHAR(255) NULL,
  created_at       DATETIME(3)  NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

CREATE TABLE IF NOT EXISTS cb_trace_events (
  id          BIGINT       NOT NULL AUTO_INCREMENT PRIMARY KEY,
  run_id      VARCHAR(32)  NOT NULL,
  seq         INT          NOT NULL,
  node        VARCHAR(64)  NOT NULL,
  event_type  VARCHAR(32)  NOT NULL,    -- node_start / node_end / tool_call / llm_output / interrupt / resume / error
  payload     JSON         NOT NULL,
  created_at  DATETIME(3)  NOT NULL,
  UNIQUE KEY uk_run_seq (run_id, seq)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
