import subprocess
import sys

from sqlalchemy import inspect, text
from billing_mysql_fixture import MysqlCase
from service.billing.db import drop_database
from service.billing.errors import BillingError


class InfrastructureTests(MysqlCase):
    def test_initialize_repeat_and_reset(self):
        command = [
            sys.executable,
            "-m",
            "scripts.init_billing_db",
            "--database",
            self.database,
        ]
        subprocess.run(command, check=True)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "INSERT INTO cb_customers VALUES ('CUS-TEST','test',UTC_TIMESTAMP())"
                )
            )
        subprocess.run(command, check=True)
        with self.engine.connect() as conn:
            self.assertEqual(conn.scalar(text("SELECT COUNT(*) FROM cb_customers")), 3)
        subprocess.run(command + ["--reset"], check=True)
        inspector = inspect(self.engine)
        self.assertEqual(len(inspector.get_table_names()), 19)
        self.assertEqual(
            {index["name"] for index in inspector.get_indexes("cb_customer_memories")},
            {"uk_customer_key", "idx_mem_customer_active"},
        )
        self.assertEqual(
            {index["name"] for index in inspector.get_indexes("cb_memory_audit_logs")},
            {"idx_audit_customer", "idx_audit_memory"},
        )
        with self.engine.connect() as conn:
            for table, count in [
                ("customers", 2),
                ("wallets", 2),
                ("chargers", 3),
                ("tariff_periods", 6),
                ("tariff_policies", 1),
                ("customer_memories", 0),
                ("memory_audit_logs", 0),
            ]:
                self.assertEqual(
                    conn.scalar(text(f"SELECT COUNT(*) FROM cb_{table}")), count
                )
        with self.assertRaises(BillingError):
            drop_database("charging_billing")
