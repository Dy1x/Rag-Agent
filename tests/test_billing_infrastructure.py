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
        self.assertEqual(len(inspect(self.engine).get_table_names()), 17)
        with self.engine.connect() as conn:
            for table, count in [
                ("customers", 2),
                ("wallets", 2),
                ("chargers", 3),
                ("tariff_periods", 6),
                ("tariff_policies", 1),
            ]:
                self.assertEqual(
                    conn.scalar(text(f"SELECT COUNT(*) FROM cb_{table}")), count
                )
        with self.assertRaises(BillingError):
            drop_database("charging_billing")
