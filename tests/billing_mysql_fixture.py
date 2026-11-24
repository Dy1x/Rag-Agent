import os
import unittest
from uuid import uuid4

from service.billing.db import (
    apply_schema,
    create_database,
    drop_database,
    get_billing_engine,
)


@unittest.skipUnless(
    os.environ.get("BILLING_TEST_MYSQL") == "1", "需要 BILLING_TEST_MYSQL=1"
)
class MysqlCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.database = "cb_test_" + uuid4().hex[:12]
        create_database(cls.database)
        cls.addClassCleanup(drop_database, cls.database)
        cls.engine = get_billing_engine(cls.database)
        apply_schema(cls.engine)
