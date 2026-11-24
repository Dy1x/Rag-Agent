from concurrent.futures import ThreadPoolExecutor

from billing_mysql_fixture import MysqlCase
from service.billing.errors import BillingError
from service.billing.executor import FundActionExecutor
from service.billing.ids import new_id
from service.billing.models import ApprovedAction
from service.billing.scenarios import build_scenario
from service.billing.simulator_service import SimulatorService


class ExecutorTests(MysqlCase):
    def test_concurrent_retry_limits_and_ownership(self):
        sim = SimulatorService(self.engine)
        h = build_scenario(sim, "S2")
        executor = FundActionExecutor(self.engine)
        action = ApprovedAction(
            proposal_id=new_id("PRP"),
            action_index=0,
            action_type="refund",
            target_txn_id=h.refs["second_deduct"],
            amount_fen=560,
            customer_id=h.customer_id,
        )
        with ThreadPoolExecutor(2) as pool:
            first, second = pool.map(executor.execute, [action, action])
        self.assertEqual(first, second)
        self.assertEqual(first.status, "succeeded")
        self.assertEqual(
            len(
                [
                    t
                    for t in sim.list_transactions(h.customer_id)
                    if t.txn_type == "refund"
                ]
            ),
            1,
        )
        with self.assertRaisesRegex(BillingError, "参数"):
            executor.execute(action.model_copy(update={"amount_fen": 1}))
        for changes in [
            {"action_index": 1},
            {"action_index": 2, "customer_id": "CUS-DEMO-BOB"},
        ]:
            self.assertEqual(
                executor.execute(action.model_copy(update=changes)).status, "rejected"
            )
        self.assertTrue(sim.get_wallet(h.customer_id).consistent)

    def test_two_approved_proposals_share_order_limit(self):
        sim = SimulatorService(self.engine)
        h = build_scenario(sim, "S4")
        executor = FundActionExecutor(self.engine)
        actions = [
            ApprovedAction(
                proposal_id=new_id("PRP"),
                action_index=0,
                action_type="refund",
                target_txn_id=h.refs["first_deduct"],
                amount_fen=240,
                customer_id=h.customer_id,
            )
            for _ in range(2)
        ]
        with ThreadPoolExecutor(2) as pool:
            results = list(pool.map(executor.execute, actions))
        self.assertCountEqual([r.status for r in results], ["succeeded", "rejected"])
        self.assertEqual(
            sum(
                t.amount_fen
                for t in sim.list_transactions(h.customer_id)
                if t.txn_type == "refund"
            ),
            240,
        )
