import unittest
from datetime import datetime, timedelta

from service.billing.tariff import (
    TariffPolicy,
    compute_order_amount,
    find_period,
    round_half_up_div,
)
from service.billing.timeutil import local_to_utc

TZ = "Asia/Shanghai"
START = local_to_utc(datetime(2026, 11, 20, 8, 30), TZ)
END = START + timedelta(hours=1)
POLICY = TariffPolicy(
    idle_grace_minutes=15,
    idle_fee_fen_per_minute=50,
    idle_fee_cap_fen=3000,
    periods=[
        dict(
            start_local=s,
            end_local=e,
            period_type=k,
            electricity_fen_per_kwh=a,
            service_fen_per_kwh=b,
        )
        for s, e, k, a, b in [
            ("23:00", "07:00", "valley", 40, 40),
            ("07:00", "10:00", "flat", 80, 60),
            ("10:00", "23:00", "peak", 120, 80),
        ]
    ],
)


class TariffTests(unittest.TestCase):
    def test_cross_midnight_and_end_exclusive(self):
        for hour, expected in [
            (23, "valley"),
            (0, "valley"),
            (6, "valley"),
            (7, "flat"),
            (10, "peak"),
        ]:
            at = local_to_utc(datetime(2026, 11, 20, hour), TZ)
            self.assertEqual(find_period(POLICY, at, TZ).period_type, expected)

    def test_rounding(self):
        for numerator, expected in [
            (499, 0),
            (500, 1),
            (1500, 2),
            (10**20 + 500, 10**17 + 1),
        ]:
            self.assertEqual(round_half_up_div(numerator, 1000), expected)

    def test_scenario_amounts(self):
        def bill(policy=POLICY, start=START, delay=10):
            return compute_order_amount(
                policy, start, END, END + timedelta(minutes=delay), 0, 4000, TZ
            )

        self.assertEqual(bill().total_fen, 560)
        self.assertEqual(
            bill(start=START + timedelta(hours=2)).total_fen - bill().total_fen, 240
        )
        self.assertEqual(bill(delay=35).idle_fee_fen, 1000)
        self.assertEqual(
            bill(policy=POLICY.model_copy(update={"idle_grace_minutes": 0})).total_fen
            - bill().total_fen,
            500,
        )
        self.assertEqual(bill(delay=150).idle_fee_fen, 3000)
        self.assertEqual(bill(delay=15).idle_fee_fen, 0)
        self.assertEqual(
            compute_order_amount(
                POLICY,
                START,
                END,
                END + timedelta(minutes=15, milliseconds=1),
                0,
                0,
                TZ,
            ).idle_fee_fen,
            50,
        )

    def test_missing_readings_and_unplug(self):
        result = compute_order_amount(POLICY, START, END, None, None, None, TZ)
        self.assertCountEqual(result.missing, ["start_wh", "end_wh", "unplugged_at"])
        self.assertIsNone(result.energy_wh)
        self.assertIsNone(result.total_fen)
        self.assertEqual(result.idle_fee_fen, 0)
