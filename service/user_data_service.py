from utils.db_handler import fetch_one, fetch_all


def get_user_profile(user_code: str) -> dict | None:
    return fetch_one(
        """
        SELECT user_code, username, city, house_area, family_type
        FROM users
        WHERE user_code = :user_code
        """,
        {"user_code": user_code},
    )


def get_user_devices(user_code: str) -> list[dict]:
    return fetch_all(
        """
        SELECT id, device_name, model, serial_no, firmware_version,
               purchase_date, status
        FROM devices
        WHERE user_code = :user_code
        """,
        {"user_code": user_code},
    )


def get_cleaning_records(user_code: str, record_month: str) -> list[dict]:
    return fetch_all(
        """
        SELECT id, user_code, device_id, record_month, clean_area,
               clean_count, coverage_rate, obstacle_count, missed_area,
               brush_life_days, filter_life_percent, summary
        FROM cleaning_records
        WHERE user_code = :user_code
          AND record_month = :record_month
        """,
        {"user_code": user_code, "record_month": record_month},
    )


def get_monthly_report_data(user_code: str, record_month: str) -> dict:
    user = get_user_profile(user_code)
    devices = get_user_devices(user_code)
    records = get_cleaning_records(user_code, record_month)

    return {
        "user": user,
        "devices": devices,
        "records": records,
    }
