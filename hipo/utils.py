from datetime import datetime

import pytz
from dateutil.parser import parse


def format_datetime(dt: datetime):
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def datetime_now(format=False):
    dt = datetime.now(tz=pytz.timezone("Asia/Shanghai"))
    dt = dt.replace(tzinfo=None, microsecond=0)
    if format:
        dt = format_datetime(dt)
    return dt


def parse_datetime(dt):
    dt = parse(dt)
    dt = dt.replace(tzinfo=None, microsecond=0)
    return dt
