"""
Mathematical Engineering - Financial Engineering, FY 2025-2026
Risk Management - Date Utilities
"""

import pandas as pd
import datetime as dt
import calendar


from typing import Union, List


def year_frac_act_x(t1: dt.datetime, t2: dt.datetime, x: int) -> float:
    """
    Compute the year fraction between two dates using the ACT/x convention.

    Parameters:
        t1 (dt.datetime): First date.
        t2 (dt.datetime): Second date.
        x (int): Number of days in a year.

    Returns:
        float: Year fraction between the two dates.
    """

    return (t2 - t1).days / x


def year_frac_30e_360(t1: dt.datetime, t2: dt.datetime):
    """
    Compute the year fraction between two dates using the 30E/360 convention.

    Parameters:
        t1 (dt.datetime): First date.
        t2 (dt.datetime): Second date.

    Returns:
        float: Year fraction between the two dates.
    """

    Y1, M1, D1 = t1.year, t1.month, min(30, t1.day)
    Y2, M2, D2 = t2.year, t2.month, min(30, t2.day)

    return ((360 * (Y2 - Y1)) + (30 * (M2 - M1)) + (D2 - D1)) / 360


def business_date_offset(
    base_date: Union[dt.date, pd.Timestamp],
    year_offset: int = 0,
    month_offset: int = 0,
    day_offset: int = 0,
) -> Union[dt.date, pd.Timestamp]:
    """
    Return the closest following business date to a reference date after applying the specified offset.

    Parameters:
        base_date (Union[dt.date, pd.Timestamp]): Reference date.
        year_offset (int): Number of years to add.
        month_offset (int): Number of months to add.
        day_offset (int): Number of days to add.

    Returns:
        Union[dt.date, pd.Timestamp]: Closest following business date to ref_date once the specified
            offset is applied.
    """

    # Adjust the year and month
    total_months = base_date.month + month_offset - 1
    year, month = divmod(total_months, 12)
    year += base_date.year + year_offset
    month += 1

    # Adjust the day and handle invalid days
    day = base_date.day
    try:
        adjusted_date = base_date.replace(
            year=year, month=month, day=day
        ) + dt.timedelta(days=day_offset)
    except ValueError:
        # Set to the last valid day of the adjusted month
        last_day_of_month = calendar.monthrange(year, month)[1]
        adjusted_date = base_date.replace(
            year=year, month=month, day=last_day_of_month
        ) + dt.timedelta(days=day_offset)

    # Adjust to the closest business day
    if adjusted_date.weekday() == 5:  # Saturday
        adjusted_date += dt.timedelta(days=2)
    elif adjusted_date.weekday() == 6:  # Sunday
        adjusted_date += dt.timedelta(days=1)

    return adjusted_date


def schedule_year_fraction(
    schedule: List[dt.datetime],
) -> List[float]:
    """
    Given a list of dates return the year fractions between each pair of consecutive dates.

    Parameters:
        schedule (List[dt.datetime]): List of dates.

    Returns:
        List[float]: List of year fractions.
    """

    sorted_schedule = sorted(schedule)

    return [
        year_frac_30e_360(sorted_schedule[i - 1], sorted_schedule[i])
        for i in range(1, len(sorted_schedule))
    ]


def date_series(
    t0: Union[dt.date, pd.Timestamp], t1: Union[dt.date, pd.Timestamp], freq: int
) -> Union[List[dt.date], List[pd.Timestamp]]:
    """
    Return a list of dates from t0 to t1 inclusive with frequency freq, where freq is specified as
    the number of dates per year.
    """

    dates = [t0]
    while dates[-1] < t1:
        dates.append(business_date_offset(t0, month_offset=len(dates) * 12 // freq))
    if dates[-1] > t1:
        dates.pop()
    if dates[-1] != t1:
        dates.append(t1)

    return dates
