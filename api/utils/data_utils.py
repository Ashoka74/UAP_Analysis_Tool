import pandas as pd
import numpy as np
import datetime

def sanitize_dataframe_for_json(df: pd.DataFrame) -> pd.DataFrame:
    """Convert non-JSON-serializable values (datetime, date, time, period, timedelta) to strings."""
    df_safe = df.copy()

    # Ensure simple integer index and string column names
    df_safe = df_safe.reset_index(drop=True)
    df_safe.columns = df_safe.columns.map(str)

    # Datetime columns: export as ISO 8601 strings with millisecond precision, null-safe
    for col in df_safe.select_dtypes(include=["datetime64[ns]"]).columns:
        try:
            dt = pd.to_datetime(df_safe[col], errors='coerce')
            # format to ISO with milliseconds (slice microseconds to 3 digits)
            iso = dt.dt.strftime('%Y-%m-%dT%H:%M:%S.%f').str.slice(0, 23)
            # set None where NaT
            iso = iso.where(dt.notna(), None)
            df_safe[col] = iso
        except Exception:
            # Fallback to plain string
            df_safe[col] = df_safe[col].astype(object).where(~df_safe[col].isna(), None)

    # Object columns: convert date/time-like objects to strings
    def _to_serializable(x):
        try:
            # Safe null check using a list contact for unhashable types
            if x is None or (not isinstance(x, (dict, list)) and pd.isna(x)):
                return None
        except Exception:
            # Fallback if the check itself fails
            if x is None:
                return None
            
        if isinstance(x, pd.Timestamp):
            try:
                return x.strftime('%Y-%m-%d %H:%M:%S')
            except Exception:
                return None
        if isinstance(x, (datetime.datetime, datetime.date)):
            return x.isoformat()
        if isinstance(x, datetime.time):
            return x.strftime('%H:%M:%S')
        # If it's a dict or list, it's already serializable by json.dumps if types match
        # but we might want to ensure it's clean. For now we leave it for the JSON serializer.
        return x

    for col in df_safe.select_dtypes(include=['object']).columns:
        try:
            # Try efficient mapping
            df_safe[col] = df_safe[col].map(_to_serializable)
        except (TypeError, Exception):
            # Fallback to slower apply or manual loop
            try:
                df_safe[col] = df_safe[col].apply(_to_serializable)
            except Exception:
                # Last resort: convert everything to string if mapping fails entirely
                df_safe[col] = df_safe[col].astype(str).where(df_safe[col].notna(), None)

    return df_safe

def detect_and_combine_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect separate year, month, day, hour columns and combine them into datetime columns.
    Removed streamlit logging for API safety.
    """
    df_combined = df.copy()
    
    year_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['year', 'yr', 'yyyy'])]
    month_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['month', 'mon', 'mm'])]
    day_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['day', 'dd', 'date'])]
    hour_cols = [col for col in df.columns if any(pattern in col.lower() for pattern in ['hour', 'hr', 'hh', 'time'])]
    
    date_combinations = []
    
    for year_col in year_cols:
        for month_col in month_cols:
            for day_col in day_cols:
                try:
                    sample_size = min(10, len(df))
                    test_sample = df.iloc[:sample_size]
                    
                    years = pd.to_numeric(test_sample[year_col], errors='coerce')
                    months = pd.to_numeric(test_sample[month_col], errors='coerce')
                    days = pd.to_numeric(test_sample[day_col], errors='coerce')
                    
                    valid_years = years.between(1900, 2100).all()
                    valid_months = months.between(1, 12).all()
                    valid_days = days.between(1, 31).all()
                    
                    if valid_years and valid_months and valid_days:
                        combination = {'year': year_col, 'month': month_col, 'day': day_col, 'hour': None}
                        for hour_col in hour_cols:
                            try:
                                hours = pd.to_numeric(test_sample[hour_col], errors='coerce')
                                if hours.between(0, 23).all():
                                    combination['hour'] = hour_col
                                    break
                            except:
                                continue
                        date_combinations.append(combination)
                except Exception:
                    continue
    
    for i, combo in enumerate(date_combinations):
        try:
            datetime_col_name = f"datetime_combined_{i+1}"
            year = pd.to_numeric(df_combined[combo['year']], errors='coerce')
            month = pd.to_numeric(df_combined[combo['month']], errors='coerce')
            day = pd.to_numeric(df_combined[combo['day']], errors='coerce')
            
            if combo['hour']:
                hour = pd.to_numeric(df_combined[combo['hour']], errors='coerce')
                df_combined[datetime_col_name] = pd.to_datetime({'year': year, 'month': month, 'day': day, 'hour': hour}, errors='coerce')
            else:
                df_combined[datetime_col_name] = pd.to_datetime({'year': year, 'month': month, 'day': day}, errors='coerce')
        except Exception:
            continue
    
    return df_combined

def auto_create_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Automatically create a unified datetime column `date_x` for mapping."""
    df_auto = df.copy()

    if 'date_x' in df_auto.columns:
        try:
            parsed = pd.to_datetime(df_auto['date_x'], errors='coerce')
            if parsed.notna().sum() > 0:
                df_auto['date_x'] = parsed
                return df_auto
        except Exception:
            pass

    candidate_cols = [c for c in df_auto.columns if any(k in c.lower() for k in ['date', 'datetime', 'timestamp'])]
    candidate_cols.extend([c for c in df_auto.select_dtypes(include=["datetime64[ns]"]).columns if c not in candidate_cols])
    
    best_col = None
    best_valid = -1
    for col in candidate_cols:
        try:
            parsed = pd.to_datetime(df_auto[col], errors='coerce')
            valid = parsed.notna().sum()
            if valid > best_valid:
                best_valid = valid
                best_col = col
        except Exception:
            continue

    if best_col is not None and best_valid > 0:
        try:
            df_auto['date_x'] = pd.to_datetime(df_auto[best_col], errors='coerce')
            try:
                df_auto['date_x'] = df_auto['date_x'].dt.tz_localize(None)
            except Exception:
                pass
            df_auto['date_x'] = df_auto['date_x'].dt.floor('ms')
            return df_auto
        except Exception:
            pass

    df_combined = detect_and_combine_date_columns(df_auto)
    combined_cols = [c for c in df_combined.columns if c.startswith('datetime_combined_')]
    best_combined = None
    best_combined_valid = -1
    for col in combined_cols:
        try:
            valid = df_combined[col].notna().sum()
            if valid > best_combined_valid:
                best_combined_valid = valid
                best_combined = col
        except Exception:
            continue

    if best_combined is not None and best_combined_valid > 0:
        df_combined['date_x'] = pd.to_datetime(df_combined[best_combined], errors='coerce')
        try:
            df_combined['date_x'] = df_combined['date_x'].dt.tz_localize(None)
        except Exception:
            pass
        df_combined['date_x'] = df_combined['date_x'].dt.floor('ms')
        return df_combined

    return df_auto
