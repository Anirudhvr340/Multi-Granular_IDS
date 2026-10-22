"""Causal temporal feature engineering for 30-second window features."""

import numpy as np

HISTORY_WINDOWS = 4
DEFAULT_TEMPORAL_BASES = (
    "total_flows", "total_packets", "total_bytes", "flow_rate", "packet_rate", "byte_rate",
    "mean_flow_duration", "flow_duration_p50", "mean_iat", "mean_inter_arrival",
    "total_syn", "total_fin", "total_rst", "total_ack", "unique_dst_ports",
    "total_fwd_pkts", "total_bwd_pkts", "attack_ratio",
)


def _finite(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(values, -1e12, 1e12)


def fit_temporal_config(X, feature_names, train_indices=None, history=HISTORY_WINDOWS):
    """Fit thresholds on training rows only; no labels are used."""
    names = np.asarray(feature_names).astype(str)
    source = X if train_indices is None else X[np.asarray(train_indices)]
    config = {"history": int(history), "bases": [], "thresholds": {}}
    for name in DEFAULT_TEMPORAL_BASES:
        if name in names:
            index = int(np.flatnonzero(names == name)[0])
            config["bases"].append(name)
            values = _finite(source[:, index])
            config["thresholds"][name] = {
                "high": float(np.quantile(values, 0.90)),
                "low": float(np.quantile(values, 0.10)),
            }
    return config


def build_temporal_features(X, times, feature_names, config=None):
    """Build causal lag, change, rolling, trend, and persistence features.

    Row i only reads rows i and earlier. Missing history is represented by the
    current value, which is neutral for deltas and avoids future imputation.
    """
    X = _finite(X)
    times = np.asarray(times)
    names = np.asarray(feature_names).astype(str)
    config = config or fit_temporal_config(X, names)
    history = int(config.get("history", HISTORY_WINDOWS))
    bases = [name for name in config.get("bases", []) if name in names]
    columns = [X]
    output_names = list(names)
    name_to_index = {name: index for index, name in enumerate(names)}

    for base in bases:
        values = X[:, name_to_index[base]]
        for lag in range(1, history + 1):
            if len(values):
                previous = np.concatenate((
                    np.repeat(values[0], min(lag, len(values))),
                    values[:-lag] if lag < len(values) else np.array([], dtype=values.dtype),
                ))
            else:
                previous = values
            columns.append(previous[:, None])
            output_names.append(f"{base}_lag_{lag}")
            delta = values - previous
            columns.append(delta[:, None])
            output_names.append(f"{base}_delta_{lag}")
            pct = delta / (np.abs(previous) + 1e-6)
            columns.append(np.clip(pct, -1e6, 1e6)[:, None])
            output_names.append(f"{base}_pct_change_{lag}")

        rolling_mean = np.zeros(len(values))
        rolling_std = np.zeros(len(values))
        rolling_min = np.zeros(len(values))
        rolling_max = np.zeros(len(values))
        trend = np.zeros(len(values))
        persistence = np.zeros(len(values))
        threshold = config.get("thresholds", {}).get(base, {})
        high = threshold.get("high", float(np.quantile(values, 0.90)))
        low = threshold.get("low", float(np.quantile(values, 0.10)))
        for index in range(len(values)):
            start = max(0, index - history)
            context = values[start:index + 1]
            rolling_mean[index] = context.mean()
            rolling_std[index] = context.std()
            rolling_min[index] = context.min()
            rolling_max[index] = context.max()
            if len(context) >= 2:
                trend[index] = np.polyfit(np.arange(len(context), dtype=float), context, 1)[0]
            suspicious = context >= high
            if low > 0.0:
                suspicious |= context <= low
            persistence[index] = suspicious.sum()
        for suffix, values_out in (
            ("roll_mean_4", rolling_mean), ("roll_std_4", rolling_std),
            ("roll_min_4", rolling_min), ("roll_max_4", rolling_max),
            ("trend", trend), ("persistence_4", persistence),
        ):
            columns.append(values_out[:, None])
            output_names.append(f"{base}_{suffix}")

    # Cross-feature behavioral indicators use traffic statistics only.
    if "total_flows" in name_to_index and "mean_flow_duration" in name_to_index:
        flows = X[:, name_to_index["total_flows"]]
        durations = X[:, name_to_index["mean_flow_duration"]]
        duration_threshold = config.get("thresholds", {}).get("mean_flow_duration", {}).get("high", 0.0)
        long_ratio = (durations >= duration_threshold).astype(float)
        columns.append(long_ratio[:, None])
        output_names.append("long_duration_window_flag")
        for index in range(len(X)):
            start = max(0, index - history)
            long_ratio[index] = long_ratio[start:index + 1].sum()
        columns.append(long_ratio[:, None])
        output_names.append("long_duration_window_count_4")
        low_rate = (flows <= config.get("thresholds", {}).get("total_flows", {}).get("low", 0.0)).astype(float)
        columns.append(low_rate[:, None])
        output_names.append("low_flow_window_count_4")

    result = np.nan_to_num(np.column_stack(columns), nan=0.0, posinf=0.0, neginf=0.0)
    result = np.clip(result, -1e12, 1e12).astype(np.float32)
    return result, np.asarray(output_names, dtype=object)
