//! Small helpers shared with `tools/puzzle-pipeline/src/util.ts`.

use serde_json::Value;

pub use crate::canonical::sha256_hex;

/// SHA-256 of a UTF-8 string, lowercase hex.
pub fn sha256_text(text: &str) -> String {
    sha256_hex(text.as_bytes())
}

/// `stableJson`: JSON with object keys sorted, used for config hashes.
pub fn stable_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let body = keys
                .iter()
                .map(|key| {
                    format!(
                        "{}:{}",
                        Value::String((*key).clone()),
                        stable_json(&map[*key])
                    )
                })
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        Value::Array(items) => {
            let body = items.iter().map(stable_json).collect::<Vec<_>>().join(",");
            format!("[{body}]")
        }
        other => other.to_string(),
    }
}

/// `buildTimestamp`: `SOURCE_DATE_EPOCH` as an ISO-8601 instant with milliseconds.
pub fn build_timestamp() -> String {
    match std::env::var("SOURCE_DATE_EPOCH")
        .ok()
        .and_then(|raw| raw.trim().parse::<i64>().ok())
    {
        None => "1970-01-01T00:00:00.000Z".to_string(),
        Some(epoch) => iso_8601(epoch),
    }
}

/// Formats Unix seconds as `YYYY-MM-DDTHH:MM:SS.000Z`.
pub fn iso_8601(epoch_seconds: i64) -> String {
    let days = epoch_seconds.div_euclid(86_400);
    let seconds = epoch_seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.000Z",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

/// Formats Unix milliseconds as Japan Standard Time with milliseconds,
/// `YYYY-MM-DDTHH:MM:SS.mmm+09:00` (the catalog records' `solvedAt` and
/// `reducedAt`, [`crate::record_clock`]; the owner reports in JST, 2026-10-01).
pub fn iso_8601_jst_ms(epoch_millis: i64) -> String {
    let local = epoch_millis + 9 * 3_600_000;
    let millis = local.rem_euclid(1000);
    let seconds = local.div_euclid(1000);
    let days = seconds.div_euclid(86_400);
    let seconds = seconds.rem_euclid(86_400);
    let (year, month, day) = civil_from_days(days);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}.{millis:03}+09:00",
        seconds / 3600,
        (seconds / 60) % 60,
        seconds % 60
    )
}

/// Howard Hinnant's `civil_from_days`.
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m as u32, d as u32)
}

/// Reads the leading 16 hex characters of a SHA-256 digest as a `u64`, the seed
/// derivation used by `reduceStage`.
pub fn seed_from_hex_prefix(hex: &str) -> u64 {
    u64::from_str_radix(&hex[..16], 16).unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_8601_matches_javascript() {
        assert_eq!(iso_8601(0), "1970-01-01T00:00:00.000Z");
        assert_eq!(iso_8601(1_700_000_000), "2023-11-14T22:13:20.000Z");
    }

    #[test]
    fn iso_8601_jst_is_nine_hours_ahead() {
        assert_eq!(iso_8601_jst_ms(0), "1970-01-01T09:00:00.000+09:00");
        assert_eq!(
            iso_8601_jst_ms(1_700_000_000_123),
            "2023-11-15T07:13:20.123+09:00"
        );
        // `SOURCE_DATE_EPOCH` of the builds (CLAUDE.md)
        assert_eq!(
            iso_8601_jst_ms(1_789_257_600_000),
            "2026-09-13T09:00:00.000+09:00"
        );
        assert_eq!(iso_8601_jst_ms(-1), "1970-01-01T08:59:59.999+09:00");
    }

    #[test]
    fn stable_json_sorts_keys() {
        let value: Value = serde_json::from_str(r#"{"b":1,"a":[2,{"d":3,"c":4}]}"#).unwrap();
        assert_eq!(stable_json(&value), r#"{"a":[2,{"c":4,"d":3}],"b":1}"#);
    }
}
