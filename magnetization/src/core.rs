use std::collections::HashMap;
use serde::{Serialize, Deserialize};
use serde_json;

fn bitstring_to_int(bitstring: &str) -> Result<i64, String> {
  let as_num = u64::from_str_radix(bitstring, 2)
    .map_err(|e| format!("Invalid bitstring: {}", e))?;
  Ok(as_num as i64)
}

fn partial_magnetization(size: i64, bitstring: i64, count: i64) -> i64 {
  (((bitstring.count_ones() << 1) as i64) - size) * count
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Measurement {
  bitstring_size: i64,
  counts: HashMap<String, i64>
}

impl Measurement {
  pub fn from_str(json_data: &str) -> Result<Measurement, String> {
    serde_json::from_str(json_data)
      .map_err(|e| format!("Invalid JSON fromat: {}", e))
  }

  pub fn total_magnetization(&self) -> Result<f64, String> {
    let mut total_counts: i64 = 0;
    let mut total_mag: i64 = 0;
    for (key, &value) in self.counts.iter() {
      let bstr = bitstring_to_int(key)?;
      total_mag += partial_magnetization(self.bitstring_size, bstr, value);
      total_counts += value;
    }

    Ok((total_mag as f64) / (total_counts as f64))
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_load_str() {
    let json_data = r#"{
      "bitstring_size": 4,
      "counts": {
        "0000": 12,
        "0101": 3,
        "0110": 5
      }
    }"#;
    
    let data = Measurement::from_str(json_data).unwrap();
    let counts = [
      ("0000".to_string(), 12),
      ("0101".to_string(), 3),
      ("0110".to_string(), 5)
    ];
    let target = Measurement { bitstring_size: 4, counts: HashMap::from_iter(counts) };
    assert_eq!(data, target);
  }
}