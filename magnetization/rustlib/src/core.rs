use std::collections::HashMap;
use serde_json;

fn bitstring_to_int(bitstring: &str) -> i64 {
  u64::from_str_radix(bitstring, 2).expect("Invalid bitstring.") as i64
}

fn convert(original: HashMap<String, i64>) -> HashMap<i64, i64> {
  original
    .into_iter()
    .map(|(key, value)| (bitstring_to_int(&key), value))
    .collect()
}

fn partial_magnetization(size: i64, bitstring: i64, count: i64) -> i64 {
  (((bitstring.count_ones() << 1) as i64) - size) * count
}

#[derive(Debug)]
pub struct Measurement {
  bitstring_size: i64,
  counts: HashMap<i64, i64>
}

impl Measurement {
  pub fn from_str(json_data: &str) -> Measurement {
    let data: HashMap<String, i64> = serde_json::from_str(json_data)
      .expect("Invalid string format.");
    let bitstring_size = data.keys().next().unwrap().len() as i64;
    let counts = convert(data);
    Measurement { bitstring_size, counts }
  }

  pub fn total_magnetization(&self) -> f64 {
    let mut total_counts: i64 = 0;
    let mut total_mag: i64 = 0;
    for (&key, &value) in self.counts.iter() {
      total_mag += partial_magnetization(self.bitstring_size, key, value);
      total_counts += value;
    }

    (total_mag as f64) / (total_counts as f64)
  }
}
