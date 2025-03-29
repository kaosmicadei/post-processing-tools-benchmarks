use pyo3::prelude::*;

#[pymodule]
mod magnetization {
  use crate::core::Measurement;
  use super::*;

  #[pyfunction]
  fn total_magnetization(json_data: &str) -> PyResult<f64> {
    let measures = Measurement::from_str(json_data);
    Ok(measures.total_magnetization())
  }
}
