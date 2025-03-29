use pyo3::prelude::*;

#[pymodule]
mod magnetization {
  use super::*;
  use pyo3::exceptions;
  use crate::core::Measurement;

  #[pyfunction]
  fn total_magnetization(json_data: &str) -> PyResult<f64> {
    let measures = Measurement::from_str(json_data)
      .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e))?;
    
    measures.total_magnetization()
      .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e))
  }
}
