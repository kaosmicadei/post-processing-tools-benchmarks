use pyo3::prelude::*;

#[pymodule]
mod tensor {
  use super::*;
  use pyo3::exceptions;
  use crate::core;
  use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

  #[pyfunction]
  pub fn mat_vec_multiply(py: Python<'_>, m: PyReadonlyArray2<f32>, data: PyReadonlyArray1<f32>) -> PyResult<Py<PyArray1<f32>>> {
    let mm = m.transpose()?.to_owned_array();
    let dd = data.to_owned_array();
    let res = core::mat_vec_multiply(&mm, &dd)
      .map_err(|e| PyErr::new::<exceptions::PyValueError, _>(e))?;
    Ok(res.into_pyarray(py).unbind())
  }
}
