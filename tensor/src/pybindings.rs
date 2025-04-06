use pyo3::prelude::*;

#[pymodule]
mod tensor {
  use super::*;
  use crate::core;
  use numpy::{IntoPyArray, PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};

  #[pyfunction]
  pub fn mat_vec_multiply(py: Python<'_>, m: PyReadonlyArray2<f32>, data: PyReadonlyArray1<f32>) -> PyResult<Py<PyArray1<f32>>> {
    let mm = m.transpose().unwrap().to_owned_array();
    let dd = data.to_owned_array();
    let res = core::mat_vec_multiply(&mm, &dd).into_pyarray(py).unbind();
    Ok(res)
  }
}
