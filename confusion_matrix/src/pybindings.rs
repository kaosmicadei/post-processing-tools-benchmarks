use pyo3::prelude::*;

#[pymodule]
mod confusion_matrix {
  use super::*;
  use crate::core;
  use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};

  #[pyfunction]
  pub fn apply(py: Python<'_>, m: PyReadonlyArray2<f32>, data: PyReadonlyArray1<f32>) -> PyResult<Py<PyArray1<f32>>> {
    let mm = m.as_array().to_owned();
    let dd = data.as_array().to_owned();
    let res = core::apply(&mm, &dd).into_pyarray(py).unbind();
    Ok(res)
  }
}
