use pyo3::prelude::*;

#[pymodule]
mod confusion_matrix {
  use super::*;
  use crate::core;
  use nalgebra::{DMatrix, DVector};
  use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};

  #[pyfunction]
  pub fn apply(py: Python<'_>, m: PyReadonlyArray2<f32>, data: PyReadonlyArray1<f32>) -> PyResult<Py<PyArray1<f32>>> {
    let m_ng = DMatrix::from_row_slice(2, 2, m.as_slice().unwrap());
    let data_ng = DVector::from_column_slice(data.as_slice().unwrap());

    let res = core::apply(&m_ng, &data_ng).as_slice().to_pyarray(py).unbind();
    Ok(res)
  }
}
