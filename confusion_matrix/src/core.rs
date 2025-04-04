use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

fn bit_swap0(idx: usize, value: usize) -> usize {
  let x = (value ^ (value >> idx)) & 1;
  value ^ ((x << idx) | x)
}

pub fn apply(m: &DMatrix<f32>, data: &DVector<f32>) -> DVector<f32> {
  assert!(data.len().is_power_of_two(), "data must have 2^N elements");

  let len = data.len();
  let rank = len.trailing_zeros() as usize;
  let half_len = len / 2;

  let mut res = DMatrix::from_column_slice(2, half_len, data.as_slice());
  let mut tmp = res.clone();

  for i in 0..rank {
    let raw_view = res.as_slice();
    let raw_tmp = tmp.as_mut_slice();

    raw_tmp
      .par_iter_mut()
      .enumerate()
      .for_each(|(j, v)| *v = raw_view[bit_swap0(i, j)]);
      // Since this as borrowed as mutable, the transformation propagates back
      // to the `tmp` view.

    res = m * &tmp;
  }

  // Fast flatten transposed.
  DVector::from_column_slice(res.transpose().as_slice())
}

#[cfg(test)]
mod tests {
  use super::*;
  use nalgebra::{DMatrix, DVector};

  #[test]
  fn test_bitswap() {
    assert_eq!(bit_swap0(2, 0b0110), 0b0011);
  }

  #[test]
  fn test_multiply_m() {
    let data = DVector::from_fn(8, |i, _| (i+1) as f32);
    let m = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
    let result = apply(&m, &data);
    let target = DVector::from_row_slice(&[153.0, 351.0, 345.0, 791.0, 333.0, 763.0, 749.0, 1715.0]);
    assert_eq!(result, target);
  }
}