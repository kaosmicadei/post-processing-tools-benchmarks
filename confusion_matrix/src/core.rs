use ndarray::{Array1, Array2, Order};
use rayon::prelude::*;

fn bit_swap0(idx: usize, value: usize) -> usize {
  let x = (value ^ (value >> idx)) & 1;
  value ^ ((x << idx) | x)
}

pub fn apply(m: &Array2<f32>, data: &Array1<f32>) -> Array1<f32> {
  assert!(data.len().is_power_of_two(), "data must have 2^N elements");
  assert_eq!(m.dim(), (2, 2), "m must be a 2x2 matrix.");

  let rank = (data.len() as f32).log2() as usize;
  let half_len = data.len() >> 1;

  let mut res = data.to_shape((half_len, 2)).unwrap();
  let mut tmp = res.clone();

  for i in 0..rank {
    // Fast "flatten": access the raw position through slice.
    let raw_view = res.as_slice()  // WARNING! Only works with row major!
      .unwrap();
    let raw_tmp = tmp.as_slice_mut().unwrap();

    raw_tmp
      .par_iter_mut()
      .enumerate()
      .for_each(|(j, v)| *v = raw_view[bit_swap0(i, j)]);
      // Since this as borrowed as mutable, the transformation propagates back
      // to the `tmp` view.

    res.assign(&tmp.dot(m));
  }

  // Fast flatten transposed.
  res.flatten_with_order(Order::ColumnMajor).into_owned()
}