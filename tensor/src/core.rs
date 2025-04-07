use ndarray::{Array1, Array2, Order};
use rayon::prelude::*;


/// Swaps the bit at position `idx` with the least significant bit (LSB) in the
/// given `value`.
/// 
/// # Example
/// ```rust
/// let result = bit_swap0(2, 0b0110);
/// assert_eq!(result, 0b0011);
/// ```
fn bit_swap0(idx: usize, value: usize) -> usize {
  let x = (value ^ (value >> idx)) & 1;
  value ^ ((x << idx) | x)
}


/// Computes the product of a `2x2` matrix `m` with a `2^N`-length vector `data`
/// without explicitly constructing the Kronecker product, avoiding exponential
/// memory growth.
///
/// This function is parallelized using the `rayon` crate for improved
/// performance.
/// 
/// # Examples
/// ```rust
/// use ndarray::{array, Array1};
///
/// let m = array![[1.0, 3.0], [2.0, 4.0]];
/// let data = Array1::from_iter((1..=8).map(|i| i as f32));
/// let result = mat_vec_multiply(&m, &data);
/// let expect = array![153.0, 351.0, 345.0, 791.0, 333.0, 763.0, 749.0, 1715.0];
/// assert_eq!(result, expect);
/// ```
///
/// # Panics
/// - If `m` is not `2x2`, panics with `"m must be a 2x2 matrix."`.
/// - If `data.len()` is not a power of two, panics with `"data must have 2^N
///   elements"`.
///
/// # Implementation Details
/// - Reshapes `data` into a 2D array, applies `m` iteratively.
/// - Performs a bit-reversal permutations at each iteraction using `bit_swap0`.
/// - Returns the result flattened in column-major order (equivalent to tranpse
///   then flatten).
pub fn mat_vec_multiply(m: &Array2<f32>, data: &Array1<f32>) -> Result<Array1<f32>, String> {
  if !data.len().is_power_of_two() {
    return Err("data must have 2^N elements".to_string())
  }

  if m.dim() != (2, 2) {
    return Err("m must be a 2x2 matrix.".to_string())
  }

  let rank = (data.len() as f32).log2() as usize;
  let half_len = data.len() >> 1;

  let mut res = data.to_shape((half_len, 2)).unwrap();
  let mut tmp = res.clone();

  for i in 0..rank {
    // Fast flatten: access the raw positions through slice.
    let raw_view = res
      .as_slice()  // WARNING! Only works with row major!
      .unwrap();
    let raw_tmp = tmp.as_slice_mut().unwrap();

    raw_tmp
      .par_iter_mut()
      .enumerate()
      .for_each(|(j, v)| *v = raw_view[bit_swap0(i, j)]);
      // Since this was borrowed as mutable, the transformation propagates back
      // to the `tmp` view.

    res.assign(&tmp.dot(m));
  }

  // Fast flatten transposed.
  Ok(res.flatten_with_order(Order::ColumnMajor).into_owned())
}


#[cfg(test)]
mod tests {
  use super::*;
  use ndarray::array;

  #[test]
  fn test_bitswap() {
    assert_eq!(bit_swap0(2, 0b0110), 0b0011);
  }

  #[test]
  fn test_multiply_m() {
    let m = array![[1.0, 3.0], [2.0, 4.0]];
    let data = Array1::from_iter((1..=8).map(|i| i as f32));
    let result = mat_vec_multiply(&m, &data);
    let expect = array![153.0, 351.0, 345.0, 791.0, 333.0, 763.0, 749.0, 1715.0];
    assert_eq!(result, expect);
  }
}
