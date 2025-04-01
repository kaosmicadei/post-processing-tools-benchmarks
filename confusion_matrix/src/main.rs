use ndarray::{array, Array1, Array2, Axis, Order};

fn bit_swap0(idx: usize, value: usize) -> usize {
  let x = (value ^ (value >> idx)) & 1;
  value ^ ((x << idx) | x)
}

fn apply_confusion_matrix(m: &Array2<f32>, data: &Array1<f32>) -> Array1<f32> {
  assert!(data.len().is_power_of_two(), "data must have 2^N elements");
  assert_eq!(m.dim(), (2, 2), "m must be a 2x2 matrix.");

  let rank = (data.len() as f32).log2() as usize;
  let half_len = data.len() >> 1;

  let mut res = data
    .to_shape(((2, half_len), Order::ColumnMajor))
    .unwrap();

  for i in 0..rank {
    let order: Vec<_> = (0..8).map(|v| bit_swap0(i, v)).collect();

    let reordered = res
      .flatten_with_order(Order::ColumnMajor)
      .select(Axis(0), &order);

    let as_matirx = reordered
      .to_shape(((2, half_len), Order::ColumnMajor))
      .unwrap();

    res = m.dot(&as_matirx).into();
  }

    res.flatten().into_owned()
}

fn main() {
  let data = Array1::from_iter((1..=8).map(|i| i as f32));
  let m = array![[1.0, 2.0], [3.0, 4.0]];

  let result = apply_confusion_matrix(&m, &data);
  println!("{}", result);
  // [153, 351, 345, 791, 333, 763, 749, 1715]
}
