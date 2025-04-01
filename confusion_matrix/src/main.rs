use ndarray::{array, Array1, Axis, Order};

fn bit_swap0(idx: usize, value: usize) -> usize {
  let x = (value ^ (value >> idx)) & 1;
  value ^ ((x << idx) | x)
}

fn main() {
  let data = Array1::from_iter(1..=8);
  let m = array![[1, 2], [3, 4]];

  let mut data = data.to_shape(((2, 4), Order::ColumnMajor)).unwrap();

  for i in 0..3 {
    // let order: [usize; 8] = core::array::from_fn(|v| bit_swap0(i, v));
    let order: Vec<_> = (0..8).map(|v| bit_swap0(i, v)).collect();

    let reordered = data
      .flatten_with_order(Order::ColumnMajor)
      .select(Axis(0), &order);

    let as_matirx = reordered
      .to_shape(((2, 4), Order::ColumnMajor))
      .unwrap();

    data = m.dot(&as_matirx).into();
  }

  println!("{}", data.flatten());
  // [153, 351, 345, 791, 333, 763, 749, 1715]
}
