use ndarray::{array, Array1, Axis, Order};

fn bit_swap0(idx: usize, value: usize) -> usize {
  let x = (value ^ (value >> idx)) & 1;
  value ^ ((x << idx) | x)
}

fn main() {
  let data = Array1::from_iter(1..=8);
  let m = array![[1, 2], [3, 4]];

  let mut data = data;

  for i in 0..3 {
    let order: [usize; 8] = core::array::from_fn(|v| bit_swap0(i, v));

    let view = data.select(Axis(0), &order);
    let binding = view.to_shape(((2, 4), Order::ColumnMajor)).unwrap();
    let binding1 = m.dot(&binding);

    let major = if i < 2 { Order::ColumnMajor } else { Order::RowMajor };
    data = binding1.flatten_with_order(major).into_owned();
  }

  println!("{}", data);
  // [153, 351, 345, 791, 333, 763, 749, 1715]
}
