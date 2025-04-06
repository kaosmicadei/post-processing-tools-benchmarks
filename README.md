# Benchmarks

## Running
Create and activate an Python environment.
```bash
$ python3 -m venv .venv && source .venv/bin/activate`
```

Compile the libs using the PyO3 tool `maturin`.
```bash
$ cd magnetization && maturin develop --release && cd ..
$ cd tensor && maturin develop --release && cd ..
```