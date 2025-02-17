use ndarray::{Array2, Array3, Axis, s};

fn init_grid(timesteps: usize, n: usize) -> Array3<f64> {
    let mut grid = Array3::<f64>::zeros((timesteps, n, n));
    grid.slice_mut(s![.., 0..1, ..]).fill(1.0);
    grid
}

fn diffuse(grid: Array2<f64>, diffusion_coeffficient:f64) -> Array2<f64> {
    let mut new_grid = grid.to_owned();
    new_grid
}

fn main() {
    let timesteps = 2;
    let n = 50;
    
    let grid = init_grid(timesteps, n);
}
