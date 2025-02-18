use ndarray::{Array2, Array3, ArrayView2, s};

fn init_grid(timesteps: usize, n: usize) -> Array3<f64> {
    let mut grid = Array3::<f64>::zeros((timesteps, n, n));
    grid.slice_mut(s![.., 0..1, ..]).fill(1.0);
    grid
}

fn diffuse(grid: &ArrayView2<f64>, diffusion_coefficient: f64) -> Array2<f64> {
    let mut new_grid = grid.to_owned();
    let n = new_grid.shape()[0];

    for i in 0..n {
        for j in 1..n - 1 {
            new_grid[(i, j)] += diffusion_coefficient * (
                grid[((i + 1) % n, j)]
                + grid[((i - 1 + n) % n, j)]
                + grid[(i, j + 1)]
                + grid[(i, j - 1)]
                - 4.0 * grid[(i, j)]
            );
        }
    }

    new_grid
}

fn main() {
    let timesteps = 2;
    let n = 4;

    let mut full_grid = init_grid(timesteps, n);
    
    for t in 0..timesteps {
        let spatial_grid = diffuse(&full_grid.slice(s![t, .., ..]), 0.4);
        full_grid.slice_mut(s![t, .., ..]).assign(&spatial_grid);
    }
    println!("Updated full grid after diffusion:\n{:?}", full_grid);
}   
