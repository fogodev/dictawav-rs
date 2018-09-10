extern crate rand;

use self::rand::distributions::{IndependentSample, Range};
use self::rand::ThreadRng;
use std::f64;

pub struct Kernel {
    coordinates: Vec<f64>,
}

impl Kernel {
    pub fn check_distance_squared(&self, other: &Kernel) -> f64 {
        let mut distance_squared = 0f64;
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            distance_squared += (*coordinate - other.coordinates[index]).powf(2f64)
        }

        distance_squared
    }
}

pub struct KernelBuilder {
    dimension: usize,
    range: Range<f64>,
    random_generator: ThreadRng,
}

impl KernelBuilder {
    pub fn new(dimension: usize) -> KernelBuilder {
        let range = Range::new(-1f64, 1f64 + f64::MIN_POSITIVE);
        let random_generator = rand::thread_rng();

        KernelBuilder {
            dimension,
            range,
            random_generator
        }
    }

    pub fn build(&mut self) -> Kernel {
        let mut coordinates = Vec::with_capacity(self.dimension);

        for _ in 0..self.dimension {
            coordinates.push(self.range.ind_sample(&mut self.random_generator));
        }

        Kernel {
            coordinates
        }
    }

    pub fn build_from_coordinates(&self, coordinates: Vec<f64>) -> Kernel {
        if self.dimension != coordinates.len() {
            panic!("KernelBuilder Error: Trying to build a kernel from coordinates with different dimension!");
        }

        Kernel {
            coordinates
        }
    }
}