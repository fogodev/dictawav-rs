extern crate rand;

use self::rand::distributions::{IndependentSample, Range};
use self::rand::ThreadRng;
use std::f32;

pub struct Kernel {
    coordinates: Vec<f32>,
}

impl Kernel {
    pub fn check_distance_squared(&mut self, other: &Kernel) -> f32 {
        let mut distance_squared = 0f32;
        for (index, coordinate) in self.coordinates.iter().enumerate() {
            distance_squared += (*coordinate - other.coordinates[index]).powf(2f32)
        }

        distance_squared
    }
}

pub struct KernelBuilder {
    dimension: usize,
    range: Range<f32>,
    random_generator: ThreadRng,
}

impl KernelBuilder {
    pub fn new(dimension: usize) -> KernelBuilder {
        let range = Range::new(-1f32, 1f32 + f32::MIN);
        let mut random_generator = rand::thread_rng();

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

    pub fn build_from_coordinates(&self, coordinates: Vec<f32>) -> Kernel {
        if self.dimension != coordinates.len() {
            panic!("KernelBuilder Error: Trying to build a kernel from coordinates with different dimension!");
        }

        Kernel {
            coordinates
        }
    }
}