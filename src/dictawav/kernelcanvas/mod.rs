mod kernel;

use std::f32;
use self::kernel::Kernel;
use self::kernel::KernelBuilder;

pub struct KernelCanvas {
    kernel_count: usize,
    kernel_dimension: usize,
    output_factor: usize,
    kernel_builder: KernelBuilder,
    kernels: Vec<Kernel>,
    active_kernels: Vec<bool>,
    processed_frames: Vec<Vec<f32>>,
}

impl KernelCanvas {
    pub fn new(kernel_count: usize, kernel_dimension: usize, output_factor: usize) -> KernelCanvas {
        let mut kernel_builder = KernelBuilder::new(kernel_dimension * 4usize);

        let mut kernels = Vec::with_capacity(kernel_count);
        for _ in 0..kernel_count {
            kernels.push(kernel_builder.build());
        }
        let active_kernels = Vec::new();
        let processed_frames = Vec::new();

        KernelCanvas {
            kernel_count,
            kernel_dimension,
            output_factor,
            kernel_builder,
            kernels,
            active_kernels,
            processed_frames,
        }
    }

    pub fn process(&mut self, frames: Vec<Vec<f32>>) {

        // Cleaning current canvas
        self.processed_frames = Vec::new();

        self.append_sum(frames);
        self.zscore_and_tanh();
        self.replicate_features();
    }

    pub fn get_painted_canvas(&mut self) -> Vec<bool> {
        self.paint_canvas();

        let mut painted_canvas = Vec::with_capacity(self.kernel_count * self.output_factor);

        for _ in 0..self.output_factor {
            painted_canvas.extend(self.active_kernels.iter());
        }

        self.clean_canvas();
        painted_canvas
    }

    fn append_sum(&mut self, frames: Vec<Vec<f32>>) {

        // First frame is a special case
        let mut first_frame = frames[0].clone();
        first_frame.extend(frames[0].iter());
        self.processed_frames.push(first_frame);

        // Other frames
        let mut previous_frame = frames[0].clone();
        for frame in frames.into_iter().skip(1) {
            let mut current_frame = frame.clone();
            current_frame.extend(previous_frame.iter()
                .zip(frame.iter())
                .map(|(a, b)| *a + *b)
            );

            self.processed_frames.push(current_frame);
            previous_frame = frame;
        }
    }

    fn zscore_and_tanh(&mut self) {
        let processed_frames_count = self.processed_frames.len();
        let mut processed = Vec::with_capacity(processed_frames_count);
        let doubled_kernel_dimension = self.kernel_dimension * 2usize;

        let mut means = vec![0f32; doubled_kernel_dimension];
        let mut std_deviations = vec![0f32; doubled_kernel_dimension];

        for frame in self.processed_frames.iter() {
            for (index, num) in means.iter_mut().enumerate() {
                *num += frame[index];
            }
        }

        for mean in means.iter_mut() {
            *mean /= processed_frames_count as f32; // Calculating the mean of each dimension
        }

        for frame in self.processed_frames.iter() {
            for (index, num) in std_deviations.iter_mut().enumerate() {
                *num += (frame[index] - means[index]).powf(2f32);
            }
        }

        for std_deviation in std_deviations.iter_mut() {
            *std_deviation /= (processed_frames_count - 1usize) as f32;
        }

        for frame in self.processed_frames.iter() {
            let mut zscored_frame = Vec::with_capacity(doubled_kernel_dimension);
            for index in 0..doubled_kernel_dimension {
                zscored_frame.push(
                    ((frame[index] - means[index]) / std_deviations[index]).tanh()
                );
            }
            processed.push(zscored_frame);
        }

        self.processed_frames = processed;
    }

    fn replicate_features(&mut self) {
        let doubled_kernel_dimension = self.kernel_dimension * 2usize;

        // "Replicating features" on the first frame just fill it with zeros
        for _ in 0..doubled_kernel_dimension {
            self.processed_frames[0].push(0f32);
        }

        for index in 1..self.processed_frames.len() {
            let previous_frame: Vec<f32> = self.processed_frames[index - 1].iter().take(doubled_kernel_dimension).cloned().collect();
            self.processed_frames[index].extend(previous_frame.into_iter());
        }
    }

    fn get_nearest_kernel_index(&self, frame: Vec<f32>) -> usize {
        let current_kernel = self.kernel_builder.build_from_coordinates(frame);
        let mut nearest_kernel_index = 0usize;
        let mut nearest_kernel_distance = f32::MAX;

        for index in 0..self.kernel_count {
            let distance = self.kernels[index].check_distance_squared(&current_kernel);
            if distance < nearest_kernel_distance {
                nearest_kernel_distance = distance;
                nearest_kernel_index = index;
            }
        }

        nearest_kernel_index
    }

    fn paint_canvas(&mut self) {
        let mut active_kernels = vec![false; self.kernel_count];
        for frame in self.processed_frames.iter() {
            active_kernels[self.get_nearest_kernel_index(frame.clone())] = true;
        }

        self.active_kernels = active_kernels;
    }

    fn clean_canvas(&mut self) {
        for active in self.active_kernels.iter_mut() {
            *active = false;
        }
    }

}