mod kernel;

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
        let active_kernels = vec![false; kernel_count];
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
        self.clean_canvas();

        self.append_sum(frames);
        self.zscore_and_tanh();
        self.replicate_features();
    }

    pub fn get_painted_canvas(&mut self) -> Vec<bool> {
        let mut painted_canvas = Vec::with_capacity(self.kernel_count * self.output_factor);

        for _ in 0..self.output_factor {
            painted_canvas.extend(self.active_kernels.iter());
        }

        painted_canvas
    }

    fn append_sum(&mut self, frames: Vec<Vec<f32>>) {

        // First frame is a special case
        let mut first_frame = frames[0];
        first_frame.extend(frames[0].iter());
        self.processed_frames.push(first_frame);

        // Other frames
        let mut previous_frame = frames[0];
        for frame in frames.iter().skip(1) {
            let mut current_frame = frame.clone();
            current_frame.extend(previous_frame.iter()
                                               .zip(frame.iter())
                                               .map(|(a, b)| *a + *b)
            );

            self.processed_frames.push(current_frame);
            previous_frame = *frame;
        }
    }
}