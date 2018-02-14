extern crate rustdct;

use self::rustdct::dct2::{DCT2, DCT2ViaFFT};
use self::rustdct::rustfft::FFTplanner;

pub struct DCTHandler {
    dct: DCT2ViaFFT<f32>,
    size: usize,
}

impl DCTHandler {
    pub fn new(size: usize) -> DCTHandler {
        let mut planner = FFTplanner::new(false);
        let dct = DCT2ViaFFT::new(planner.plan_fft(size));

        DCTHandler { dct, size }
    }

    pub fn process(&mut self, mut input: Vec<f32>) -> Vec<f32> {
        let mut output = vec![0f32; self.size];

        self.dct.process(&mut input, &mut output);

        output.into_iter().take(self.size / 2usize).collect()
    }
}