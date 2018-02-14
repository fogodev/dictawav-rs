extern crate rustfft;

use self::rustfft::algorithm::Radix4;
use self::rustfft::FFT;
use self::rustfft::num_complex::Complex;
use self::rustfft::num_traits::Zero;

pub struct FFTHandler {
    fft: Radix4<f32>,
    size: usize,
    input: Vec<Complex<f32>>,
    output: Vec<Complex<f32>>,
}

impl FFTHandler {
    pub fn new(size: usize) -> FFTHandler {
        let fft = Radix4::new(size, false);
        let input: Vec<Complex<f32>> = Vec::with_capacity(size);
        let output: Vec<Complex<f32>> = vec![Complex::zero(); size];

        FFTHandler { fft, size, input, output }
    }

    pub fn process(&mut self, input: Vec<f32>) -> Vec<f32> {
        for num in input.into_iter() {
            self.input.push(Complex::new(num, 0f32))
        }

        self.fft.process(&mut self.input, &mut self.output);

        // ToDo Check output
        let output: Vec<f32> = self.output
                                   .iter()
                                   .map(
                                       |complex|
                                           complex.norm()
                                   ).collect();

        self.input = Vec::with_capacity(self.size);
        self.output = vec![Complex::zero(); self.size];

        output
    }
}