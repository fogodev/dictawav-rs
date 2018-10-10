extern crate rustfft;

use self::rustfft::algorithm::Radix4;
use self::rustfft::FFT;
use self::rustfft::num_complex::Complex;
use self::rustfft::num_traits::Zero;

pub struct FFTHandler {
    fft: Radix4<f64>,
    size: usize,
    input: Vec<Complex<f64>>,
    output: Vec<Complex<f64>>,
}

impl FFTHandler {
    pub fn new(size: usize) -> FFTHandler {
        let fft = Radix4::new(size, false);
        let input: Vec<Complex<f64>> = Vec::with_capacity(size);
        let output: Vec<Complex<f64>> = vec![Complex::zero(); size];

        FFTHandler { fft, size, input, output }
    }

    pub fn process(&mut self, input: Vec<f64>) -> Vec<f64> {
        for num in input {
            self.input.push(Complex::new(num, 0f64))
        }

        self.fft.process(&mut self.input, &mut self.output);

        // ToDo Check output
        let output: Vec<f64> = self.output
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