mod wav_handler;
mod preprocessor;
mod kernelcanvas;
mod wisard;

use std::path;

use self::wav_handler::WavHandler;
use self::preprocessor::PreProcessor;
use self::kernelcanvas::KernelCanvas;
use self::wisard::Wisard;

// KernelCanvas parameters
const KERNELS_COUNT: usize = 256usize;
const KERNELS_DIMENSION: usize = 13usize;
const OUTPUT_FACTOR: usize = 2usize;

// WiSARD parameters
const RETINA_SIZE: usize = KERNELS_COUNT * OUTPUT_FACTOR;
const RAM_NUM_BITS: usize = 24usize;
const USE_BLEACHING: bool = true;
const MINIMUM_CONFIDENCE: f64 = 0.1f64;
const BLEACHING_THRESHOLD: u64 = 1;
const RANDOMIZE_POSITIONS: bool = true;
const IS_CUMULATIVE: bool = true;

pub struct DictaWav<'a> {
    kernelcanvas: KernelCanvas,
    wisard: Wisard<'a>
}

impl<'a> DictaWav<'a> {
    pub fn new() -> DictaWav<'a> {
        let kernelcanvas = KernelCanvas::new(KERNELS_COUNT, KERNELS_DIMENSION, OUTPUT_FACTOR);
        let wisard = Wisard::new(
            RETINA_SIZE,
            RAM_NUM_BITS,
            USE_BLEACHING,
            MINIMUM_CONFIDENCE,
            BLEACHING_THRESHOLD,
            RANDOMIZE_POSITIONS,
            IS_CUMULATIVE
        );

        DictaWav {
            kernelcanvas,
            wisard
        }
    }

    pub fn train<P: AsRef<path::Path>>(&'a mut self, wav_file: P, class_name: String) {
        self.read_and_process_wav_file(wav_file);
        self.wisard.train(class_name, &self.kernelcanvas.get_painted_canvas());
    }

    pub fn classify<P: AsRef<path::Path>>(&mut self, wav_file: P ) -> String {
        self.read_and_process_wav_file(wav_file);
        self.wisard.classify(&self.kernelcanvas.get_painted_canvas())
    }

    fn read_and_process_wav_file<P: AsRef<path::Path>>(&mut self, wav_file: P) {
        let mut wav_handler = WavHandler::new(wav_file).unwrap();
        let mut preprocessor = PreProcessor::new(wav_handler.get_sample_rate() as usize);

        preprocessor.process(wav_handler.extract_audio_data());

        self.kernelcanvas.process(preprocessor.extract_processed_frames());
    }
}



#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
