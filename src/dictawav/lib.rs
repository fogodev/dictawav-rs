use self::kernelcanvas::KernelCanvas;
use self::preprocessor::PreProcessor;
use self::wav_handler::WavHandler;
use self::wisard::Wisard;
use std::{collections::HashMap, path};

mod wav_handler;
mod preprocessor;
mod kernelcanvas;
mod wisard;


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

pub struct DictaWav {
    kernelcanvas: KernelCanvas,
    wisard: Wisard,
    processed_painted_canvas_cache: HashMap<String, Vec<bool>>,
}

impl DictaWav {
    pub fn new() -> DictaWav {
        let kernelcanvas = KernelCanvas::new(KERNELS_COUNT, KERNELS_DIMENSION, OUTPUT_FACTOR);
        let wisard = Wisard::new(
            RETINA_SIZE,
            RAM_NUM_BITS,
            USE_BLEACHING,
            MINIMUM_CONFIDENCE,
            BLEACHING_THRESHOLD,
            RANDOMIZE_POSITIONS,
            IS_CUMULATIVE,
        );

        DictaWav {
            kernelcanvas,
            wisard,
            processed_painted_canvas_cache: HashMap::new(),
        }
    }

    pub fn train<P: AsRef<path::Path>>(&mut self, wav_file: P, class_name: String) {
        let painted_canvas = self.read_and_process_wav_file(wav_file);
        self.wisard.train(class_name, &painted_canvas);
    }

    pub fn forget<P: AsRef<path::Path>>(&mut self, wav_file: P, class_name: String) {
        let painted_canvas = self.read_and_process_wav_file(wav_file);
        self.wisard.forget(class_name, &painted_canvas);
    }

    pub fn classify<P: AsRef<path::Path>>(&mut self, wav_file: P) -> String {
        let painted_canvas = self.read_and_process_wav_file(wav_file);
        self.wisard.classify(&painted_canvas)
    }

    pub fn classification_and_probability<P: AsRef<path::Path>>(&mut self, wav_file: P) -> (String, f64) {
        let painted_canvas = self.read_and_process_wav_file(wav_file);
        self.wisard.classification_and_probability(&painted_canvas)
    }

    pub fn classification_confidence_and_probability<P: AsRef<path::Path>>(&mut self, wav_file: P) -> (f64, (String, f64)) {
        let painted_canvas = self.read_and_process_wav_file(wav_file);
        self.wisard.classification_confidence_and_probability(&painted_canvas)
    }

    fn read_and_process_wav_file<P: AsRef<path::Path>>(&mut self, wav_file: P) -> Vec<bool> {
        let string_wav_file = String::from(wav_file.as_ref().to_str().unwrap());

        if !self.processed_painted_canvas_cache.contains_key(&string_wav_file) {
            let wav_handler = WavHandler::new(wav_file).unwrap();
            let mut preprocessor = PreProcessor::new(wav_handler.get_sample_rate() as usize);
            preprocessor.process(wav_handler.extract_audio_data());

            self.kernelcanvas.process(preprocessor.extract_processed_frames());
            self.processed_painted_canvas_cache.insert(string_wav_file.clone(), self.kernelcanvas.get_painted_canvas());
        }

        self.processed_painted_canvas_cache.get(&string_wav_file).unwrap().clone()
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
