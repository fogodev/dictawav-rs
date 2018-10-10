use self::kernelcanvas::KernelCanvas;
use self::preprocessor::PreProcessor;
use self::wav_handler::WavHandler;
use self::wisard::Wisard;
use std::path;

mod wav_handler;
mod preprocessor;
mod kernelcanvas;
mod wisard;

pub struct DictaWav {
    kernelcanvas: KernelCanvas,
    wisard: Wisard,
}

impl DictaWav {
    pub fn new(kernelcanvas_kernels_count: usize,
               kernelcanvas_kernels_dimension: usize,
               kernelcanvas_output_factor: usize,
               wisard_retina_size: usize,
               wisard_ram_num_bits: usize,
               wisard_use_bleaching: bool,
               wisard_minimum_confidence: f64,
               wisard_bleaching_threshold: u64,
               wisard_randomize_positions: bool,
               wisard_is_cumulative: bool,
    ) -> DictaWav {
        let kernelcanvas = KernelCanvas::new(
            kernelcanvas_kernels_count,
            kernelcanvas_kernels_dimension,
            kernelcanvas_output_factor,
        );
        let wisard = Wisard::new(
            wisard_retina_size,
            wisard_ram_num_bits,
            wisard_use_bleaching,
            wisard_minimum_confidence,
            wisard_bleaching_threshold,
            wisard_randomize_positions,
            wisard_is_cumulative,
        );

        DictaWav {
            kernelcanvas,
            wisard,
        }
    }

    pub fn train<P: AsRef<path::Path>>(&mut self, wav_file: P, class_name: String) {
        let painted_canvas = self.read_and_process_wav_file(wav_file);
        self.wisard.train(class_name, &painted_canvas);
    }

    pub fn forget<P: AsRef<path::Path>>(&mut self, wav_file: P, class_name: &str) {
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
        let wav_handler = WavHandler::new(wav_file).unwrap();
        let mut preprocessor = PreProcessor::new(wav_handler.get_sample_rate() as usize);
        preprocessor.process(wav_handler.extract_audio_data());

        self.kernelcanvas.process(preprocessor.extract_processed_frames());
        self.kernelcanvas.get_painted_canvas()
    }
}


#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
