mod fft_handler;
mod mfcc;

use std::f64;
use self::fft_handler::FFTHandler;
use self::mfcc::MFCC;

type Frame = Vec<f64>;

pub struct PreProcessor {
    samples_per_frame: usize,
    processed_frames: Vec<Frame>,
    fft_handler: FFTHandler,
    mfcc: mfcc::MFCC,
}

impl PreProcessor {
    pub fn new(sample_rate: usize) -> PreProcessor {
        // Divide by 50 to get approximately 20ms sized frames
        let samples_per_frame = PreProcessor::get_next_power_of_2(sample_rate / 50usize);
        let processed_frames: Vec<Frame> = Vec::new();
        let fft_handler = FFTHandler::new(samples_per_frame);
        let filterbank_count = 26usize;
        let lowest_frequency = 0f64;
        let highest_frequency = PreProcessor::get_highest_frequency(sample_rate);
        let mfcc = MFCC::new(filterbank_count,
                             sample_rate,
                             samples_per_frame,
                             lowest_frequency,
                             highest_frequency,
        );

        PreProcessor {
            samples_per_frame,
            processed_frames,
            fft_handler,
            mfcc,
        }
    }

    pub fn process(&mut self, audio_data: Vec<f64>) {
        let frame_mid_point = self.samples_per_frame / 2usize;

        let mut first_frame = Frame::with_capacity(self.samples_per_frame);
        let mut second_frame = Frame::with_capacity(self.samples_per_frame);
        let mut third_frame_first_half = Frame::with_capacity(self.samples_per_frame);
        let mut third_frame_complete = Frame::new();

        let mut sample_counter = 0usize;

        for sample in audio_data {
            if sample_counter < frame_mid_point {
                if !third_frame_complete.is_empty() {
                    self.push_windowed_sample(sample, &mut third_frame_complete);
                }
                self.push_windowed_sample(sample, &mut first_frame);
            } else if sample_counter >= frame_mid_point && sample_counter < self.samples_per_frame {
                if !third_frame_complete.is_empty() && third_frame_complete.len() == self.samples_per_frame {
                    self.process_and_add_frame(third_frame_complete);
                    third_frame_complete = Frame::new();
                }

                self.push_windowed_sample(sample, &mut first_frame);
                self.push_windowed_sample(sample, &mut second_frame);
            } else {
                self.push_windowed_sample(sample, &mut second_frame);
                self.push_windowed_sample(sample, &mut third_frame_first_half);
            }

            if third_frame_first_half.len() == frame_mid_point {
                third_frame_complete = third_frame_first_half;
                third_frame_first_half = Frame::with_capacity(self.samples_per_frame);
            }

            if first_frame.len() == self.samples_per_frame {
                self.process_and_add_frame(first_frame);
                first_frame = Frame::with_capacity(self.samples_per_frame);
            }

            if second_frame.len() == self.samples_per_frame {
                self.process_and_add_frame(second_frame);
                second_frame = Frame::with_capacity(self.samples_per_frame);
            }

            if sample_counter > self.samples_per_frame + frame_mid_point {
                sample_counter = 0usize;
            } else {
                sample_counter += 1usize;
            }
        }

        // Adding remaining samples on incomplete frames
        self.check_fill_and_add_incomplete_frame(first_frame);
        self.check_fill_and_add_incomplete_frame(second_frame);
        self.check_fill_and_add_incomplete_frame(third_frame_first_half);
        self.check_fill_and_add_incomplete_frame(third_frame_complete);
    }

    pub fn extract_processed_frames(self) -> Vec<Frame> {
        self.processed_frames
    }

    #[inline]
    fn process_and_add_frame(&mut self, frame: Frame) {
        self.processed_frames.push(
            self.mfcc.compute(
                &self.fft_handler.process(frame)
            )
        );
    }

    #[inline]
    fn push_windowed_sample(&self, sample: f64, frame: &mut Frame) {
        let length = frame.len();
        frame.push(sample * self.hann_window_funcion(length));
    }

    #[inline]
    fn check_fill_and_add_incomplete_frame(&mut self, mut frame: Frame) {
        if !frame.is_empty() {
            while frame.len() < self.samples_per_frame {
                frame.push(0f64);
            }
            self.process_and_add_frame(frame);
        }
    }

    #[inline]
    fn hann_window_funcion(&self, index: usize) -> f64 {
        0.5f64 * (1f64 - ((2f64 * f64::consts::PI * index as f64) / self.samples_per_frame as f64).cos())
    }

    #[inline]
    fn get_next_power_of_2(num: usize) -> usize {
        let mut base2 = 1usize;
        while base2 <= num {
            base2 <<= 1;
        }
        base2
    }

    #[inline]
    fn get_highest_frequency(sample_rate: usize) -> f64 {
        sample_rate as f64 / 2f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn next_power_of_2() {
        let num32 = PreProcessor::get_next_power_of_2(31usize);
        assert_eq!(num32, 32usize);
        let num1024 = PreProcessor::get_next_power_of_2(844usize);
        assert_eq!(num1024, 1024usize);
        let num2 = PreProcessor::get_next_power_of_2(1usize);
        assert_eq!(num2, 2usize);
    }
}