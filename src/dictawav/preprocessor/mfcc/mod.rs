mod dct_handler;

use self::dct_handler::DCTHandler;
use super::Frame;

pub struct MFCC {
    filterbank_count: usize,
    sample_rate: usize,
    frame_size: usize,
    lowest_frequency: f32,
    highest_frequency: f32,
    dct_handler: DCTHandler,
    filterbank: Vec<[usize; 3]>,
}

impl MFCC {
    pub fn new(filterbank_count: usize,
               sample_rate: usize,
               frame_size: usize,
               lowest_frequency: f32,
               highest_frequency: f32,
    ) -> MFCC {
        let dct_handler = DCTHandler::new(filterbank_count);
        let filterbank = Vec::with_capacity(filterbank_count);
        let mut mfcc = MFCC {
            filterbank_count,
            sample_rate,
            frame_size,
            lowest_frequency,
            highest_frequency,
            dct_handler,
            filterbank,
        };
        mfcc.create_filterbank();

        mfcc
    }

    pub fn compute(&mut self, frame: Frame) -> Frame {
        let mut filtered_values = vec![0f32; self.filterbank_count];

        let mut current_filter = 0usize;
        for fb in self.filterbank.iter() {
            let begin = fb[0];
            let mid = fb[1];
            let end = fb[2];

            for pos in begin..mid {
                filtered_values[current_filter] += frame[pos] * (pos - begin) as f32 / (mid - begin) as f32;
            }

            for pos in mid..end {
                filtered_values[current_filter] += frame[pos] * (end - pos) as f32 / (end - mid) as f32;
            }

            current_filter += 1usize;
        }

        self.dct_handler.process(
            filtered_values.into_iter()
                           .map(|value|
                               value.ln()
                           ).collect()
        )
    }

    fn create_filterbank(&mut self) {
        let lowest_mel = MFCC::hertz_to_mel(self.lowest_frequency);
        let highest_mel = MFCC::hertz_to_mel(self.highest_frequency);
        let delta_mel = (highest_mel - lowest_mel) / (self.filterbank_count as f32 + 1f32);

        let mut bins = Vec::with_capacity(self.filterbank_count + 2);
        for pos in 0..(self.filterbank_count + 2) {
            bins.push(
                (
                    (self.frame_size as f32 + 1f32)
                        * MFCC::mel_to_hertz(lowest_mel + pos as f32 * delta_mel)
                        / self.sample_rate as f32
                ).floor() as usize
            )
        }

        for pos in 0..self.filterbank_count {
            let current_filterbank = [bins[pos], bins[pos + 1], bins[pos + 2]];
            self.filterbank.push(current_filterbank);
        }
    }

    #[inline]
    fn hertz_to_mel(hertz: f32) -> f32 {
        1127f32 * (1f32 + hertz / 700f32).ln()
    }

    #[inline]
    fn mel_to_hertz(mels: f32) -> f32 {
        700f32 * ((mels / 1127f32).exp() - 1f32)
    }
}