extern crate hound;

use std::path;
use std::{i8, i16, i32};

/// A handler to wav files using Hound library
///
/// WavHandler opens a wav file, retrieve some information like format, sample rate, bits per sample,
/// then reads the file data, storing data in a Vec<f32>, converting the file data if necessary
pub struct WavHandler {
    /// Specification of the file
    wav_spec: hound::WavSpec,
    /// Store of file's data
    audio_data: Vec<f32>,
}

impl WavHandler {
    /// Attempts to create a WavHandler object from a given filename, if file have a int 8, int 16
    /// or int 32, converts it's data to float 32, getting samples in a range from -1.0 to 1.0
    pub fn new<P: AsRef<path::Path>>(filename: P) -> hound::Result<WavHandler> {
        let mut wav_reader = hound::WavReader::open(filename)?;
        let wav_spec = wav_reader.spec();

        let mut audio_data: Vec<f32> = match wav_spec.sample_format {
            hound::SampleFormat::Float => wav_reader.samples::<f32>().map(
                |sample| sample.unwrap()
            ).collect(),
            hound::SampleFormat::Int => {
                match wav_spec.bits_per_sample {
                    8u16 => wav_reader.samples::<i8>()
                                      .map(
                                          |sample|
                                              (sample.unwrap() as f32) * (1f32 / (i8::MAX as f32 + 1f32))
                                      ).collect(),

                    16u16 => wav_reader.samples::<i16>()
                                       .map(
                                           |sample|
                                               (sample.unwrap() as f32) * (1f32 / (i16::MAX as f32 + 1f32))
                                       ).collect(),

                    32u16 => wav_reader.samples::<i32>()
                                       .map(
                                           |sample|
                                               (sample.unwrap() as f32) * (1f32 / (i32::MAX as f32 + 1f32))
                                       ).collect(),

                    _ => return Err(hound::Error::Unsupported)
                }
            }
        };

        if wav_spec.channels > 1u16 {
            audio_data = WavHandler::convert_to_mono(wav_spec.channels, audio_data);
        }

        Ok(WavHandler { wav_spec, audio_data })
    }

    // Convert a audio data with multiple channels to mono
    fn convert_to_mono(channels: u16, audio_data: Vec<f32>) -> Vec<f32> {
        let mut new_data: Vec<f32> = Vec::with_capacity((audio_data.len() as f32 / channels as f32) as usize);
        audio_data.chunks(channels as usize)
                  .for_each(
                      |chunk|
                          new_data.push(chunk.iter().sum::<f32>() / channels as f32)
                  );
        new_data
    }

    /// Extract the audio data, consuming the handler in process
    pub fn extract_audio_data(self) -> Vec<f32> {
        self.audio_data
    }

    /// Get the sample rate from wav file
    pub fn get_sample_rate(&self) -> u32 {
        self.wav_spec.sample_rate
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::hound::*;

    #[test]
    fn read_against_hound_test() {
        let mut wav_reader = WavReader::open("testsamples/waveformatex-ieeefloat-44100Hz-mono.wav").unwrap();
        let hound_samples: Vec<f32> = wav_reader.samples::<f32>().map(|sample| sample.unwrap()).collect();

        let wav_handler = WavHandler::new("testsamples/waveformatex-ieeefloat-44100Hz-mono.wav").unwrap();

        assert_eq!(hound_samples, wav_handler.audio_data);
    }

    #[test]
    fn read_i8bit_wav_file_test() { // Test based on Hound's tests
        let wav_handler = WavHandler::new("testsamples/pcmwaveformat-8bit-44100Hz-mono.wav").unwrap();
        let file_data = vec![19i8, -53i8, 89i8, -127i8];
        let file_data_f32_conversion: Vec<f32> = file_data.iter()
                                                          .map(|x| (*x as f32) * (1f32 / (i8::MAX as f32 + 1f32)))
                                                          .collect();

        assert_eq!(file_data_f32_conversion, wav_handler.audio_data);
    }

    #[test]
    fn read_i16bit_wav_file_test() { // Test based on Hound's tests
        let wav_handler = WavHandler::new("testsamples/pcmwaveformat-16bit-44100Hz-mono.wav").unwrap();
        let file_data = vec![2i16, -3i16, 5i16, -7i16];
        let file_data_f32_conversion: Vec<f32> = file_data.iter()
                                                          .map(|x| (*x as f32) * (1f32 / (i16::MAX as f32 + 1f32)))
                                                          .collect();

        assert_eq!(file_data_f32_conversion, wav_handler.audio_data);
    }

    #[test]
    fn read_i32bit_wav_file_test() { // Test based on Hound's tests
        let wav_handler = WavHandler::new("testsamples/waveformatextensible-32bit-48kHz-stereo.wav").unwrap();
        let file_data = vec![19, -229373, 33587161, -2147483497];
        let mut file_data_f32_conversion: Vec<f32> = file_data.iter()
                                                              .map(|x| (*x as f32) * (1f32 / (i32::MAX as f32 + 1f32)))
                                                              .collect();

        file_data_f32_conversion = WavHandler::convert_to_mono(wav_handler.wav_spec.channels, file_data_f32_conversion);

        assert_eq!(file_data_f32_conversion, wav_handler.audio_data);
    }

    #[test]
    fn convert_to_mono_test() {
        let data = vec![1f32, 3f32, 4f32, 8f32, 5f32, 11f32];
        let channels = 2u16;
        assert_eq!(WavHandler::convert_to_mono(channels, data), &[2f32, 6f32, 8f32]);
    }
}


