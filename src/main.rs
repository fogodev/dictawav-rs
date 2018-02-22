#[macro_use]
extern crate structopt;
extern crate dictawav;

use std::path::PathBuf;
use structopt::StructOpt;
use std::io::BufReader;
use std::io::prelude::*;
use std::fs::File;
use std::collections::HashMap;
use dictawav::DictaWav;

#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(short = "t", long = "training-file", parse(from_os_str))]
    training_file: Option<PathBuf>
}

fn main(){
    let opt = Opt::from_args();

    let mut dictawav = DictaWav::new();

    if let Some(training_file) = opt.training_file {
        let wav_files = BufReader::new(File::open(training_file).unwrap());

        let mut filenames_and_classes = HashMap::new();
        for wav_file in wav_files.lines() {
            let line = wav_file.unwrap();
            let file_and_class = line.split_whitespace().collect::<Vec<&str>>();
            filenames_and_classes.insert(file_and_class[0].to_string(), file_and_class[1].to_string());
        }
        for (filename, class) in filenames_and_classes.into_iter() {
            dictawav.train(&filename, class);
        }
    }
}