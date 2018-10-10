extern crate dictawav;

use std::collections::{HashMap, HashSet};

use dictawav::DictaWav;
use std::path::PathBuf;
use std::f64;

// KernelCanvas parameters
const KERNELS_COUNT: usize = 2048;
const KERNELS_DIMENSION: usize = 13;
const OUTPUT_FACTOR: usize = 10;

// WiSARD parameters
const RETINA_SIZE: usize = KERNELS_COUNT * OUTPUT_FACTOR;
const RAM_NUM_BITS: usize = 32;
const USE_BLEACHING: bool = true;
const MINIMUM_CONFIDENCE: f64 = 0.002;
const BLEACHING_THRESHOLD: u64 = 1;
const RANDOMIZE_POSITIONS: bool = true;
const IS_CUMULATIVE: bool = true;

fn main() {
    let words = vec![
        "a", "ah!", "ai!", "ainda", "ano", "assim", "até", "au!",
        "bom", "casa", "cem", "certo", "coisa", "com", "como", "dar",
        "de", "depois", "dia", "dois", "e", "ele", "em", "esse",
        "estar", "este", "eu", "fazer", "ficar", "grande", "haver", "homem",
        "hum!", "ir", "isso", "já", "mais", "mas", "melhor", "mesmo",
        "mil", "moça", "moço", "muito", "não", "o", "ou", "para",
        "pior", "poder", "por", "porque", "primeiro", "próprio", "quando", "que",
        "se", "senhor", "senhora", "ser", "sua", "também", "tempo", "ter",
        "ui!", "último", "um", "uma", "ver", "vez", "você"
    ];


    let mut classification_paths: HashMap<&str, HashSet<PathBuf>> = HashMap::new();

    for word in words {
        for file_number in 1..6 {
            let mut path = std::env::current_dir().unwrap();
            path.push("dataset");
            path.push(word);
            path.push(file_number.to_string());
            path.set_extension("wav");
            let mut paths = classification_paths.entry(word).or_insert_with(||HashSet::with_capacity(5));
            paths.insert(path.clone());
        }
    }

    let mut accuracies = Vec::with_capacity(10);
    let num_tests = 10;
    for _ in 0..num_tests {
        accuracies.push(run_tests_kfold(classification_paths.clone()));
    }

    let mean = accuracies.iter().sum::<f64>() / f64::from(num_tests);
    let standard_deviation = (accuracies.iter().map(|&x|  (x - mean) * (x - mean)).sum::<f64>() / f64::from(num_tests)).sqrt();

    println!("Total accuracy on {} tests: {}%", num_tests, 100.0 * mean);
    println!("Standard deviation on {} tests: {}%", num_tests, 100.0 * standard_deviation);
}

fn run_tests_kfold(classification_paths: HashMap<&str, HashSet<PathBuf>>) -> f64 {

    let mut dictawav = DictaWav::new(
        KERNELS_COUNT,
        KERNELS_DIMENSION,
        OUTPUT_FACTOR,
        RETINA_SIZE,
        RAM_NUM_BITS,
        USE_BLEACHING,
        MINIMUM_CONFIDENCE,
        BLEACHING_THRESHOLD,
        RANDOMIZE_POSITIONS,
        IS_CUMULATIVE
    );

    let total_words_per_fold = classification_paths.len();
    // 5 folds, each one with 1 path from each word
    let mut folds = vec![
        HashMap::with_capacity(total_words_per_fold),
        HashMap::with_capacity(total_words_per_fold),
        HashMap::with_capacity(total_words_per_fold),
        HashMap::with_capacity(total_words_per_fold),
        HashMap::with_capacity(total_words_per_fold),
    ];

    let mut summed_accuracy = 0f64;
    let num_folds = folds.len();

    for (word, file_paths) in classification_paths {
        for (index, file_path) in file_paths.into_iter().enumerate() {
            folds[index].insert(String::from(word), file_path);
        }
    }

    // Training all examples
    for fold in &folds {
        for (word, file_path) in fold {
            dictawav.train(file_path.clone(), word.clone());
        }
    }

    // K-Fold cross validation
    for current_testing_fold in folds {
        for (word, file_path) in &current_testing_fold {
            dictawav.forget(file_path.clone(), word);
        }

        let mut got_right = 0usize;
        for (word, file_path) in &current_testing_fold {
            if *word == dictawav.classify(file_path.clone()) {
                got_right += 1;
            }
        }
        summed_accuracy += got_right as f64 / total_words_per_fold as f64;

        for (word, file_path) in current_testing_fold {
            dictawav.train(file_path.clone(), word.clone());
        }
    }

    let accuracy = summed_accuracy / num_folds as f64;
    println!("Got {}% of accuracy", accuracy * 100.0);

    accuracy
}

//fn run_tests_leave_one_out(mut dictawav: DictaWav, classification_paths: HashMap<&str, HashSet<PathBuf>>) -> f64 {
//    let mut got_right = 0usize;
//    let total_words = classification_paths.len() * 5;
//
//
//
//    for(word, file_paths) in classification_paths.iter() {
//        for file_path in file_paths.iter() {
//            dictawav.train(file_path, String::from(*word));
//        }
//    }
//
//
//    for (word, file_paths) in classification_paths {
//        for file_path in file_paths {
////            println!("Forgetting {:#?}", file_path);
//            dictawav.forget(file_path.clone(), String::from(word));
//
//            let (_confidence, (classification, _probability)) = dictawav.classification_confidence_and_probability(file_path.clone());
////            println!("Word: {}, classified as: {}, with probability = {}, and confidence = {}", word, classification, probability, confidence);
//            if word == classification {
//                got_right += 1;
//            }
//
////            println!("Training again {:#?}\n", file_path);
//            dictawav.train(file_path, String::from(word));
//        }
//    }
//    let accuracy = got_right as f64 / total_words as f64;
//    println!("Got {} words right from {}, {}% of accuracy", got_right, total_words, accuracy * 100.0);
//
//    accuracy
//}