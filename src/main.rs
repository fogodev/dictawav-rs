extern crate dictawav;

use std::collections::HashMap;

use dictawav::DictaWav;
use std::path::PathBuf;


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


    let mut classification_paths: HashMap<&str, Vec<PathBuf>> = HashMap::new();

    for word in words {
        for file_number in 1..6 {
            let mut path = std::env::current_dir().unwrap();
            path.push("dataset");
            path.push(word);
            path.push(file_number.to_string());
            path.set_extension("wav");
            let mut paths = classification_paths.entry(word).or_insert(Vec::with_capacity(5));
            paths.push(path.clone());
        }
    }

    let mut accuracy_mean = 0.0;
    let num_tests = 10;
    for _ in 0..num_tests {
        accuracy_mean += run_tests_leave_one_out(classification_paths.clone());
    }

    println!("Total accuracy on {} tests: {}%", num_tests, 100.0 * (accuracy_mean / num_tests as f64));
}

fn run_tests_leave_one_out(classification_paths: HashMap<&str, Vec<PathBuf>>) -> f64 {
    let mut got_right = 0usize;
    let total_words = classification_paths.len() * 5;

    let mut dictawav = DictaWav::new();

    for(word, file_paths) in classification_paths.iter() {
        for file_path in file_paths.iter() {
            dictawav.train(file_path, String::from(*word));
        }
    }


    for (word, file_paths) in classification_paths {
        for file_path in file_paths {
//            println!("Forgetting {:#?}", file_path);
            dictawav.forget(file_path.clone(), String::from(word));

            let (_confidence, (classification, _probability)) = dictawav.classification_confidence_and_probability(file_path.clone());
//            println!("Word: {}, classified as: {}, with probability = {}, and confidence = {}", word, classification, probability, confidence);
            if word == classification {
                got_right += 1;
            }

//            println!("Training again {:#?}\n", file_path);
            dictawav.train(file_path, String::from(word));
        }
    }
    let accuracy = got_right as f64 / total_words as f64;
    println!("Got {} words right from {}, {}% of accuracy", got_right, total_words, accuracy * 100.0);

    accuracy
}