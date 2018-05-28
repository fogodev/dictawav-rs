extern crate dictawav;

use std::collections::BTreeMap;

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

    let mut dictawav = DictaWav::new();
    let mut classification_paths: BTreeMap<&str, Vec<PathBuf>> = BTreeMap::new();

    for word in words {
        for file_number in 1..6 {
            let mut path = std::env::current_dir().unwrap();
            path.push("dataset");
            path.push(word);
            path.push(file_number.to_string());
            path.set_extension("wav");
            let mut paths = classification_paths.entry(word).or_insert(Vec::with_capacity(5));
            paths.push(path.clone());
            dictawav.train(path, String::from(word));
            }
        }


    for (word, file_paths) in classification_paths {

        for file_path in file_paths {
            println!("Forgetting {:#?}", file_path);
            dictawav.forget(file_path.clone(), String::from(word));

            let (confidence, (classification, probability)) = dictawav.classification_confidence_and_probability(file_path.clone());
            println!("Word: {}, classified as: {}, with probability = {}, and confidence = {}", word, classification, probability, confidence);

            println!("Training again {:#?}\n", file_path);
            dictawav.train(file_path, String::from(word));
        }
    }
}
