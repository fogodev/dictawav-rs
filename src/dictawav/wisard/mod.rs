extern crate rand;

mod ram;
mod discriminator;

use std::rc::Rc;
use std::collections::hash_map::HashMap;
use self::rand::Rng;
use self::discriminator::Discriminator;

pub struct Wisard {
    retina_size: usize,
    ram_num_bits: usize,
    use_bleaching: bool,
    minimum_confidence: f64,
    bleaching_threshold: u64,
    is_cumulative: bool,
    discriminators: HashMap<String, Discriminator>,
    ram_address_mapping: Rc<Vec<usize>>,
}

impl Wisard {
    pub fn new(
        retina_size: usize,
        ram_num_bits: usize,
        use_bleaching: bool,
        minimum_confidence: f64,
        bleaching_threshold: u64,
        randomize_positions: bool,
        is_cumulative: bool,
    ) -> Wisard {
        let discriminators = HashMap::new();
        let mut ram_address_mapping =  (0..retina_size).collect::<Vec<usize>>();

        if randomize_positions {
            rand::thread_rng().shuffle(&mut ram_address_mapping);
        }

        let ram_address_mapping = Rc::new(ram_address_mapping);

        Wisard {
            retina_size,
            ram_num_bits,
            use_bleaching,
            minimum_confidence,
            bleaching_threshold,
            is_cumulative,
            discriminators,
            ram_address_mapping,
        }
    }

    pub fn train(&mut self, class_name: String, retina: &[bool]) {
        // Checking if class name exist before creating a new one
        self.discriminators.entry(class_name).or_insert(Discriminator::new(
            self.retina_size,
            self.ram_num_bits,
            self.ram_address_mapping.clone(),
            self.is_cumulative,
        )).train(retina);
    }

    pub fn forget(&mut self, class_name: &str, retina: &[bool]) {
        if let Some(discriminator) = self.discriminators.get_mut(class_name) {
            discriminator.forget(retina);
        }
    }

    pub fn classification_probabilities(&self, retina: &[bool]) -> HashMap<String, f64> {
        let mut results = HashMap::with_capacity(self.discriminators.len());
        let mut rams_results = HashMap::with_capacity(self.discriminators.len());

        let rams_count = (self.retina_size as f64 / self.ram_num_bits as f64).ceil();

        for (class_name, discriminator) in &self.discriminators {
            let ram_result = discriminator.classify(retina);

            // Counting how many rams have positive results
            let positive_votes = ram_result.iter().filter(|&result| *result > 0u64).count();

            // Calculating probability to see what percentage of rams recognize the element
            results.insert(class_name.clone(), positive_votes as f64 / rams_count);
            rams_results.insert(class_name.clone(), ram_result);
        }

        if self.use_bleaching {
            results = self.apply_bleaching(results, &rams_results, rams_count);
        }

        results
    }

    pub fn classify(&self, retina: &[bool]) -> String {
        let (_, (class_name, _)) = self.classification_confidence_and_probability(retina);
        class_name
    }

    pub fn classification_and_probability(&self, retina: &[bool]) -> (String, f64) {
        let (_, best_class) = self.classification_confidence_and_probability(retina);
        best_class
    }

    pub fn classification_confidence_and_probability(&self, retina: &[bool]) -> (f64, (String, f64)) {
        let (confidence, best_class) = self.calculate_confidence(&self.classification_probabilities(retina));
        if confidence < self.minimum_confidence {
            return (0f64, (String::from("Not enough confidence to decide"), 0f64))
        }
        (confidence, best_class)
    }

    fn apply_bleaching(
        &self,
        results: HashMap<String, f64>,
        rams_results: &HashMap<String, Vec<u64>>,
        rams_count: f64
    ) -> HashMap<String, f64> {
        let mut bleached_results = results.clone();
        let (mut confidence, _) = self.calculate_confidence(&results);
        let mut current_bleaching_threshold = self.bleaching_threshold;

        while confidence < self.minimum_confidence {
            let mut max_value = 0f64;

            for (class_name, result) in &mut bleached_results {
                let summed_ram_values = rams_results[class_name]
                    .iter()
                    .filter(|&value| *value > current_bleaching_threshold )
                    .count();

                *result = summed_ram_values as f64 / rams_count;

                if *result - max_value > 0.0001 {
                    max_value = *result;
                }
            }

            // If no ram recognizes the pattern, return previous results
            if max_value <= 0.000_001 {
                return results;
            }

            current_bleaching_threshold += 1u64;
            confidence = self.calculate_confidence(&bleached_results).0;
        }

        bleached_results
    }

    fn calculate_confidence(
        &self,
        classifications_probabilities: &HashMap<String, f64>,
    ) -> (f64, (String, f64)) {
        let mut best_class_name = &String::new();
        let mut max = 0f64;
        let mut second_max = 0f64;

        for (class_name, probability) in classifications_probabilities.iter() {
            if max < *probability {
                second_max = max;
                max = *probability;
                best_class_name = class_name;
            } else if second_max < *probability {
                second_max = *probability;
            }
        }

        let confidence = if max != 0f64 { 1f64 - (second_max) / max } else { 0f64 };

        (confidence, (best_class_name.clone(), max))
    }
}