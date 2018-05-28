use std::collections::hash_map::HashMap;

pub struct Ram {
    data: HashMap<usize, u64>,
    max_address: usize,
    is_cumulative: bool,
}

impl Ram {
    pub fn new(num_bits: usize, is_cumulative: bool) -> Ram {
        let max_address = 2usize.pow(num_bits as u32);
        let data = HashMap::new();

        Ram {
            data,
            max_address,
            is_cumulative,
        }
    }

    pub fn insert(&mut self, address: usize) {
        if address > self.max_address {
            panic!("WiSARD Error: Invalid address to add value");
        }

        if !self.is_cumulative {
            self.data.insert(address, 1);
        } else {
            *self.data.entry(address).or_insert(0) += 1;
        }
    }

    pub fn remove(&mut self, address: usize) {
        if address > self.max_address {
            panic!("WiSARD Error: Invalid address to add value");
        }

        if !self.is_cumulative {
            self.data.insert(address, 0);
        } else {
            *self.data.entry(address).or_insert(1) -= 1;
        }
    }

    pub fn get(&self, address: usize) -> u64 {
        match self.data.get(&address) {
            Some(value) => *value,
            None => 0u64,
        }
    }
}
