use std::rc::Rc;
use super::ram::Ram;

pub struct Discriminator {
    retina_size: usize,
    ram_num_bits: usize,
    rams_count: usize,
    rams: Vec<Ram>,
    ram_address_mapping: Rc<Vec<usize>>,
}

impl Discriminator {
    pub fn new(
        retina_size: usize,
        ram_num_bits: usize,
        ram_address_mapping: Rc<Vec<usize>>,
        is_cumulative: bool,
    ) -> Discriminator {
        if ram_num_bits > 62usize {
            panic!("WiSARD ERROR: Representation overflow due to number of bits");
        }
        let rams_count = (retina_size as f64 / ram_num_bits as f64).ceil() as usize;
        let mut rams = Vec::with_capacity(rams_count);

        if retina_size % ram_num_bits == 0 {
            for _ in 0..rams_count {
                rams.push(Ram::new(ram_num_bits, is_cumulative));
            }
        } else {
            for _ in 0..(rams_count - 1usize) {
                rams.push(Ram::new(ram_num_bits, is_cumulative));
            }
            // The remaining smaller ram
            rams.push(Ram::new(retina_size % ram_num_bits, is_cumulative));
        }

        Discriminator {
            retina_size,
            ram_num_bits,
            rams_count,
            rams,
            ram_address_mapping,
        }
    }

    pub fn train(&mut self, retina: &Vec<bool>) {
        let mut address: usize;
        let mut base: usize;
        let mut ram_index = 0usize;
        let mut index = 0usize;

        // Each group os ram_num_bits is related with a ram
        while index <= (self.retina_size - self.ram_num_bits) {
            address = 0usize;
            base = 1usize;

            for bit_index in 0..self.ram_num_bits {
                if retina[self.ram_address_mapping[index + bit_index]] {
                    address += base;
                }
                base *= 2usize;
            }

            ram_index = index / self.ram_num_bits;
            self.rams[ram_index].insert(address);

            index += self.ram_num_bits;
        }

        // The remaining retina when retina length isn't a multiple of ram_num_bits
        let rest_of_positions = self.retina_size % self.ram_num_bits;
        if rest_of_positions != 0usize {
            address = 0usize;
            base = 1usize;

            for bit_index in 0..rest_of_positions {
                if retina[
                    self.ram_address_mapping[
                        self.retina_size - rest_of_positions - 1usize + bit_index
                        ]
                    ] {
                    address += base;
                }
                base *= 2usize;
            }
            self.rams[ram_index + 1].insert(address);
        }
    }

    pub fn forget(&mut self, retina: &Vec<bool>) {
        let mut address: usize;
        let mut base: usize;
        let mut ram_index = 0usize;
        let mut index = 0usize;

        // Each group os ram_num_bits is related with a ram
        while index <= (self.retina_size - self.ram_num_bits) {
            address = 0usize;
            base = 1usize;

            for bit_index in 0..self.ram_num_bits {
                if retina[self.ram_address_mapping[index + bit_index]] {
                    address += base;
                }
                base *= 2usize;
            }

            ram_index = index / self.ram_num_bits;
            self.rams[ram_index].remove(address);

            index += self.ram_num_bits;
        }

        // The remaining retina when retina length isn't a multiple of ram_num_bits
        let rest_of_positions = self.retina_size % self.ram_num_bits;
        if rest_of_positions != 0usize {
            address = 0usize;
            base = 1usize;

            for bit_index in 0..rest_of_positions {
                if retina[
                    self.ram_address_mapping[
                        self.retina_size - rest_of_positions - 1usize + bit_index
                        ]
                    ] {
                    address += base;
                }
                base *= 2usize;
            }
            self.rams[ram_index + 1].remove(address);
        }
    }

    pub fn classify(&self, retina: &Vec<bool>) -> Vec<u64> {
        let mut ram_index = 0usize;
        let mut address: usize;
        let mut base: usize;
        let mut result = Vec::with_capacity(self.rams_count);

        let mut index = 0usize;
        while index <= (self.retina_size - self.ram_num_bits) {
            address = 0usize;
            base = 1usize;

            for bit_index in 0..self.ram_num_bits {
                if retina[self.ram_address_mapping[index + bit_index]] {
                    address += base;
                }
                base *= 2usize;
            }

            ram_index = index / self.ram_num_bits;
            result.push(self.rams[ram_index].get(address));

            index += self.ram_num_bits;
        }

        let rest_of_positions = self.retina_size % self.ram_num_bits;
        if rest_of_positions != 0 {
            address = 0usize;
            base = 1usize;

            for bit_index in 0..rest_of_positions {
                if retina[
                    self.ram_address_mapping[
                        self.retina_size - rest_of_positions - 1usize + bit_index
                        ]
                    ] {
                    address += base;
                }
                base *= 2usize;
            }

            result.push(self.rams[ram_index + 1usize].get(address));
        }

        result
    }
}