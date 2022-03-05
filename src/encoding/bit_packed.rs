const BLOCK_LEN: usize = 32;

#[inline]
fn saturating_shl(next_pack: u32, shift: u8) -> u32 {
    if shift == 32 {
        0
    } else {
        next_pack << shift
    }
}

fn decompress(compressed: &[u8], decompressed: &mut [u32; BLOCK_LEN], num_bits: u8) {
    let mask = !0u32 << (32 - num_bits);

    let mut packs = compressed
        .windows(4)
        .map(|bytes| {
            let mut compressed = [0u8; std::mem::size_of::<u32>()];
            compressed.copy_from_slice(bytes);
            u32::from_be_bytes(compressed)
        })
        .peekable();

    let mut current_pack = packs.next().unwrap_or(0);
    let mut next_pack = packs.next().unwrap_or(0);
    let mut next_next_pack = 0;
    // number of bits remaining on the `current_pack`
    let mut remaining_bits = 32;

    // whether we need to load `next_next_pack` or we are still using the remaining bits from it
    let mut load_next = true;
    decompressed.iter_mut().for_each(|x| {
        if remaining_bits < num_bits {
            current_pack |= next_pack >> remaining_bits;
            next_pack = saturating_shl(next_pack, 32 - remaining_bits);
            if load_next {
                next_next_pack = packs.next().unwrap_or(0);
                next_pack |= next_next_pack >> remaining_bits;
                next_next_pack = saturating_shl(next_next_pack, 32 - remaining_bits);
            } else {
                next_pack |= next_next_pack >> remaining_bits;
            }

            remaining_bits = 32;
            load_next = !load_next;
        }
        let a = current_pack & mask;
        current_pack <<= num_bits;
        *x = a >> (32 - num_bits);
        remaining_bits -= num_bits;
    });
}

fn decode_pack(compressed: &[u8], num_bits: u8, pack: &mut [u32; BLOCK_LEN]) {
    let compressed_block_size = BLOCK_LEN * num_bits as usize / 8;

    if compressed.len() < compressed_block_size {
        let mut buf = [0u8; BLOCK_LEN * std::mem::size_of::<u32>()];
        buf[..compressed.len()].copy_from_slice(compressed);
        decompress(&buf, pack, num_bits);
    } else {
        decompress(compressed, pack, num_bits);
    }
}

/// Decoder of parquet's BIT_PACKED (https://github.com/apache/parquet-format/blob/master/Encodings.md#bit-packed-deprecated-bit_packed--4)
#[derive(Debug, Clone)]
pub struct Decoder<'a> {
    compressed_chunks: std::slice::Chunks<'a, u8>,
    num_bits: u8,
    remaining: usize,
    current_pack_index: usize, // invariant: <BLOCK_LEN
    current_pack: [u32; BLOCK_LEN],
}

impl<'a> Decoder<'a> {
    pub fn new(compressed: &'a [u8], num_bits: u8, length: usize) -> Self {
        let compressed_block_size = BLOCK_LEN * num_bits as usize / 8;

        let mut compressed_chunks = compressed.chunks(compressed_block_size);
        let mut current_pack = [0; BLOCK_LEN];
        decode_pack(
            compressed_chunks.next().unwrap(),
            num_bits,
            &mut current_pack,
        );

        Self {
            remaining: length,
            compressed_chunks,
            num_bits,
            current_pack,
            current_pack_index: 0,
        }
    }
}

impl<'a> Iterator for Decoder<'a> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let result = self.current_pack[self.current_pack_index];
        self.current_pack_index += 1;
        if self.current_pack_index == BLOCK_LEN {
            if let Some(chunk) = self.compressed_chunks.next() {
                decode_pack(chunk, self.num_bits, &mut self.current_pack);
                self.current_pack_index = 0;
            }
        }
        self.remaining -= 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.remaining, Some(self.remaining))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decode_small() {
        // Test data: 0-7 with bit width 3
        // 0: 000
        // 1: 001
        // 2: 010
        // 3: 011
        // 4: 100
        // 5: 101
        // 6: 110
        // 7: 111
        let num_bits = 3;
        let length = 8;
        let data = vec![0b00000101u8, 0b00111001, 0b01110111];

        let decoded = Decoder::new(&data, num_bits, length).collect::<Vec<_>>();
        assert_eq!(decoded, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn decode_large() {
        // Test data: 0-7 with bit width 3
        // 0: 000
        // 1: 001
        // 2: 010
        // 3: 011
        // 4: 100
        // 5: 101
        // 6: 110
        // 7: 111
        let num_bits = 3;
        let length = 8 * 7;
        let data = vec![
            0b00000101u8,
            0b00111001,
            0b01110111,
            0b00000101u8,
            0b00111001,
            0b01110111,
            0b00000101u8,
            0b00111001,
            0b01110111,
            0b00000101u8,
            0b00111001,
            0b01110111,
            0b00000101u8,
            0b00111001,
            0b01110111,
            0b00000101u8,
            0b00111001,
            0b01110111,
            0b00000101u8,
            0b00111001,
            0b01110111,
        ];

        let decoded = Decoder::new(&data, num_bits, length).collect::<Vec<_>>();
        assert_eq!(
            decoded,
            vec![
                0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3,
                4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7,
            ]
        );
    }

    #[test]
    fn decode_binary() {
        // 0: 0
        // 1: 1
        // 1: 1
        // 0: 0
        let num_bits = 1;
        let length = 4;
        let data = vec![0b01100000u8];

        let decoded = Decoder::new(&data, num_bits, length).collect::<Vec<_>>();
        assert_eq!(decoded, vec![0, 1, 1, 0]);
    }

    #[test]
    fn decode_larger() {
        // 255: 11111111
        // 0: 00000000
        // 1: 00000001
        let num_bits = 8;
        let length = 3;
        let data = vec![0b11111111u8, 0b00000000u8, 0b00000001u8];

        let decoded = Decoder::new(&data, num_bits, length).collect::<Vec<_>>();
        assert_eq!(decoded, vec![255, 0, 1]);
    }
}
