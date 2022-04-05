use std::collections::VecDeque;

use crate::{
    encoding::hybrid_rle::{self, HybridRleDecoder},
    indexes::Interval,
    page::{split_buffer, DataPage},
    read::levels::get_bit_width,
};

use super::hybrid_rle::{HybridDecoderBitmapIter, HybridRleIter};

pub(super) fn dict_indices_decoder(page: &DataPage) -> hybrid_rle::HybridRleDecoder {
    let (_, _, indices_buffer) = split_buffer(page);

    // SPEC: Data page format: the bit width used to encode the entry ids stored as 1 byte (max bit width = 32),
    // SPEC: followed by the values encoded using RLE/Bit packed described above (with the given bit width).
    let bit_width = indices_buffer[0];
    let indices_buffer = &indices_buffer[1..];

    hybrid_rle::HybridRleDecoder::new(indices_buffer, bit_width as u32, page.num_values())
}

/// Decoder of definition levels.
#[derive(Debug)]
pub enum DefLevelsDecoder<'a> {
    /// When the maximum definition level is 1, the definition levels are RLE-encoded and
    /// the bitpacked runs are bitmaps. This variant contains [`HybridDecoderBitmapIter`]
    /// that decodes the runs, but not the individual values
    Bitmap(HybridDecoderBitmapIter<'a>),
    /// When the maximum definition level is larger than 1
    Levels(HybridRleDecoder<'a>, u32),
}

impl<'a> DefLevelsDecoder<'a> {
    pub fn new(page: &'a DataPage) -> Self {
        let (_, def_levels, _) = split_buffer(page);

        let max_def_level = page.descriptor.max_def_level;
        if max_def_level == 1 {
            let iter = hybrid_rle::Decoder::new(def_levels, 1);
            let iter = HybridRleIter::new(iter, page.num_values());
            Self::Bitmap(iter)
        } else {
            let iter =
                HybridRleDecoder::new(def_levels, get_bit_width(max_def_level), page.num_values());
            Self::Levels(iter, max_def_level as u32)
        }
    }
}

/// Iterator adapter to convert an iterator of non-null values and an iterator over validity
/// into an iterator of optional values.
#[derive(Debug, Clone)]
pub struct OptionalValues<T, V: Iterator<Item = bool>, I: Iterator<Item = T>> {
    validity: V,
    values: I,
}

impl<T, V: Iterator<Item = bool>, I: Iterator<Item = T>> OptionalValues<T, V, I> {
    pub fn new(validity: V, values: I) -> Self {
        Self { validity, values }
    }
}

impl<T, V: Iterator<Item = bool>, I: Iterator<Item = T>> Iterator for OptionalValues<T, V, I> {
    type Item = Option<T>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.validity
            .next()
            .map(|x| if x { self.values.next() } else { None })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.validity.size_hint()
    }
}

/// An iterator adapter that converts an iterator over items into an iterator over slices of
/// those N items.
///
/// This iterator is best used with iterators that implement `nth` since skipping items
/// allows this iterator to skip sequences of items without having to call each of them.
#[derive(Debug, Clone)]
pub struct SliceFilteredIter<I> {
    iter: I,
    selected_rows: VecDeque<Interval>,
    total: usize, // a cache
    current_remaining: usize,
    current: usize, // position in the slice
}

impl<I> SliceFilteredIter<I> {
    /// Return a new [`SliceFilteredIter`]
    pub fn new(iter: I, selected_rows: VecDeque<Interval>) -> Self {
        let total = selected_rows.iter().map(|x| x.length).sum();
        Self {
            iter,
            selected_rows,
            total,
            current_remaining: 0,
            current: 0,
        }
    }
}

impl<T, I: Iterator<Item = T>> Iterator for SliceFilteredIter<I> {
    type Item = T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.current_remaining == 0 {
            if let Some(interval) = self.selected_rows.pop_front() {
                // skip the hole between the previous start end this start
                // (start + length) - start
                let item = self.iter.nth(interval.start - self.current);
                self.current = interval.start + interval.length;
                self.current_remaining = interval.length - 1;
                self.total -= 1;
                item
            } else {
                None
            }
        } else {
            self.current_remaining -= 1;
            self.total -= 1;
            self.iter.next()
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let (min, max) = self.iter.size_hint();
        (min.min(self.total), max.map(|x| x.min(self.total)))
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for SliceFilteredIter<I> {}

#[cfg(test)]
mod test {
    use std::collections::VecDeque;

    use super::*;

    #[test]
    fn basic() {
        let iter = 0..=100;

        let intervals = vec![
            Interval::new(0, 2),
            Interval::new(20, 11),
            Interval::new(31, 1),
        ];

        let a: VecDeque<Interval> = intervals.clone().into_iter().collect();
        let a = SliceFilteredIter::new(iter, a);

        let expected: Vec<usize> = intervals
            .into_iter()
            .flat_map(|interval| interval.start..(interval.start + interval.length))
            .collect();

        assert_eq!(expected, a.collect::<Vec<_>>());
    }

    #[test]
    fn size_hint() {
        let iter = 0..=100;

        let intervals = vec![
            Interval::new(0, 2),
            Interval::new(20, 11),
            Interval::new(31, 1),
        ];

        let a = intervals.into_iter().collect();
        let mut iter = SliceFilteredIter::new(iter, a);

        iter.next();
        iter.next();
        iter.next();

        let expected = 2 + 11 + 1 - 3;
        assert_eq!(iter.size_hint(), (expected, Some(expected)))
    }
}
