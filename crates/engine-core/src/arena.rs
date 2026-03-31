//! Zero-Copy Tensor Arena
//! 
//! Pre-allocated memory pool that eliminates runtime allocations during inference.
//! Memory is divided into three sections:
//! 1. Weight Buffer: Immutable model parameters (shared across requests)
//! 2. Activation Arena: Mutable workspace for layer outputs (reused per layer)
//! 3. IO Buffer: Input/Output gateway for Sink/Stream interface

use bytemuck::{Pod, Zeroable};
use std::alloc::{alloc, dealloc, Layout};
use std::marker::PhantomData;
use std::ptr::NonNull;

/// Memory section types for type-safe arena access
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Section {
    Weights,
    Activations,
    IO,
}

/// A tensor view into the arena - zero-copy reference to a slice
#[derive(Debug)]
pub struct TensorView<'a, T: Pod> {
    data: &'a [T],
    shape: Vec<usize>,
}

/// A mutable tensor view into the arena
#[derive(Debug)]
pub struct TensorViewMut<'a, T: Pod> {
    data: &'a mut [T],
    shape: Vec<usize>,
}

impl<'a, T: Pod> TensorView<'a, T> {
    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl<'a, T: Pod> TensorViewMut<'a, T> {
    #[inline]
    pub fn data(&self) -> &[T] {
        self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T] {
        self.data
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }
}

/// Memory layout descriptor for arena sections
#[derive(Debug, Clone)]
struct SectionLayout {
    offset: usize,
    size: usize,
    section: Section,
}

/// The Tensor Arena: A single contiguous memory block for all inference operations
pub struct Arena<T: Pod + Zeroable> {
    ptr: NonNull<T>,
    capacity: usize,
    layout: Layout,
    sections: Vec<SectionLayout>,
    _marker: PhantomData<T>,
}

impl<T: Pod + Zeroable> Arena<T> {
    /// Create a new arena with specified section sizes
    /// 
    /// # Safety
    /// This allocates raw memory. The arena must be properly initialized before use.
    pub fn new(weights_size: usize, activations_size: usize, io_size: usize) -> Self {
        let total_capacity = weights_size + activations_size + io_size;
        let layout = Layout::array::<T>(total_capacity).expect("Invalid layout");

        let ptr = unsafe {
            let raw_ptr = alloc(layout) as *mut T;
            NonNull::new(raw_ptr).expect("Allocation failed")
        };

        let mut sections = Vec::with_capacity(3);
        let mut offset = 0;

        // Weight section
        sections.push(SectionLayout {
            offset,
            size: weights_size,
            section: Section::Weights,
        });
        offset += weights_size;

        // Activation section
        sections.push(SectionLayout {
            offset,
            size: activations_size,
            section: Section::Activations,
        });
        offset += activations_size;

        // IO section
        sections.push(SectionLayout {
            offset,
            size: io_size,
            section: Section::IO,
        });

        Self {
            ptr,
            capacity: total_capacity,
            layout,
            sections,
            _marker: PhantomData,
        }
    }

    /// Get a tensor view at the specified offset within a section
    pub fn view(&self, section: Section, offset: usize, len: usize, shape: Vec<usize>) -> TensorView<T> {
        let section_layout = self.get_section(section);
        let absolute_offset = section_layout.offset + offset;
        
        assert!(offset + len <= section_layout.size, "View exceeds section bounds");
        
        let data = unsafe {
            let ptr = self.ptr.as_ptr().add(absolute_offset);
            std::slice::from_raw_parts(ptr, len)
        };

        TensorView { data, shape }
    }

    /// Get a mutable tensor view at the specified offset within a section
    pub fn view_mut(&mut self, section: Section, offset: usize, len: usize, shape: Vec<usize>) -> TensorViewMut<T> {
        let section_layout = self.get_section(section);
        let absolute_offset = section_layout.offset + offset;
        
        assert!(offset + len <= section_layout.size, "View exceeds section bounds");
        
        let data = unsafe {
            let ptr = self.ptr.as_ptr().add(absolute_offset);
            std::slice::from_raw_parts_mut(ptr, len)
        };

        TensorViewMut { data, shape }
    }

    /// Get the raw slice for a section (for bulk operations)
    pub fn section_slice(&self, section: Section) -> &[T] {
        let layout = self.get_section(section);
        unsafe {
            let ptr = self.ptr.as_ptr().add(layout.offset);
            std::slice::from_raw_parts(ptr, layout.size)
        }
    }

    /// Get the raw mutable slice for a section
    pub fn section_slice_mut(&mut self, section: Section) -> &mut [T] {
        let layout = self.get_section(section);
        unsafe {
            let ptr = self.ptr.as_ptr().add(layout.offset);
            std::slice::from_raw_parts_mut(ptr, layout.size)
        }
    }

    /// Zero out the activation arena (prepare for next inference)
    #[inline]
    pub fn clear_activations(&mut self) {
        let slice = self.section_slice_mut(Section::Activations);
        slice.fill(T::zeroed());
    }

    /// Load weights from a byte slice (typically from mmap)
    pub fn load_weights(&mut self, data: &[u8]) -> Result<(), ArenaError> {
        let weights = self.section_slice_mut(Section::Weights);
        let byte_slice = bytemuck::cast_slice_mut::<T, u8>(weights);
        
        if data.len() != byte_slice.len() {
            return Err(ArenaError::SizeMismatch {
                expected: byte_slice.len(),
                actual: data.len(),
            });
        }

        byte_slice.copy_from_slice(data);
        Ok(())
    }

    #[inline]
    fn get_section(&self, section: Section) -> &SectionLayout {
        self.sections
            .iter()
            .find(|s| s.section == section)
            .expect("Section not found")
    }
}

impl<T: Pod + Zeroable> Drop for Arena<T> {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// Safety: Arena can be sent across threads if T is Send
unsafe impl<T: Pod + Zeroable + Send> Send for Arena<T> {}
// Arena is not Sync because interior mutability requires external synchronization
// Use Arc<Mutex<Arena>> if needed across threads

#[derive(Debug, thiserror::Error)]
pub enum ArenaError {
    #[error("Size mismatch: expected {expected} bytes, got {actual}")]
    SizeMismatch { expected: usize, actual: usize },
    
    #[error("Section not found: {0:?}")]
    SectionNotFound(Section),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_arena_creation() {
        let arena: Arena<f32> = Arena::new(1024, 2048, 512);
        assert_eq!(arena.capacity, 1024 + 2048 + 512);
    }

    #[test]
    fn test_tensor_views() {
        let mut arena: Arena<f32> = Arena::new(100, 200, 50);
        
        // Write to activation section
        {
            let mut view = arena.view_mut(Section::Activations, 0, 10, vec![2, 5]);
            let data = view.data_mut();
            data[0] = 1.0;
            data[9] = 9.0;
        }

        // Read from activation section
        {
            let view = arena.view(Section::Activations, 0, 10, vec![2, 5]);
            let data = view.data();
            assert_eq!(data[0], 1.0);
            assert_eq!(data[9], 9.0);
        }
    }

    #[test]
    fn test_clear_activations() {
        let mut arena: Arena<f32> = Arena::new(100, 200, 50);
        
        // Write some data
        {
            let mut view = arena.view_mut(Section::Activations, 0, 10, vec![10]);
            view.data_mut().fill(5.0);
        }

        // Clear
        arena.clear_activations();

        // Verify cleared
        {
            let view = arena.view(Section::Activations, 0, 10, vec![10]);
            assert!(view.data().iter().all(|&x| x == 0.0));
        }
    }
}
