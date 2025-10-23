use anyhow::{anyhow, Result};
use csv::ReaderBuilder;
use ndarray::{Array2, Axis, s};
use rand::prelude::*;
use std::fs::File;
use std::path::Path;

/// Represents a numerical dataset loaded from a TSV file
#[derive(Debug, Clone)]
pub struct DataSet {
    pub data: Array2<f32>,
    pub headers: Option<Vec<String>>,
}

impl DataSet {
    /// Read a TSV file into a DataSet
    pub fn from_tsv<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(&path)
            .map_err(|e| anyhow!("Failed to open {:?}: {}", path.as_ref(), e))?;

        let mut rdr = ReaderBuilder::new()
            .delimiter(b'\t')
            .has_headers(true)
            .from_reader(file);

        let headers = rdr.headers()
            .ok()
            .map(|h| h.iter().map(|s| s.to_string()).collect::<Vec<_>>());

        let mut records: Vec<Vec<f32>> = Vec::new();

        for (i, result) in rdr.records().enumerate() {
            let record = result.map_err(|e| anyhow!("Error reading record {}: {}", i, e))?;
            let row: Vec<f32> = record
                .iter()
                .map(|x| x.parse::<f32>().unwrap_or(f32::NAN))
                .collect();
            records.push(row);
        }

        if records.is_empty() {
            return Err(anyhow!("No data lines found in {:?}", path.as_ref()));
        }

        let nrows = records.len();
        let ncols = records[0].len();
        let flat: Vec<f32> = records.into_iter().flatten().collect();
        let data = Array2::from_shape_vec((nrows, ncols), flat)?;

        Ok(Self { data, headers })
    }

    /// Get a view of the first N columns (for clustering)
    pub fn numeric_view(&self, ncols: usize) -> Array2<f32> {
        let cols = usize::min(ncols, self.data.ncols());
        self.data.slice(s![.., 0..cols]).to_owned()
    }

    /// Euclidean distance between two 3D points
    #[inline]
    fn e_dist3(a: &[f32; 3], b: &[f32; 3]) -> f32 {
        ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt()
    }

    /// Perform K-means clustering on the first three columns
    pub fn kmeans3d(&self, k: usize, max_iter: usize) -> Result<Vec<usize>> {
        let data = self.numeric_view(3);
        let nrows = data.nrows();

        if nrows < k {
            return Err(anyhow!("Not enough data points ({}) for {} clusters", nrows, k));
        }

        let mut rng = thread_rng();

        // Randomly select k initial centroids
        let mut indices: Vec<usize> = (0..nrows).collect();
        indices.shuffle(&mut rng);

        let mut centroids = Array2::<f32>::zeros((k, 3));
        for (ci, &idx) in indices.iter().take(k).enumerate() {
            centroids.row_mut(ci).assign(&data.row(idx));
        }

        // Each point’s cluster assignment
        let mut assignments = vec![0usize; nrows];

        for _ in 0..max_iter {
            // Step 1: assign points to nearest centroid
            for (i, row) in data.outer_iter().enumerate() {
                let point = [row[0], row[1], row[2]];
                let mut best_cluster = 0;
                let mut best_dist = f32::MAX;

                for (ci, c_row) in centroids.outer_iter().enumerate() {
                    let centroid = [c_row[0], c_row[1], c_row[2]];
                    let dist = Self::e_dist3(&point, &centroid);
                    if dist < best_dist {
                        best_dist = dist;
                        best_cluster = ci;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Step 2: update centroids as mean of assigned points
            let mut new_centroids = Array2::<f32>::zeros((k, 3));
            let mut counts = vec![0usize; k];

            for (i, row) in data.outer_iter().enumerate() {
                let c = assignments[i];
                new_centroids.row_mut(c).assign(&row);
                counts[c] += 1;
            }

            for ci in 0..k {
                if counts[ci] > 0 {
                    new_centroids.row_mut(ci).mapv_inplace(|x| x / counts[ci] as f32);
                } else {
                    // reinitialize empty cluster randomly
                    let idx = rng.gen_range(0..nrows);
                    new_centroids.row_mut(ci).assign(&data.row(idx));
                }
            }

            // convergence check (optional)
            let diff = (&centroids - &new_centroids)
                .mapv(|x| x.abs())
                .sum();
            if diff < 1e-4 {
                break;
            }
            centroids = new_centroids;
        }

        Ok(assignments)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e_dist3() {
        let a = [0.0, 0.0, 0.0];
        let b = [0.0, 3.0, 4.0];
        assert_eq!(DataSet::e_dist3(&a, &b), 5.0);
    }

    #[test]
    fn test_kmeans_basic() {
        // Make some fake clusters
        let data = array![
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [9.0, 9.0, 9.0],
            [9.1, 9.1, 9.1],
        ];
        let ds = DataSet {
            data,
            headers: None,
        };
        let labels = ds.kmeans3d(2, 20).unwrap();
        assert_eq!(labels.len(), 4);
    }
}