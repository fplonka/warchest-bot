pub struct Rng(pub u64);

impl Default for Rng {
    fn default() -> Rng {
        Rng::new(0)
    }
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let s = seed ^ 0x9E3779B97F4A7C15;
        Rng(if s == 0 { 0xDEADBEEFCAFEF00D } else { s })
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % (n as u64)) as usize
    }

    #[inline]
    pub fn unit_f64(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    pub fn gamma(&mut self, shape: f64) -> f64 {
        if shape < 1.0 {
            return self.gamma(shape + 1.0) * self.unit_f64().powf(1.0 / shape);
        }
        let d = shape - 1.0 / 3.0;
        let c = 1.0 / (9.0 * d).sqrt();
        loop {
            let x = (-2.0 * (1.0 - self.unit_f64()).ln()).sqrt() * (std::f64::consts::TAU * self.unit_f64()).cos();
            let v = (1.0 + c * x).powi(3);
            if v > 0.0 && (1.0 - self.unit_f64()).ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }

    pub fn weighted_index(&mut self, weights: &[f64]) -> usize {
        let total: f64 = weights.iter().copied().sum();
        assert!(total > 0.0 && total.is_finite());
        let mut needle = self.unit_f64() * total;
        for (i, &weight) in weights.iter().enumerate() {
            assert!(weight >= 0.0 && weight.is_finite());
            if needle < weight {
                return i;
            }
            needle -= weight;
        }
        weights.len() - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gamma_draws_have_the_shape_as_mean() {
        let mut rng = Rng::new(7);
        for shape in [0.1, 0.5, 1.0, 4.0] {
            let mean = (0..200_000).map(|_| rng.gamma(shape)).sum::<f64>() / 200_000.0;
            assert!((mean / shape - 1.0).abs() < 0.03, "shape {shape} gave mean {mean}");
        }
    }
}
