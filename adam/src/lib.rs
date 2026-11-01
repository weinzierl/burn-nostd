//! Adam for Burn modules, without std.
//!
//! `burn-optim` 0.22 needs std, so this is the optimizer step written against
//! burn-core alone: a `ModuleMapper` that walks the parameters, takes each
//! gradient and applies Adam. No schedulers, no clipping, no weight decay.

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use burn::module::{Module, ModuleMapper, Param};
use burn::tensor::{Gradients, Tensor, TensorData};

pub struct Adam {
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    step: i32,
    /// First and second moments per parameter, in visiting order.
    moments: Vec<(Vec<f32>, Vec<f32>)>,
}

impl Default for Adam {
    fn default() -> Self {
        Self { beta1: 0.9, beta2: 0.999, epsilon: 1e-8, step: 0, moments: Vec::new() }
    }
}

impl Adam {
    pub fn step<M: Module>(&mut self, lr: f32, module: M, grads: Gradients) -> M {
        self.step += 1;
        module.map(&mut Step { adam: self, lr, grads, param: 0 })
    }
}

struct Step<'a> {
    adam: &'a mut Adam,
    lr: f32,
    grads: Gradients,
    param: usize,
}

impl ModuleMapper for Step<'_> {
    fn map_float<const D: usize>(&mut self, param: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let (id, tensor, mapper) = param.consume();
        let index = self.param;
        self.param += 1;
        let Some(grad) = tensor.grad_remove(&mut self.grads) else {
            return Param::from_mapped_value(id, tensor, mapper);
        };

        // Parameters are small; plain slices keep the update independent of
        // which tensor ops the backend implements.
        let shape = tensor.shape();
        let device = grad.device();
        let g = grad.into_data().try_into_vec::<f32>().expect("f32 gradient");
        let mut w = tensor.inner().into_data().try_into_vec::<f32>().expect("f32 parameter");

        let a = &mut *self.adam;
        if a.moments.len() <= index {
            a.moments.resize_with(index + 1, Default::default);
        }
        let (m, v) = &mut a.moments[index];
        m.resize(g.len(), 0.0);
        v.resize(g.len(), 0.0);
        let bias1 = 1.0 - libm::powf(a.beta1, a.step as f32);
        let bias2 = 1.0 - libm::powf(a.beta2, a.step as f32);
        for k in 0..g.len() {
            m[k] = a.beta1 * m[k] + (1.0 - a.beta1) * g[k];
            v[k] = a.beta2 * v[k] + (1.0 - a.beta2) * g[k] * g[k];
            w[k] -= self.lr * (m[k] / bias1) / (libm::sqrtf(v[k] / bias2) + a.epsilon);
        }

        let updated = Tensor::<D>::from_data(TensorData::new(w, shape), &device);
        Param::from_mapped_value(id, Tensor::from_inner(updated).require_grad(), mapper)
    }
}
