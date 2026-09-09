//! The composition pass: color math (`color.rs`, `downscale.rs`) plus, eventually,
//! the GPU pipeline that runs it (compute shader, descriptor sets, image resources).
//! See the plan's milestone 4 and `color.rs`'s module doc comment for what's
//! rederived from where.
//!
//! Every function/constant in `color.rs`/`downscale.rs` is genuinely unused outside
//! `#[cfg(test)]` right now, not forgotten: nothing in `device.rs`'s present hook
//! calls into this module yet (see the crate's `CLAUDE.md` entry for exactly what's
//! wired up vs. still open) -- the math is real and tested; the GPU dispatch that
//! would call it is milestone 4's remaining work.
#![allow(dead_code)]

pub mod color;
pub mod downscale;
