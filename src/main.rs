//! damascene-gallery — a color-managed media gallery.
//!
//! Browse a directory of images — including the HDR formats nothing else
//! opens (JPEG XR, JPEG XL, AVIF, OpenEXR, Radiance) — with correct color
//! on Wayland: the decode pipeline (copied from prism-bg) preserves each
//! file's actual encoding, and damascene's host negotiates
//! `wp_color_management_v1` for an extended-range scRGB swapchain on HDR
//! outputs, so HDR wallpapers display with real highlights.
//!
//! Usage: `damascene-gallery [DIRECTORY | FILE...]` (default `.`).

mod app;
mod convert;
mod loader;

// The image input pipeline is copied from prism-bg (2026-06-05), which
// stays unpublished. Kept close to upstream for easy diffing — wallpaper-
// path items the gallery doesn't call stay in place rather than diverge.
#[allow(dead_code)]
mod cms;
#[allow(dead_code)]
mod color;
#[allow(dead_code)]
mod decode;

use std::path::PathBuf;

use anyhow::{bail, Context, Result};
use damascene_core::color::ColorPreferences;
use damascene_core::Rect;
use damascene_winit_wgpu::{run_with_config, HostConfig};

use app::GalleryApp;
use loader::Loader;

/// Extensions worth queuing for decode. Dispatch inside the pipeline is
/// by magic bytes; this only filters the directory scan.
const EXTENSIONS: &[&str] = &[
    "jxr", "jxl", "avif", "png", "jpg", "jpeg", "webp", "exr", "hdr",
];

fn scan(args: Vec<String>) -> Result<Vec<PathBuf>> {
    let inputs = if args.is_empty() {
        vec![".".to_string()]
    } else {
        args
    };

    let mut files = Vec::new();
    for input in inputs {
        let path = PathBuf::from(&input);
        if path.is_dir() {
            for entry in
                std::fs::read_dir(&path).with_context(|| format!("reading directory {input}"))?
            {
                let p = entry?.path();
                let ext = p
                    .extension()
                    .map(|e| e.to_string_lossy().to_lowercase())
                    .unwrap_or_default();
                if p.is_file() && EXTENSIONS.contains(&ext.as_str()) {
                    files.push(p);
                }
            }
        } else if path.is_file() {
            files.push(path);
        } else {
            bail!("{input}: not a file or directory");
        }
    }
    files.sort();
    files.dedup();
    Ok(files)
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "damascene_gallery=info".into()),
        )
        .init();

    let files = scan(std::env::args().skip(1).collect())?;
    if files.is_empty() {
        bail!(
            "no supported images found (looked for: {})",
            EXTENSIONS.join(", ")
        );
    }
    tracing::info!(count = files.len(), "collection scanned");

    let workers = std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(2, 6))
        .unwrap_or(2);
    let (loader, results) = Loader::spawn(files.clone(), workers);

    let app = GalleryApp::new(files, loader.clone(), results);

    let config = HostConfig::default()
        .with_app_id("damascene-gallery")
        // Extended-range linear swapchain on HDR outputs; degrades to
        // P3/sRGB per compositor capability.
        .with_color_preferences(ColorPreferences::hdr_extended())
        .with_external_wakeup(move |wakeup| loader.set_wakeup(wakeup));

    let viewport = Rect::new(0.0, 0.0, 1600.0, 1000.0);
    run_with_config("Damascene Gallery", viewport, app, config)
        .map_err(|e| anyhow::anyhow!("host error: {e}"))
}
