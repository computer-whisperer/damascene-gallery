//! damascene-gallery — a color-managed media gallery.
//!
//! Browse a directory of images — including the HDR formats nothing else
//! opens (JPEG XR, JPEG XL, AVIF, OpenEXR, Radiance) — with correct color
//! on Wayland: the decode pipeline (copied from prism-bg) preserves each
//! file's actual encoding, and damascene's host negotiates
//! `wp_color_management_v1` for an extended-range scRGB swapchain on HDR
//! outputs, so HDR wallpapers display with real highlights.
//!
//! Usage: `damascene-gallery [DIRECTORY | FILE...]` — without arguments
//! the app opens on a welcome screen with a system folder picker.

mod app;
mod convert;
mod loader;
mod scan;

// The image input pipeline is copied from prism-bg (2026-06-05), which
// stays unpublished. Kept close to upstream for easy diffing — wallpaper-
// path items the gallery doesn't call stay in place rather than diverge.
#[allow(dead_code)]
mod cms;
#[allow(dead_code)]
mod color;
#[allow(dead_code)]
mod decode;

use anyhow::{bail, Result};
use damascene_core::color::ColorPreferences;
use damascene_core::Rect;
use damascene_winit_wgpu::{run_with_config, HostConfig};

use app::GalleryApp;
use loader::SharedWakeup;

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "damascene_gallery=info".into()),
        )
        .init();

    let args: Vec<String> = std::env::args().skip(1).collect();
    let explicit = !args.is_empty();
    let files = scan::scan_args(args)?;
    if explicit && files.is_empty() {
        // An explicit path with nothing usable is a CLI error; with no
        // arguments the app opens on the welcome screen instead.
        bail!(
            "no supported images found (looked for: {})",
            scan::EXTENSIONS.join(", ")
        );
    }
    tracing::info!(count = files.len(), "collection scanned");

    let workers = std::thread::available_parallelism()
        .map(|n| (n.get() / 2).clamp(2, 6))
        .unwrap_or(2);

    // One wakeup handle shared by decode workers and dialog threads;
    // filled in once the event loop exists.
    let wakeup = SharedWakeup::default();
    let app = GalleryApp::new(files, workers, wakeup.clone());

    let config = HostConfig::default()
        .with_app_id("damascene-gallery")
        // Extended-range linear swapchain on HDR outputs; degrades to
        // P3/sRGB per compositor capability.
        .with_color_preferences(ColorPreferences::hdr_extended())
        .with_external_wakeup(move |w| *wakeup.lock().unwrap() = Some(w));

    let viewport = Rect::new(0.0, 0.0, 1600.0, 1000.0);
    run_with_config("Damascene Gallery", viewport, app, config)
        .map_err(|e| anyhow::anyhow!("host error: {e}"))
}
