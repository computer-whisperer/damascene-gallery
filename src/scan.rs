//! Turning CLI arguments or a picked directory into a sorted file list.

use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};

/// Extensions worth queuing for decode. Dispatch inside the pipeline is
/// by magic bytes; this only filters the directory scan.
pub const EXTENSIONS: &[&str] = &[
    "jxr", "jxl", "avif", "png", "jpg", "jpeg", "webp", "exr", "hdr",
];

/// Supported image files directly inside `dir` (not recursive), sorted.
pub fn scan_dir(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for entry in
        std::fs::read_dir(dir).with_context(|| format!("reading directory {}", dir.display()))?
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
    files.sort();
    Ok(files)
}

/// Expand CLI arguments (directories and/or files) into a deduplicated,
/// sorted file list. No arguments means no collection — the app opens
/// on the welcome screen.
pub fn scan_args(args: Vec<String>) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    for input in args {
        let path = PathBuf::from(&input);
        if path.is_dir() {
            files.extend(scan_dir(&path)?);
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
