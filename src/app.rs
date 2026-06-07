//! The gallery [`App`]: a welcome screen with a system folder picker, a
//! virtualized thumbnail grid over the open collection, and a full-size
//! viewer with keyboard navigation.
//!
//! All decode work happens on the loader threads; this module only moves
//! `Image` handles (cheap `Arc` clones) into the El tree. Thumbnails fill
//! in as the background sweep completes — realized grid rows jump the
//! queue, so whatever is on screen loads first.

use std::cell::{Cell, RefCell};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver};

use damascene_core::prelude::*;
use damascene_core::scroll::{ScrollAlignment, ScrollRequest};
use damascene_core::{BuildCx, EventCx, KeyChord, UiEvent, UiEventKind, UiKey};
use lru::LruCache;

use crate::convert::ImageMeta;
use crate::loader::{self, JobKind, Loaded, Loader, LoaderResults, SharedWakeup};
use crate::scan;

/// Grid tile geometry (logical px). Wallpaper-shaped (16:10) tiles; the
/// thumbnail covers the tile, cropping a little on mismatched aspects.
const TILE_W: f32 = 256.0;
const TILE_H: f32 = 160.0;
const TILE_GAP: f32 = tokens::SPACE_2;
const GRID_PAD: f32 = tokens::SPACE_4;

/// Full-size images kept decoded (each 4K fp16 frame is ~66 MB; five is
/// a comfortable working set for flipping back and forth).
const FULL_CACHE: usize = 5;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Grid,
    Viewer,
}

pub struct GalleryApp {
    files: Vec<PathBuf>,
    names: Vec<String>,
    thumbs: Vec<Option<Image>>,
    metas: Vec<Option<ImageMeta>>,
    errors: Vec<Option<String>>,
    fulls: LruCache<usize, Image>,
    /// Full-size decodes in flight (cleared on arrival so LRU-evicted
    /// entries can be re-requested later).
    full_requested: std::collections::HashSet<usize>,
    loaded_count: usize,

    mode: Mode,
    selected: usize,
    /// Viewer preview toggle: tonemap the image to SDR (`Standard`)
    /// instead of full panel headroom — "how would this look on an SDR
    /// screen?" without leaving the chair.
    sdr_preview: bool,
    /// Columns from the last `build` — `on_event` needs it for up/down
    /// navigation, and `build` is `&self`.
    cols: Cell<usize>,
    scroll_requests: RefCell<Vec<ScrollRequest>>,

    loader: Loader,
    results: LoaderResults,

    /// Worker count and wakeup handle, kept so a newly picked folder
    /// can spawn a replacement loader pool.
    workers: usize,
    wakeup: SharedWakeup,
    /// Folder dialog in flight: the picker thread sends the chosen
    /// directory (or `None` on cancel) and pokes the wakeup.
    picker: Option<Receiver<Option<PathBuf>>>,
    /// Last open-folder problem (empty folder, unreadable dir) — shown
    /// on the welcome screen / as a toolbar badge until the next open.
    notice: Option<String>,
}

impl GalleryApp {
    pub fn new(files: Vec<PathBuf>, workers: usize, wakeup: SharedWakeup) -> Self {
        let n = files.len();
        let names = files
            .iter()
            .map(|p| {
                p.file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_else(|| p.display().to_string())
            })
            .collect();
        let (loader, results) = Loader::spawn(files.clone(), workers, wakeup.clone());
        Self {
            files,
            names,
            thumbs: vec![None; n],
            metas: vec![None; n],
            errors: vec![None; n],
            fulls: LruCache::new(NonZeroUsize::new(FULL_CACHE).expect("nonzero")),
            full_requested: std::collections::HashSet::new(),
            loaded_count: 0,
            mode: Mode::Grid,
            selected: 0,
            sdr_preview: false,
            cols: Cell::new(1),
            scroll_requests: RefCell::new(Vec::new()),
            loader,
            results,
            workers,
            wakeup,
            picker: None,
            notice: None,
        }
    }

    /// Swap the whole collection for `files`: stop the old worker pool
    /// (its in-flight results go to a dropped channel) and rebuild every
    /// per-collection field via `new`.
    fn open_collection(&mut self, files: Vec<PathBuf>) {
        self.loader.shutdown();
        let picker = self.picker.take();
        *self = Self::new(files, self.workers, self.wakeup.clone());
        self.picker = picker;
    }

    /// Launch the system folder picker (XDG portal) on its own thread —
    /// the dialog blocks until dismissed, and the UI keeps rendering.
    fn open_folder_dialog(&mut self) {
        if self.picker.is_some() {
            return; // one dialog at a time
        }
        let (tx, rx) = channel();
        self.picker = Some(rx);
        let wakeup = self.wakeup.clone();
        let start_dir = self
            .files
            .first()
            .and_then(|p| p.parent())
            .map(PathBuf::from);
        std::thread::Builder::new()
            .name("folder-picker".into())
            .spawn(move || {
                let mut dialog = rfd::FileDialog::new().set_title("Open image folder");
                if let Some(dir) = start_dir {
                    dialog = dialog.set_directory(dir);
                }
                let _ = tx.send(dialog.pick_folder());
                loader::wake(&wakeup);
            })
            .expect("spawning folder picker");
    }

    /// Picker outcome, if the dialog thread reported one this frame.
    fn drain_picker(&mut self) {
        let Some(rx) = &self.picker else { return };
        let picked = match rx.try_recv() {
            Ok(picked) => picked,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => None,
        };
        self.picker = None;
        let Some(dir) = picked else { return }; // cancelled
        match scan::scan_dir(&dir) {
            Ok(files) if files.is_empty() => {
                self.notice = Some(format!("no supported images in {}", dir.display()));
            }
            Ok(files) => {
                tracing::info!(count = files.len(), dir = %dir.display(), "folder opened");
                self.open_collection(files);
            }
            Err(e) => {
                self.notice = Some(format!("{e:#}"));
            }
        }
    }

    fn select(&mut self, index: usize) {
        self.selected = index.min(self.files.len().saturating_sub(1));
        let cols = self.cols.get().max(1);
        self.scroll_requests.borrow_mut().push(ScrollRequest::new(
            "grid",
            self.selected / cols,
            ScrollAlignment::Visible,
        ));
        if self.mode == Mode::Viewer {
            self.ensure_fulls();
        }
    }

    /// Request the selected full-size image and prefetch its neighbors.
    fn ensure_fulls(&mut self) {
        let want = [
            Some(self.selected),
            self.selected
                .checked_add(1)
                .filter(|&i| i < self.files.len()),
            self.selected.checked_sub(1),
        ];
        for idx in want.into_iter().flatten() {
            if self.fulls.peek(&idx).is_none()
                && self.errors[idx].is_none()
                && self.full_requested.insert(idx)
            {
                self.loader.request_full(idx);
            }
        }
    }

    /// Empty state: nothing open yet (or the picked folder had nothing).
    /// Centered card with the folder picker as the primary action — the
    /// shadcn empty-state shape composed from damascene primitives.
    fn welcome(&self, cx: &BuildCx) -> El {
        let mut body = vec![
            icon("folder")
                .icon_size(40.0)
                .text_color(tokens::MUTED_FOREGROUND),
            h3("No folder open"),
            // Single-line text rows, not `paragraph`: a Hug ancestor
            // (the card) measures wrapped text at its unwrapped height,
            // so the card comes out short and the body overflows it —
            // damascene layout bug, caught by the Overflow lint.
            text("Browse folders of images — HDR formats included.").muted(),
            text("JPEG XR · JXL · AVIF · EXR · Radiance · PNG · JPEG · WebP")
                .caption()
                .muted(),
        ];
        if let Some(notice) = &self.notice {
            body.push(badge(notice.clone()).warning().key("notice"));
        }
        body.push(
            button_with_icon("folder", "Open Folder…")
                .primary()
                .key("open-folder"),
        );
        body.push(
            text("o opens this dialog · or pass a path: damascene-gallery DIR")
                .caption()
                .muted(),
        );

        column([
            toolbar([
                toolbar_title("Damascene Gallery"),
                spacer(),
                color_mode_badge(cx),
            ]),
            column([card([column(body)
                .gap(tokens::SPACE_3)
                .align(Align::Center)
                .padding(tokens::SPACE_8)
                // Fixed body width: the default card stretches Fill.
                .width(Size::Fixed(480.0))])])
            .align(Align::Center)
            .justify(Justify::Center)
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
        .height(Size::Fill(1.0))
    }

    fn grid(&self, cx: &BuildCx) -> El {
        let vw = cx.viewport_width().unwrap_or(1280.0);
        // Page padding + grid-card inner padding on both sides, plus a
        // little slack for the card stroke and scrollbar gutter.
        let avail = vw - 2.0 * (tokens::SPACE_4 + GRID_PAD) - 16.0;
        let cols = (((avail + TILE_GAP) / (TILE_W + TILE_GAP)) as usize).max(1);
        self.cols.set(cols);
        let rows = self.files.len().div_ceil(cols);

        // Snapshots for the 'static row builder. Image clones are Arcs.
        let thumbs = self.thumbs.clone();
        let errors = self.errors.clone();
        let names = self.names.clone();
        let count = self.files.len();
        let selected = self.selected;
        let ring = cx.theme().palette().ring;
        let loader = self.loader.clone();

        let list = virtual_list(rows, TILE_H + TILE_GAP, move |row| {
            let mut cells = Vec::with_capacity(cols);
            for col in 0..cols {
                let i = row * cols + col;
                if i >= count {
                    cells.push(spacer().width(Size::Fixed(TILE_W)));
                    continue;
                }
                // The cell carries the key, so its tooltip is the one
                // that can fire — fold decode errors into it.
                let mut tip = names[i].clone();
                let cell = match (&thumbs[i], &errors[i]) {
                    // ConstrainedHigh (CSS dynamic-range-limit): a wall
                    // of 1000-nit thumbnails would be hostile; cap grid
                    // brights at 2× reference, remastered hue-preserving.
                    (Some(img), _) => image(img.clone())
                        .image_fit(ImageFit::Cover)
                        .dynamic_range_limit(DynamicRangeLimit::ConstrainedHigh)
                        .radius(tokens::RADIUS_MD),
                    (None, Some(err)) => {
                        tip = format!("{} — {err}", names[i]);
                        column([
                            icon("alert-circle"),
                            text(names[i].clone()).caption().muted(),
                        ])
                        .gap(tokens::SPACE_2)
                        .align(Align::Center)
                        .justify(Justify::Center)
                        .radius(tokens::RADIUS_MD)
                    }
                    (None, None) => {
                        // Realized but not decoded yet: jump the queue.
                        loader.request_thumb(i);
                        skeleton().radius(tokens::RADIUS_MD)
                    }
                };
                let mut cell = cell
                    .key(format!("thumb:{i}"))
                    .width(Size::Fixed(TILE_W))
                    .height(Size::Fixed(TILE_H))
                    .focusable()
                    .tooltip(tip);
                if i == selected {
                    cell = cell.stroke(ring).stroke_width(2.0);
                }
                cells.push(cell);
            }
            row_el(cells)
        });

        let mut bar = vec![
            toolbar_title("Damascene Gallery"),
            toolbar_description(format!(
                "{} files — {} loaded",
                self.files.len(),
                self.loaded_count
            )),
            spacer(),
        ];
        if let Some(notice) = &self.notice {
            bar.push(
                badge("folder skipped")
                    .warning()
                    .key("notice")
                    .tooltip(format!("{notice} — previous collection kept")),
            );
        }
        bar.push(color_mode_badge(cx));
        bar.push(
            button_with_icon("folder", "Open Folder")
                .key("open-folder")
                .tooltip("open a different folder (o)"),
        );
        bar.push(text("Enter to view · arrows to move").caption().muted());

        column([
            toolbar(bar),
            card([list
                .key("grid")
                .height(Size::Fill(1.0))
                .padding(GRID_PAD)
                .scrollbar()])
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
        .height(Size::Fill(1.0))
    }

    fn viewer(&self) -> El {
        let i = self.selected;
        let full = self.fulls.peek(&i).cloned();
        let shown = full.clone().or_else(|| self.thumbs[i].clone());

        // NoLimit = the panel's full headroom; content brighter than
        // the panel remasters (BT.2390) instead of clipping — including
        // the whole image on SDR outputs. `t` flips to Standard for an
        // SDR preview of the same image.
        let limit = if self.sdr_preview {
            DynamicRangeLimit::Standard
        } else {
            DynamicRangeLimit::NoLimit
        };
        let canvas = match (shown, &self.errors[i]) {
            (Some(img), _) => image(img)
                .image_fit(ImageFit::Contain)
                .dynamic_range_limit(limit)
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0)),
            (None, Some(err)) => column([icon("alert-circle"), text(err.clone()).muted()])
                .gap(tokens::SPACE_3)
                .align(Align::Center)
                .justify(Justify::Center)
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0)),
            (None, None) => column([spinner()])
                .align(Align::Center)
                .justify(Justify::Center)
                .width(Size::Fill(1.0))
                .height(Size::Fill(1.0)),
        };

        let meta = self.metas[i].as_ref();
        let mut detail = format!("{} / {}", i + 1, self.files.len());
        if let Some(m) = meta {
            detail.push_str(&format!(" · {}×{} · {}", m.width, m.height, m.encoding));
            if let Some(bytes) = m.file_bytes {
                detail.push_str(&format!(" · {}", human_bytes(bytes)));
            }
        }

        // Measured pixel stats (linear, ×reference white) next to what
        // the metadata declared — the interesting gap for HDR files.
        let stats_line = meta.map(|m| {
            let s = &m.stats;
            let max = s.max_rgb[0].max(s.max_rgb[1]).max(s.max_rgb[2]);
            let mut t = format!(
                "max RGB {:.2} / {:.2} / {:.2} ×ref · peak {:.0} nits measured",
                s.max_rgb[0],
                s.max_rgb[1],
                s.max_rgb[2],
                max * m.reference_nits,
            );
            if let Some(declared) = m.peak_nits {
                t.push_str(&format!(", {declared:.0} declared"));
            }
            t.push_str(&format!(
                " · mean luminance {:.3} · {:.1}% above ref white",
                s.mean_luminance,
                s.above_reference * 100.0,
            ));
            t
        });

        let mut bar = vec![
            text(self.names[i].clone()).bold(),
            text(detail).caption().muted(),
            spacer(),
        ];
        if full.is_none() && self.errors[i].is_none() {
            bar.push(spinner());
            bar.push(text("full resolution…").caption().muted());
        }
        if self.sdr_preview {
            bar.push(badge("SDR preview").warning().key("sdr-preview").tooltip(
                "tonemapped to reference white (dynamic-range-limit: standard) — press t for full headroom",
            ));
        }
        bar.push(text("t = SDR preview · Esc to close").caption().muted());

        let mut info = vec![row_el(bar)
            .gap(tokens::SPACE_3)
            .align(Align::Center)
            .width(Size::Fill(1.0))];
        if let Some(stats) = stats_line {
            info.push(text(stats).caption().muted());
        }

        column([
            canvas,
            card([column(info)
                .gap(tokens::SPACE_2)
                .padding(tokens::SPACE_3)
                .width(Size::Fill(1.0))])
            .width(Size::Fill(1.0)),
        ])
        .gap(tokens::SPACE_3)
        .width(Size::Fill(1.0))
        .height(Size::Fill(1.0))
    }
}

impl App for GalleryApp {
    fn before_build(&mut self) {
        self.drain_picker();
        for loaded in self.results.drain() {
            match loaded {
                Loaded::Thumb { index, image, meta } => {
                    self.thumbs[index] = Some(image);
                    if self.metas[index].is_none() {
                        self.metas[index] = Some(meta);
                    }
                    self.loaded_count += 1;
                }
                Loaded::Full { index, image, meta } => {
                    self.fulls.put(index, image);
                    self.full_requested.remove(&index);
                    self.metas[index] = Some(meta);
                }
                Loaded::Failed { index, kind, error } => {
                    if kind == JobKind::Full {
                        self.full_requested.remove(&index);
                    }
                    self.errors[index] = Some(error);
                }
            }
        }
    }

    fn build(&self, cx: &BuildCx) -> El {
        let page = if self.files.is_empty() {
            self.welcome(cx)
        } else {
            match self.mode {
                Mode::Grid => self.grid(cx),
                Mode::Viewer => self.viewer(),
            }
        };
        // Page scaffold (the hero-fixture idiom): a themed background
        // layer under content padded in from the window edges — rounded
        // window corners never clip chrome. Overlay root so the library
        // can float tooltip layers above it.
        overlays(
            stack([
                column(Vec::<El>::new())
                    .fill(tokens::BACKGROUND)
                    .width(Size::Fill(1.0))
                    .height(Size::Fill(1.0)),
                page.padding(tokens::SPACE_4)
                    .width(Size::Fill(1.0))
                    .height(Size::Fill(1.0)),
            ])
            .width(Size::Fill(1.0))
            .height(Size::Fill(1.0)),
            [],
        )
    }

    fn hotkeys(&self) -> Vec<(KeyChord, String)> {
        vec![
            (KeyChord::named(UiKey::ArrowLeft), "left".into()),
            (KeyChord::named(UiKey::ArrowRight), "right".into()),
            (KeyChord::named(UiKey::ArrowUp), "up".into()),
            (KeyChord::named(UiKey::ArrowDown), "down".into()),
            (KeyChord::named(UiKey::Home), "home".into()),
            (KeyChord::named(UiKey::End), "end".into()),
            (KeyChord::named(UiKey::Enter), "open".into()),
            (KeyChord::vim('h'), "left".into()),
            (KeyChord::vim('l'), "right".into()),
            (KeyChord::vim('j'), "down".into()),
            (KeyChord::vim('k'), "up".into()),
            (KeyChord::vim('t'), "sdr-preview".into()),
            (KeyChord::vim('o'), "open-folder".into()),
        ]
    }

    fn on_event(&mut self, event: UiEvent, _cx: &EventCx) {
        // Folder picking works from every screen (welcome button, grid
        // toolbar button, `o` anywhere).
        if event.is_click_or_activate("open-folder") || event.is_hotkey("open-folder") {
            self.open_folder_dialog();
            return;
        }

        if event.kind == UiEventKind::Escape {
            self.mode = Mode::Grid;
            if !self.files.is_empty() {
                self.select(self.selected); // re-anchor the grid scroll
            }
            return;
        }

        // Everything below navigates the collection.
        if self.files.is_empty() {
            return;
        }
        let last = self.files.len().saturating_sub(1);
        let cols = self.cols.get().max(1);

        if let Some(key) = event.target_key() {
            if let Some(i) = key.strip_prefix("thumb:").and_then(|s| s.parse().ok()) {
                if event.kind == UiEventKind::Click && event.click_count >= 2 {
                    self.select(i);
                    self.mode = Mode::Viewer;
                    self.ensure_fulls();
                    return;
                }
                if event.is_click_or_activate(key) {
                    self.select(i);
                    return;
                }
            }
        }

        if event.is_hotkey("open") {
            self.mode = Mode::Viewer;
            self.ensure_fulls();
        } else if event.is_hotkey("sdr-preview") {
            self.sdr_preview = !self.sdr_preview;
        } else if event.is_hotkey("left") {
            self.select(self.selected.saturating_sub(1));
        } else if event.is_hotkey("right") {
            self.select((self.selected + 1).min(last));
        } else if event.is_hotkey("up") {
            if self.mode == Mode::Grid {
                self.select(self.selected.saturating_sub(cols));
            }
        } else if event.is_hotkey("down") {
            if self.mode == Mode::Grid {
                self.select((self.selected + cols).min(last));
            }
        } else if event.is_hotkey("home") {
            self.select(0);
        } else if event.is_hotkey("end") {
            self.select(last);
        }
    }

    fn drain_scroll_requests(&mut self) -> Vec<ScrollRequest> {
        std::mem::take(&mut *self.scroll_requests.borrow_mut())
    }
}

/// Decimal units, matching what file managers show.
fn human_bytes(b: u64) -> String {
    let b = b as f64;
    if b >= 1e9 {
        format!("{:.2} GB", b / 1e9)
    } else if b >= 1e6 {
        format!("{:.1} MB", b / 1e6)
    } else if b >= 1e3 {
        format!("{:.0} kB", b / 1e3)
    } else {
        format!("{b:.0} B")
    }
}

/// `row` the layout constructor collides with `row` loop variables; tiny
/// alias keeps call sites readable.
fn row_el<I: IntoIterator<Item = El>>(children: I) -> El {
    damascene_core::row(children).gap(TILE_GAP)
}

/// What the host negotiated with the display server, as a toolbar badge.
/// The Linux host never attaches an image description (the Vulkan WSI
/// swapchain colorspace carries the tag), so `attached` is meaningless
/// here — the real signals are the compositor's preferred-target feedback
/// (`indicates_hdr`) and the swapchain format actually chosen
/// (`Rgba16Float` = extended-range scRGB out).
fn color_mode_badge(cx: &BuildCx) -> El {
    use damascene_core::color::ColorManagementStatus;

    let Some(diag) = cx.diagnostics() else {
        return badge("SDR");
    };
    let fp16 = diag
        .surface_color
        .as_ref()
        .is_some_and(|s| s.chosen_format == "Rgba16Float");
    let b = match &diag.color_management {
        ColorManagementStatus::Available { targets, .. } => {
            if fp16 && targets.indicates_hdr() {
                let peak = targets
                    .target_max_luminance_nits
                    .map(|n| format!(" · {n:.0} nits"))
                    .unwrap_or_default();
                badge(format!("HDR · scRGB{peak}"))
                    .success()
                    .tooltip("extended-range Rgba16Float swapchain; compositor reports HDR output")
            } else {
                badge("SDR").tooltip("color management available; output reports no HDR headroom")
            }
        }
        _ => badge("SDR").tooltip("no wp_color_management_v1 on this host"),
    };
    // Tooltips only fire on keyed nodes (hit-test returns keyed leaves).
    b.key("color-mode")
}

#[cfg(test)]
mod tests {
    use super::*;
    use damascene_core::{render_bundle_themed, Rect, Theme};

    /// A gallery with a representative mix of cell states: loaded
    /// thumbnails, still-loading skeletons, and a decode failure.
    fn test_app() -> GalleryApp {
        let files: Vec<PathBuf> = (0..7)
            .map(|i| PathBuf::from(format!("{i:03}.jxr")))
            .collect();
        let mut app = GalleryApp::new(files, 1, SharedWakeup::default());
        let px = Image::from_rgba8(2, 2, vec![128u8; 16]);
        for i in [0usize, 1, 3, 5] {
            app.thumbs[i] = Some(px.clone());
            app.metas[i] = Some(crate::convert::ImageMeta {
                width: 3840,
                height: 2160,
                encoding: "fp16 linear / sRGB".into(),
                peak_nits: Some(1000.0),
                reference_nits: 80.0,
                stats: crate::convert::PixelStats {
                    max_rgb: [12.48, 11.20, 8.91],
                    mean_luminance: 0.182,
                    above_reference: 0.234,
                },
                file_bytes: Some(24_300_000),
            });
            app.loaded_count += 1;
        }
        app.errors[2] = Some("decoding 002.jxr: bad header".into());
        app
    }

    fn lint_findings(app: &GalleryApp) -> Vec<String> {
        let theme = Theme::default();
        let (w, h) = (1280.0, 800.0);
        let diag = damascene_core::HostDiagnostics::default();
        let cx = BuildCx::new(&theme)
            .with_viewport(w, h)
            .with_diagnostics(&diag);
        let mut tree = app.build(&cx);
        let bundle = render_bundle_themed(&mut tree, Rect::new(0.0, 0.0, w, h), &theme);
        bundle
            .lint
            .findings
            .iter()
            .map(|f| format!("{f:?}"))
            .collect()
    }

    #[test]
    fn grid_tree_lints_clean() {
        let mut app = test_app();
        let findings = lint_findings(&app);
        assert!(
            findings.is_empty(),
            "grid lint findings:\n{}",
            findings.join("\n")
        );

        // A failed open keeps the collection and shows a toolbar badge.
        app.notice = Some("no supported images in /tmp/empty".into());
        let findings = lint_findings(&app);
        assert!(
            findings.is_empty(),
            "grid (notice) lint findings:\n{}",
            findings.join("\n")
        );
    }

    /// Swapping collections (the folder picker's path) rebuilds all
    /// per-collection state and returns to a fresh grid.
    #[test]
    fn open_collection_resets_state() {
        let mut app = test_app();
        app.mode = Mode::Viewer;
        app.selected = 5;
        app.sdr_preview = true;

        app.open_collection(vec![PathBuf::from("other.jxr")]);

        assert_eq!(app.files.len(), 1);
        assert_eq!(app.selected, 0);
        assert!(app.mode == Mode::Grid);
        assert!(!app.sdr_preview);
        assert_eq!(app.loaded_count, 0);
        assert!(app.thumbs.iter().all(Option::is_none));
        assert!(app.errors.iter().all(Option::is_none));
    }

    /// Debug helper: dump the welcome tree's SVG + layout to /tmp.
    /// `cargo test dump_welcome -- --ignored`
    #[test]
    #[ignore]
    fn dump_welcome_artifacts() {
        let app = GalleryApp::new(Vec::new(), 1, SharedWakeup::default());
        let theme = Theme::default();
        let (w, h) = (1280.0, 800.0);
        let diag = damascene_core::HostDiagnostics::default();
        let cx = BuildCx::new(&theme)
            .with_viewport(w, h)
            .with_diagnostics(&diag);
        let mut tree = app.build(&cx);
        let bundle = render_bundle_themed(&mut tree, Rect::new(0.0, 0.0, w, h), &theme);
        std::fs::write("/tmp/welcome.svg", &bundle.svg).unwrap();
        std::fs::write("/tmp/welcome-tree.txt", &bundle.tree_dump).unwrap();
    }

    #[test]
    fn welcome_tree_lints_clean() {
        let mut app = GalleryApp::new(Vec::new(), 1, SharedWakeup::default());
        let findings = lint_findings(&app);
        assert!(
            findings.is_empty(),
            "welcome lint findings:\n{}",
            findings.join("\n")
        );

        // Empty/unreadable folder reports stay on the welcome screen.
        app.notice = Some("no supported images in /tmp/empty".into());
        let findings = lint_findings(&app);
        assert!(
            findings.is_empty(),
            "welcome (notice) lint findings:\n{}",
            findings.join("\n")
        );
    }

    #[test]
    fn viewer_tree_lints_clean() {
        let mut app = test_app();
        app.mode = Mode::Viewer;
        app.selected = 3; // loaded thumb, full pending → spinner branch
        let findings = lint_findings(&app);
        assert!(
            findings.is_empty(),
            "viewer lint findings:\n{}",
            findings.join("\n")
        );

        app.selected = 2; // decode-failure branch
        let findings = lint_findings(&app);
        assert!(
            findings.is_empty(),
            "viewer (error) lint findings:\n{}",
            findings.join("\n")
        );

        app.selected = 4; // nothing loaded → full-canvas spinner branch
        let findings = lint_findings(&app);
        assert!(
            findings.is_empty(),
            "viewer (loading) lint findings:\n{}",
            findings.join("\n")
        );

        app.selected = 3;
        app.sdr_preview = true; // SDR-preview badge branch
        let findings = lint_findings(&app);
        assert!(
            findings.is_empty(),
            "viewer (sdr preview) lint findings:\n{}",
            findings.join("\n")
        );
    }
}
