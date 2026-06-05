//! Background decode workers.
//!
//! Decoding a 4K JPEG XR takes long enough that the UI thread must never
//! do it. A small pool pulls jobs off a two-tier queue — full-size
//! requests (the viewer is waiting) ahead of thumbnails, and within
//! thumbnails, rows the grid actually realized ahead of the background
//! sweep — decodes with the copied prism-bg pipeline, converts to
//! damascene images, and posts results back over a channel. Each result
//! pokes the host's [`Wakeup`] so the idle event loop renders a frame.

use std::collections::{HashSet, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};

use damascene_core::image::Image;
use damascene_winit_wgpu::Wakeup;

use crate::convert::{self, ImageMeta};
use crate::decode;

/// The host's wakeup handle, shared by everything that produces results
/// off-thread (decode workers, the folder-picker dialog thread). The
/// host delivers exactly one [`Wakeup`] — which isn't `Clone` — so it
/// lives behind an `Arc` and is `None` until the event loop starts.
pub type SharedWakeup = Arc<Mutex<Option<Wakeup>>>;

/// Wake the host loop for a frame, if it exists yet.
pub fn wake(wakeup: &SharedWakeup) {
    if let Some(w) = wakeup.lock().unwrap().as_ref() {
        w.wake();
    }
}

/// Long edge of grid thumbnails, in pixels. f16 linear RGBA → ~0.7 MB per
/// 16:9 thumb; a 300-file collection stays around 200 MB resident.
pub const THUMB_EDGE: u32 = 384;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JobKind {
    Thumb,
    Full,
}

pub enum Loaded {
    Thumb {
        index: usize,
        image: Image,
        meta: ImageMeta,
    },
    Full {
        index: usize,
        image: Image,
        meta: ImageMeta,
    },
    Failed {
        index: usize,
        kind: JobKind,
        error: String,
    },
}

struct Queue {
    /// Full-size decodes, most recent request first.
    full: VecDeque<usize>,
    /// Thumbnails the grid realized (visible or about to be).
    thumb_hot: VecDeque<usize>,
    /// Startup sweep over the whole collection, in file order.
    thumb_sweep: VecDeque<usize>,
    /// Everything queued or currently decoding, to dedupe requests.
    pending: HashSet<(JobKind, usize)>,
    shutdown: bool,
}

struct Shared {
    queue: Mutex<Queue>,
    cond: Condvar,
    results: Mutex<Sender<Loaded>>,
    wakeup: SharedWakeup,
    files: Vec<PathBuf>,
}

/// Handle owned by the app; clones go into `build_row` closures.
#[derive(Clone)]
pub struct Loader {
    shared: Arc<Shared>,
}

pub struct LoaderResults {
    rx: Receiver<Loaded>,
}

impl LoaderResults {
    pub fn drain(&self) -> Vec<Loaded> {
        self.rx.try_iter().collect()
    }
}

impl Loader {
    /// Spawn `workers` decode threads over `files`. The whole collection
    /// is queued for the thumbnail sweep immediately.
    pub fn spawn(
        files: Vec<PathBuf>,
        workers: usize,
        wakeup: SharedWakeup,
    ) -> (Loader, LoaderResults) {
        let (tx, rx) = channel();
        let shared = Arc::new(Shared {
            queue: Mutex::new(Queue {
                full: VecDeque::new(),
                thumb_hot: VecDeque::new(),
                thumb_sweep: (0..files.len()).collect(),
                pending: (0..files.len()).map(|i| (JobKind::Thumb, i)).collect(),
                shutdown: false,
            }),
            cond: Condvar::new(),
            results: Mutex::new(tx),
            wakeup,
            files,
        });
        for n in 0..workers {
            let shared = Arc::clone(&shared);
            std::thread::Builder::new()
                .name(format!("decode-{n}"))
                .spawn(move || worker(shared))
                .expect("spawning decode worker");
        }
        (Loader { shared }, LoaderResults { rx })
    }

    /// Stop the worker pool: clear the queues and wake every worker so
    /// it exits. Called when the app swaps to a new collection — jobs
    /// already mid-decode finish and post into the old (now dropped)
    /// channel, which is harmless.
    pub fn shutdown(&self) {
        let mut q = self.shared.queue.lock().unwrap();
        q.shutdown = true;
        q.full.clear();
        q.thumb_hot.clear();
        q.thumb_sweep.clear();
        self.shared.cond.notify_all();
    }

    /// Jump a thumbnail to the front of the queue (a grid row realized
    /// it). No-op if done, queued, decoding, or out of range.
    pub fn request_thumb(&self, index: usize) {
        if index >= self.shared.files.len() {
            debug_assert!(
                false,
                "request_thumb({index}) beyond {} files",
                self.shared.files.len()
            );
            return;
        }
        let mut q = self.shared.queue.lock().unwrap();
        if q.pending.contains(&(JobKind::Thumb, index)) {
            // Already queued — promote out of the sweep if it's there.
            if let Some(pos) = q.thumb_sweep.iter().position(|&i| i == index) {
                q.thumb_sweep.remove(pos);
                q.thumb_hot.push_back(index);
                self.shared.cond.notify_one();
            }
            return;
        }
        q.pending.insert((JobKind::Thumb, index));
        q.thumb_hot.push_back(index);
        self.shared.cond.notify_one();
        drop(q);
    }

    /// Queue a full-size decode (viewer navigation / prefetch). Most
    /// recent request wins the front slot.
    pub fn request_full(&self, index: usize) {
        if index >= self.shared.files.len() {
            debug_assert!(
                false,
                "request_full({index}) beyond {} files",
                self.shared.files.len()
            );
            return;
        }
        let mut q = self.shared.queue.lock().unwrap();
        if !q.pending.insert((JobKind::Full, index)) {
            return;
        }
        q.full.push_front(index);
        self.shared.cond.notify_one();
    }
}

fn worker(shared: Arc<Shared>) {
    loop {
        let (kind, index) = {
            let mut q = shared.queue.lock().unwrap();
            loop {
                if q.shutdown {
                    return;
                }
                if let Some(i) = q.full.pop_front() {
                    break (JobKind::Full, i);
                }
                if let Some(i) = q.thumb_hot.pop_front() {
                    break (JobKind::Thumb, i);
                }
                if let Some(i) = q.thumb_sweep.pop_front() {
                    break (JobKind::Thumb, i);
                }
                q = shared.cond.wait(q).unwrap();
            }
        };

        let path = &shared.files[index];
        let result = match decode::load_straight(path) {
            Ok(decoded) => {
                let meta = convert::meta_of(&decoded);
                match kind {
                    JobKind::Thumb => Loaded::Thumb {
                        index,
                        image: convert::thumbnail(&decoded, THUMB_EDGE),
                        meta,
                    },
                    JobKind::Full => Loaded::Full {
                        index,
                        image: convert::to_damascene(&decoded),
                        meta,
                    },
                }
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %format!("{e:#}"), "decode failed");
                Loaded::Failed {
                    index,
                    kind,
                    error: format!("{e:#}"),
                }
            }
        };

        shared.queue.lock().unwrap().pending.remove(&(kind, index));

        if shared.results.lock().unwrap().send(result).is_err() {
            return; // receiver gone (collection swapped or app exited)
        }
        wake(&shared.wakeup);
    }
}
