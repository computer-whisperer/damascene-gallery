//! JPEG XR via the `jpegxr` crate (vendored jxrlib — no pure-Rust decoder
//! exists). The Windows HDR wallpaper/screenshot format.
//!
//! Color semantics: JXR carries no embedded description; the pixel format
//! defines it. Float/half formats are scRGB — linear light, sRGB/Rec.709
//! primaries, 1.0 = 80 cd/m², negative and >1.0 values meaningful. That is
//! exactly the protocol's `windows_scrgb` description, which we express
//! parametrically (prism takes ExtLinear reference luminance literally, so
//! ref MUST be 80 — see prism's `CreateWindowsScrgb` arm). 8-bit formats
//! are plain sRGB.

use anyhow::{bail, Result};
use jpegxr::{ImageDecode, PixelFormat, PixelInfo};

use super::{RawColor, RawImage, RawPixels};
use crate::color::{ColorEncoding, Luminances, PrimaryVolume, Tf};

/// scRGB: linear, sRGB primaries, signal 1.0 anchored at 80 cd/m²
/// (125.0 = 10000). Max 10000 matches the spec-fixed windows_scrgb
/// description.
const SCRGB: ColorEncoding = ColorEncoding {
    tf: Tf::Linear,
    primaries: PrimaryVolume::Srgb,
    luminances: Some(Luminances {
        min: 0.0,
        max: 10000.0,
        reference: 80.0,
    }),
};

pub fn decode(data: &[u8]) -> Result<RawImage> {
    let mut dec = ImageDecode::with_reader(std::io::Cursor::new(data))
        .map_err(|e| anyhow::anyhow!("reading JXR header: {e:?}"))?;
    let (w, h) = dec
        .get_size()
        .map_err(|e| anyhow::anyhow!("reading JXR size: {e:?}"))?;
    let pf = dec
        .get_pixel_format()
        .map_err(|e| anyhow::anyhow!("reading JXR pixel format: {e:?}"))?;
    let info = PixelInfo::from_format(pf);

    let (width, height) = (w as u32, h as u32);
    let n = width as usize * height as usize;
    let stride = width as usize * info.bits_per_pixel() / 8;
    let mut raw = vec![0u8; stride * height as usize];
    dec.copy_all(&mut raw, stride)
        .map_err(|e| anyhow::anyhow!("decoding JXR: {e:?}"))?;

    use PixelFormat::*;
    let (pixels, color) = match pf {
        // scRGB half float. The alpha channel is deliberately ignored:
        // Windows HDR wallpapers/screenshots carry an RGB-only codestream,
        // and jxrlib's direct Copy zero-fills A for 64bppRGBAHalf output —
        // honoring it would premultiply the image to black (observed on
        // real wallpaper packs; ground truth via raw-FFI probe).
        PixelFormat64bppRGBAHalf | PixelFormat64bppRGBHalf | PixelFormat48bppRGBHalf => {
            let half = |c: &[u8]| half::f16::from_le_bytes([c[0], c[1]]).to_f32();
            let comps = info.bits_per_pixel() / 16;
            let mut out = Vec::with_capacity(n * 4);
            for px in raw.chunks_exact(comps * 2) {
                out.push(half(&px[0..]));
                out.push(half(&px[2..]));
                out.push(half(&px[4..]));
                out.push(1.0);
            }
            (RawPixels::RgbaF32(out), RawColor::Encoding(SCRGB))
        }
        // scRGB full float (NVIDIA HDR screenshots).
        PixelFormat128bppRGBAFloat
        | PixelFormat128bppPRGBAFloat
        | PixelFormat128bppRGBFloat
        | PixelFormat96bppRGBFloat => {
            // Alpha ignored here too — same zero-filled-A failure mode as
            // the half-float formats above.
            let fl = |c: &[u8]| f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            let comps = info.bits_per_pixel() / 32;
            let mut out = Vec::with_capacity(n * 4);
            for px in raw.chunks_exact(comps * 4) {
                out.extend_from_slice(&[fl(&px[0..]), fl(&px[4..]), fl(&px[8..]), 1.0]);
            }
            (RawPixels::RgbaF32(out), RawColor::Encoding(SCRGB))
        }
        // SDR 8-bit, possibly BGR-ordered.
        PixelFormat24bppBGR | PixelFormat24bppRGB | PixelFormat32bppBGR | PixelFormat32bppBGRA
        | PixelFormat32bppRGBA => {
            let comps = info.bits_per_pixel() / 8;
            let bgr = info.bgr();
            let has_alpha = info.has_alpha();
            let mut out = Vec::with_capacity(n * 4);
            for px in raw.chunks_exact(comps) {
                let (r, b) = if bgr { (px[2], px[0]) } else { (px[0], px[2]) };
                out.extend_from_slice(&[r, px[1], b, if has_alpha { px[3] } else { 255 }]);
            }
            (
                RawPixels::Rgba8(out),
                RawColor::Encoding(ColorEncoding::SRGB),
            )
        }
        other => bail!("unsupported JXR pixel format {other:?} — please report with a sample file"),
    };

    Ok(RawImage {
        width,
        height,
        pixels,
        color,
    })
}
