//! Bridge from the decode pipeline's [`DecodedImage`] (prism-bg's
//! vocabulary: `ColorEncoding` = TF + primaries + luminances) to
//! damascene's [`Image`] (`ColorSpace` = primaries + transfer + reference
//! luminance).
//!
//! The one semantic gap is HDR luminance anchoring. Damascene's renderer
//! composites in extended-range linear where 1.0 = SDR reference white,
//! and its `to_scrgb_f16` applies *no* luminance rescale — PQ would decode
//! to 1.0 = 10000 nits and render ~50× dark on an scRGB swapchain. So PQ
//! sources are converted to linear here, anchored to their declared
//! reference white (BT.2408's 203 cd/m² when undeclared). Linear (scRGB)
//! sources already mean "multiples of SDR white" and pass through
//! untouched — exactly right for the Windows JXR wallpaper convention.

use damascene_core::color::{ColorSpace, GammaExponent, Primaries, TransferFunction};
use damascene_core::image::Image;

use crate::color::{pq_eotf, PrimaryVolume, Tf};
use crate::decode::{DecodedImage, Pixels};

/// Diffuse/graphics white for PQ content that doesn't declare a reference
/// luminance (ITU-R BT.2408).
const PQ_DEFAULT_REF_NITS: f32 = 203.0;

/// What the gallery shows about a file in the meta bar.
#[derive(Debug, Clone)]
pub struct ImageMeta {
    pub width: u32,
    pub height: u32,
    /// Human description of the source encoding, e.g. "scRGB linear" or
    /// "PQ / BT.2020".
    pub encoding: String,
    /// Declared peak luminance in nits, when the source said.
    pub peak_nits: Option<f64>,
    /// Reference white in nits — converts the relative [`PixelStats`]
    /// to absolute luminance.
    pub reference_nits: f32,
    /// Measured over the actual pixels, not declared by metadata.
    pub stats: PixelStats,
    /// Source file size in bytes (filled by the loader, which has the path).
    pub file_bytes: Option<u64>,
}

/// Single-pass measurements over the decoded pixels, in linear light
/// where 1.0 = the source's reference white.
#[derive(Debug, Clone, Copy)]
pub struct PixelStats {
    /// Per-channel maxima.
    pub max_rgb: [f32; 3],
    /// Mean relative luminance (Y row of the RGB→XYZ matrix for the
    /// source primaries).
    pub mean_luminance: f32,
    /// Fraction of pixels with any channel above reference white.
    pub above_reference: f32,
}

pub fn meta_of(img: &DecodedImage) -> ImageMeta {
    let enc = &img.encoding;
    let tf = match enc.tf {
        Tf::Srgb => "sRGB",
        Tf::Gamma22 => "gamma 2.2",
        Tf::Bt1886 => "BT.1886",
        Tf::Pq => "PQ",
        Tf::Linear => "linear",
    };
    let prim = match enc.primaries {
        PrimaryVolume::Srgb => "sRGB",
        PrimaryVolume::DisplayP3 => "Display-P3",
        PrimaryVolume::Bt2020 => "BT.2020",
        PrimaryVolume::Custom(_) => "custom primaries",
    };
    let depth = match img.pixels {
        Pixels::Rgba8(_) => "8-bit",
        Pixels::Rgba16(_) => "16-bit",
        Pixels::RgbaF16(_) => "fp16",
    };
    let alpha = if img.has_alpha { " · alpha" } else { "" };
    ImageMeta {
        width: img.width,
        height: img.height,
        encoding: format!("{depth} {tf} / {prim}{alpha}"),
        peak_nits: enc.luminances.map(|l| l.max),
        reference_nits: reference_nits(img),
        stats: stats_of(img),
        file_bytes: None,
    }
}

/// Reference white for anchoring relative linear values, by the same
/// rules the conversion paths use: the declared reference when present,
/// else BT.2408's 203 cd/m² for PQ and the tag default otherwise.
fn reference_nits(img: &DecodedImage) -> f32 {
    let declared = img.encoding.luminances.map(|l| l.reference as f32);
    match img.encoding.tf {
        Tf::Pq => declared.unwrap_or(PQ_DEFAULT_REF_NITS),
        Tf::Linear => declared.unwrap_or(ColorSpace::SCRGB_LINEAR.reference_luminance_nits),
        _ => declared.unwrap_or(ColorSpace::SRGB.reference_luminance_nits),
    }
}

/// The source's decode-to-linear mapping, 1.0 = reference white. prism's
/// `Tf::eotf` is deliberately undefined for PQ (it never converts PQ
/// client-side); here the absolute anchor is explicit, so PQ routes
/// through `pq_eotf` and everything else through the TF's own EOTF.
fn linearizer(img: &DecodedImage) -> impl Fn(f32) -> f32 {
    let tf = img.encoding.tf;
    let pq_scale = match tf {
        Tf::Pq => 10_000.0 / reference_nits(img),
        _ => 1.0,
    };
    move |v: f32| match tf {
        Tf::Pq => pq_eotf(v) * pq_scale,
        _ => tf.eotf(v),
    }
}

/// Y row of the RGB→XYZ matrix for the named volumes; custom primaries
/// snap like [`map_primaries`], unmatched ones get the BT.2020 weights.
fn luma_weights(p: PrimaryVolume) -> [f32; 3] {
    match p.snap_to_named(0.01) {
        PrimaryVolume::Srgb => [0.2126, 0.7152, 0.0722],
        PrimaryVolume::DisplayP3 => [0.2290, 0.6917, 0.0793],
        _ => [0.2627, 0.6780, 0.0593],
    }
}

/// One streaming pass over the pixels — no allocation beyond the 8-bit
/// LUT; decode time dominates this on the worker threads.
fn stats_of(img: &DecodedImage) -> PixelStats {
    let lin = linearizer(img);
    let [kr, kg, kb] = luma_weights(img.encoding.primaries);
    let mut max = [0.0f32; 3];
    let mut luma = 0.0f64;
    let mut above = 0u64;
    let mut acc = |r: f32, g: f32, b: f32| {
        max[0] = max[0].max(r);
        max[1] = max[1].max(g);
        max[2] = max[2].max(b);
        luma += (kr * r + kg * g + kb * b) as f64;
        if r > 1.0 || g > 1.0 || b > 1.0 {
            above += 1;
        }
    };
    match &img.pixels {
        Pixels::Rgba8(d) => {
            let lut: Vec<f32> = (0..=255u32).map(|v| lin(v as f32 / 255.0)).collect();
            for px in d.chunks_exact(4) {
                acc(
                    lut[px[0] as usize],
                    lut[px[1] as usize],
                    lut[px[2] as usize],
                );
            }
        }
        Pixels::Rgba16(d) => {
            for px in d.chunks_exact(4) {
                acc(
                    lin(px[0] as f32 / 65535.0),
                    lin(px[1] as f32 / 65535.0),
                    lin(px[2] as f32 / 65535.0),
                );
            }
        }
        Pixels::RgbaF16(d) => {
            for px in d.chunks_exact(4) {
                acc(
                    lin(px[0].to_f32()),
                    lin(px[1].to_f32()),
                    lin(px[2].to_f32()),
                );
            }
        }
    }
    let n = (img.width as u64 * img.height as u64).max(1);
    PixelStats {
        max_rgb: max,
        mean_luminance: (luma / n as f64) as f32,
        above_reference: above as f32 / n as f32,
    }
}

/// Map the pipeline's primaries onto damascene's named set. Custom
/// chromaticities snap to the nearest named volume when they're close;
/// genuinely exotic primaries fall back to BT.2020 as the widest
/// container (hue error beats hard gamut clipping for a viewer).
fn map_primaries(p: PrimaryVolume) -> Primaries {
    match p.snap_to_named(0.01) {
        PrimaryVolume::Srgb => Primaries::Srgb,
        PrimaryVolume::DisplayP3 => Primaries::DisplayP3,
        PrimaryVolume::Bt2020 => Primaries::Bt2020,
        PrimaryVolume::Custom(c) => {
            tracing::warn!(?c, "unmatched custom primaries; tagging BT.2020");
            Primaries::Bt2020
        }
    }
}

/// Convert to a damascene [`Image`], preserving the pixel container where
/// damascene can decode it natively and converting PQ to anchored linear.
pub fn to_damascene(img: &DecodedImage) -> Image {
    let primaries = map_primaries(img.encoding.primaries);
    let (w, h) = (img.width, img.height);

    match img.encoding.tf {
        // Display-referred SDR: damascene decodes these TFs itself.
        // 8-bit sRGB rides the fast `is_srgb8` texture path untouched.
        Tf::Srgb | Tf::Gamma22 | Tf::Bt1886 => {
            let space = ColorSpace {
                primaries,
                transfer: match img.encoding.tf {
                    Tf::Srgb => TransferFunction::Srgb,
                    Tf::Gamma22 => TransferFunction::Gamma(
                        GammaExponent::from_x100(220).expect("2.2 is nonzero"),
                    ),
                    Tf::Bt1886 => TransferFunction::Bt1886,
                    _ => unreachable!(),
                },
                ..ColorSpace::SRGB
            };
            match &img.pixels {
                Pixels::Rgba8(d) => Image::from_rgba8_in(space, w, h, d.clone()),
                Pixels::Rgba16(d) => Image::from_rgba16_in(space, w, h, d.clone()),
                Pixels::RgbaF16(d) => Image::from_rgba_f16_bits_in(space, w, h, f16_bits(d)),
            }
        }

        // Extended linear: 1.0 = reference white by both vocabularies
        // (scRGB anchors at 80 cd/m²; the compositor's SDR white is the
        // render-time anchor either way). Pass through.
        Tf::Linear => {
            let space = ColorSpace {
                primaries,
                transfer: TransferFunction::Linear,
                reference_luminance_nits: img
                    .encoding
                    .luminances
                    .map(|l| l.reference as f32)
                    .unwrap_or(ColorSpace::SCRGB_LINEAR.reference_luminance_nits),
            };
            match &img.pixels {
                Pixels::RgbaF16(d) => Image::from_rgba_f16_bits_in(space, w, h, f16_bits(d)),
                // Linear in integer containers doesn't come out of the
                // decode path, but handle it rather than panic.
                Pixels::Rgba8(d) => Image::from_rgba_f32_in(
                    space,
                    w,
                    h,
                    d.iter().map(|&v| v as f32 / 255.0).collect(),
                ),
                Pixels::Rgba16(d) => Image::from_rgba_f32_in(
                    space,
                    w,
                    h,
                    d.iter().map(|&v| v as f32 / 65535.0).collect(),
                ),
            }
        }

        // PQ: decode to linear anchored at the declared reference white
        // (damascene's own PQ decode would anchor 1.0 at 10000 nits).
        Tf::Pq => {
            let ref_nits = img
                .encoding
                .luminances
                .map(|l| l.reference as f32)
                .unwrap_or(PQ_DEFAULT_REF_NITS);
            let scale = 10_000.0 / ref_nits;
            let space = ColorSpace {
                primaries,
                transfer: TransferFunction::Linear,
                reference_luminance_nits: ref_nits,
            };
            let mut out: Vec<u16>;
            match &img.pixels {
                Pixels::RgbaF16(d) => {
                    out = Vec::with_capacity(d.len());
                    for px in d.chunks_exact(4) {
                        for c in &px[..3] {
                            out.push(half::f16::from_f32(pq_eotf(c.to_f32()) * scale).to_bits());
                        }
                        out.push(px[3].to_bits());
                    }
                }
                Pixels::Rgba16(d) => {
                    out = Vec::with_capacity(d.len());
                    for px in d.chunks_exact(4) {
                        for &c in &px[..3] {
                            out.push(
                                half::f16::from_f32(pq_eotf(c as f32 / 65535.0) * scale).to_bits(),
                            );
                        }
                        out.push(half::f16::from_f32(px[3] as f32 / 65535.0).to_bits());
                    }
                }
                Pixels::Rgba8(d) => {
                    out = Vec::with_capacity(d.len());
                    for px in d.chunks_exact(4) {
                        for &c in &px[..3] {
                            out.push(
                                half::f16::from_f32(pq_eotf(c as f32 / 255.0) * scale).to_bits(),
                            );
                        }
                        out.push(half::f16::from_f32(px[3] as f32 / 255.0).to_bits());
                    }
                }
            }
            Image::from_rgba_f16_bits_in(space, w, h, out)
        }
    }
}

/// Downscale to a thumbnail with at most `max_edge` on the long side,
/// filtering in linear light (box/area average), and return it as
/// extended-range linear f16 so HDR thumbnails still pop. Sources already
/// at or below the target just convert.
pub fn thumbnail(img: &DecodedImage, max_edge: u32) -> Image {
    let primaries = map_primaries(img.encoding.primaries);
    let (sw, sh) = (img.width as usize, img.height as usize);

    let (linear, ref_nits) = to_linear_f32(img);

    let long = sw.max(sh) as u32;
    let space = ColorSpace {
        primaries,
        transfer: TransferFunction::Linear,
        reference_luminance_nits: ref_nits,
    };
    if long <= max_edge {
        let bits = linear
            .iter()
            .map(|&v| half::f16::from_f32(v).to_bits())
            .collect();
        return Image::from_rgba_f16_bits_in(space, img.width, img.height, bits);
    }

    let scale = max_edge as f32 / long as f32;
    let dw = ((sw as f32 * scale).round() as usize).max(1);
    let dh = ((sh as f32 * scale).round() as usize).max(1);

    let horiz = box_resample_rows(&linear, sw, sh, dw);
    let full = box_resample_cols(&horiz, dw, sh, dh);

    let bits = full
        .iter()
        .map(|&v| half::f16::from_f32(v).to_bits())
        .collect();
    Image::from_rgba_f16_bits_in(space, dw as u32, dh as u32, bits)
}

/// Decode any [`DecodedImage`] to straight-alpha linear-light f32 RGBA
/// where 1.0 = SDR reference white. Returns the pixel data and the
/// reference luminance to carry on the tag.
fn to_linear_f32(img: &DecodedImage) -> (Vec<f32>, f32) {
    let lin = linearizer(img);
    let ref_nits = reference_nits(img);

    let n = (img.width as usize) * (img.height as usize) * 4;
    let mut out = Vec::with_capacity(n);
    match &img.pixels {
        Pixels::Rgba8(d) => {
            let lut: Vec<f32> = (0..=255u32).map(|v| lin(v as f32 / 255.0)).collect();
            for px in d.chunks_exact(4) {
                out.push(lut[px[0] as usize]);
                out.push(lut[px[1] as usize]);
                out.push(lut[px[2] as usize]);
                out.push(px[3] as f32 / 255.0);
            }
        }
        Pixels::Rgba16(d) => {
            for px in d.chunks_exact(4) {
                out.push(lin(px[0] as f32 / 65535.0));
                out.push(lin(px[1] as f32 / 65535.0));
                out.push(lin(px[2] as f32 / 65535.0));
                out.push(px[3] as f32 / 65535.0);
            }
        }
        Pixels::RgbaF16(d) => {
            for px in d.chunks_exact(4) {
                out.push(lin(px[0].to_f32()));
                out.push(lin(px[1].to_f32()));
                out.push(lin(px[2].to_f32()));
                out.push(px[3].to_f32());
            }
        }
    }
    (out, ref_nits)
}

/// Area-average horizontally: `src` is `sw`×`sh` RGBA f32, output `dw`×`sh`.
/// Each destination pixel integrates the source span it covers, with
/// fractional weights at the edges, so arbitrary ratios stay alias-free.
fn box_resample_rows(src: &[f32], sw: usize, sh: usize, dw: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; dw * sh * 4];
    let ratio = sw as f32 / dw as f32;
    for y in 0..sh {
        let row = &src[y * sw * 4..(y + 1) * sw * 4];
        let orow = &mut out[y * dw * 4..(y + 1) * dw * 4];
        for dx in 0..dw {
            let start = dx as f32 * ratio;
            let end = (dx as f32 + 1.0) * ratio;
            let mut acc = [0.0f32; 4];
            let mut x = start;
            while x < end {
                let xi = x.floor() as usize;
                let next = (xi + 1) as f32;
                let w = next.min(end) - x;
                let px = &row[xi.min(sw - 1) * 4..];
                for c in 0..4 {
                    acc[c] += px[c] * w;
                }
                x = next;
            }
            let inv = 1.0 / ratio;
            for c in 0..4 {
                orow[dx * 4 + c] = acc[c] * inv;
            }
        }
    }
    out
}

/// Area-average vertically: `src` is `w`×`sh` RGBA f32, output `w`×`dh`.
fn box_resample_cols(src: &[f32], w: usize, sh: usize, dh: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; w * dh * 4];
    let ratio = sh as f32 / dh as f32;
    for dy in 0..dh {
        let start = dy as f32 * ratio;
        let end = (dy as f32 + 1.0) * ratio;
        let orow = &mut out[dy * w * 4..(dy + 1) * w * 4];
        let mut y = start;
        while y < end {
            let yi = y.floor() as usize;
            let next = (yi + 1) as f32;
            let wgt = next.min(end) - y;
            let row = &src[yi.min(sh - 1) * w * 4..];
            for i in 0..w * 4 {
                orow[i] += row[i] * wgt;
            }
            y = next;
        }
        let inv = 1.0 / ratio;
        for v in orow.iter_mut() {
            *v *= inv;
        }
    }
    out
}

fn f16_bits(d: &[half::f16]) -> Vec<u16> {
    d.iter().map(|v| v.to_bits()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decode;

    /// 2×1 → 1×1: area average of linear values, exact.
    #[test]
    fn box_filter_averages_in_linear() {
        let src = vec![
            0.0, 0.0, 0.0, 1.0, //
            1.0, 1.0, 1.0, 1.0,
        ];
        let out = box_resample_rows(&src, 2, 1, 1);
        assert_eq!(out, vec![0.5, 0.5, 0.5, 1.0]);
        let out = box_resample_cols(&src[..], 1, 2, 1);
        assert_eq!(out, vec![0.5, 0.5, 0.5, 1.0]);
    }

    /// Fractional ratio (3 → 2): weights must sum to coverage.
    #[test]
    fn box_filter_fractional_coverage() {
        let src = vec![
            0.0, 0.0, 0.0, 1.0, //
            0.6, 0.6, 0.6, 1.0, //
            1.0, 1.0, 1.0, 1.0,
        ];
        let out = box_resample_rows(&src, 3, 1, 2);
        // dest 0 covers [0, 1.5): (0.0 + 0.5*0.6) / 1.5 = 0.2
        // dest 1 covers [1.5, 3): (0.5*0.6 + 1.0) / 1.5 ≈ 0.8667
        assert!((out[0] - 0.2).abs() < 1e-6, "got {}", out[0]);
        assert!((out[4] - 0.866_666_7).abs() < 1e-6, "got {}", out[4]);
    }

    /// Stats measure in linear light anchored at reference white: an
    /// scRGB f16 source passes through, an sRGB 8-bit source decodes
    /// through the EOTF, and `above_reference` counts pixels, not
    /// channels.
    #[test]
    fn stats_measure_linear_maxima() {
        use crate::color::{ColorEncoding, Tf};

        // 2×1 scRGB linear: one HDR pixel (R hottest), one in range.
        let hdr = DecodedImage {
            width: 2,
            height: 1,
            pixels: Pixels::RgbaF16(
                [4.0f32, 2.0, 1.5, 1.0, 0.25, 0.5, 1.0, 1.0]
                    .iter()
                    .map(|&v| half::f16::from_f32(v))
                    .collect(),
            ),
            encoding: ColorEncoding {
                tf: Tf::Linear,
                primaries: PrimaryVolume::Srgb,
                luminances: None,
            },
            has_alpha: false,
        };
        let s = stats_of(&hdr);
        assert_eq!(s.max_rgb, [4.0, 2.0, 1.5]);
        assert_eq!(s.above_reference, 0.5); // one of two pixels
        let [kr, kg, kb] = luma_weights(PrimaryVolume::Srgb);
        let want = (kr * 4.0 + kg * 2.0 + kb * 1.5 + kr * 0.25 + kg * 0.5 + kb * 1.0) / 2.0;
        assert!((s.mean_luminance - want).abs() < 1e-6);

        let meta = meta_of(&hdr);
        assert_eq!(meta.stats.max_rgb, [4.0, 2.0, 1.5]);
        assert_eq!(
            meta.reference_nits,
            ColorSpace::SCRGB_LINEAR.reference_luminance_nits
        );

        // 1×1 sRGB 8-bit full white: maxima decode to exactly 1.0 and
        // nothing exceeds reference.
        let sdr = DecodedImage {
            width: 1,
            height: 1,
            pixels: Pixels::Rgba8(vec![255, 255, 255, 255]),
            encoding: ColorEncoding::SRGB,
            has_alpha: false,
        };
        let s = stats_of(&sdr);
        assert_eq!(s.max_rgb, [1.0, 1.0, 1.0]);
        assert_eq!(s.above_reference, 0.0);
    }

    /// End-to-end against the real collection when it's mounted: decode
    /// one JPEG XR, convert full + thumbnail, check tags and geometry.
    #[test]
    fn jxr_from_collection_converts() {
        let path = std::path::Path::new("/ceph/public/media/Images/HDR 4K BG/001..jxr");
        if !path.exists() {
            eprintln!("collection not mounted; skipping");
            return;
        }
        let decoded = decode::load_straight(path).expect("decode 001..jxr");
        assert_eq!(
            decoded.encoding.tf,
            Tf::Linear,
            "JXR should be scRGB linear"
        );

        let full = to_damascene(&decoded);
        assert_eq!(full.color_space().transfer, TransferFunction::Linear);
        assert_eq!(full.color_space().primaries, Primaries::Srgb);
        assert_eq!(full.color_space().reference_luminance_nits, 80.0);

        let thumb = thumbnail(&decoded, crate::loader::THUMB_EDGE);
        let long = thumb.width().max(thumb.height());
        assert_eq!(long, crate::loader::THUMB_EDGE);
        let (tw, th) = (thumb.width() as f32, thumb.height() as f32);
        let (sw, sh) = (decoded.width as f32, decoded.height as f32);
        assert!((tw / th - sw / sh).abs() < 0.02, "aspect drifted");
    }
}
