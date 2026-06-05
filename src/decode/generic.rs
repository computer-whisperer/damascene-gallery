//! Formats handled through the `image` crate: JPEG, WebP, AVIF, OpenEXR,
//! Radiance HDR.
//!
//! The facade gives us pixels + an ICC profile; per-format color
//! conventions fill the gaps:
//! - JPEG/WebP: ICC if present, else sRGB (JFIF/EXIF convention).
//! - AVIF: the `image` decoder exposes neither the `colr` box nor CICP, so
//!   we parse the `colr` box ourselves — HDR AVIF (PQ/BT.2020 nclx) read
//!   as sRGB would be badly wrong.
//! - EXR: scene-linear by spec; primaries default to Rec.709 (the
//!   `chromaticities` attribute isn't exposed by the `image` crate —
//!   accepted gap, custom-chromaticity EXR wallpapers are vanishingly
//!   rare).
//! - Radiance HDR: linear, Rec.709 primaries by convention.

use anyhow::{bail, Context, Result};
use image::{DynamicImage, ImageDecoder};

use super::{Format, RawColor, RawImage, RawPixels};
use crate::color::{encoding_from_cicp, ColorEncoding, Luminances, PrimaryVolume, Tf};

/// Reference white for scene-linear sources (EXR / Radiance), in cd/m².
/// 1.0 in the file maps to this many nits; prism takes ExtLinear
/// luminance literally (`decode_luminance_scale`, transfer 0). 203 is the
/// BT.2408 HDR reference white.
const SCENE_LINEAR_REF_NITS: f64 = 203.0;

pub fn decode(data: &[u8], format: Format) -> Result<RawImage> {
    let reader = std::io::Cursor::new(data);
    let mut decoder: Box<dyn ImageDecoder> = match format {
        Format::Jpeg => {
            Box::new(image::codecs::jpeg::JpegDecoder::new(reader).context("reading JPEG header")?)
        }
        Format::WebP => {
            Box::new(image::codecs::webp::WebPDecoder::new(reader).context("reading WebP header")?)
        }
        Format::Avif => {
            Box::new(image::codecs::avif::AvifDecoder::new(reader).context("reading AVIF header")?)
        }
        Format::Exr => Box::new(
            image::codecs::openexr::OpenExrDecoder::new(reader).context("reading EXR header")?,
        ),
        Format::Hdr => {
            Box::new(image::codecs::hdr::HdrDecoder::new(reader).context("reading HDR header")?)
        }
        Format::Png | Format::Jxl | Format::Jxr => {
            unreachable!("handled by dedicated decoders")
        }
    };

    let icc = decoder.icc_profile().unwrap_or_default();
    let img = DynamicImage::from_decoder(decoder).context("decoding image")?;
    let (width, height) = (img.width(), img.height());

    let color = resolve_color(data, format, icc)?;

    // Preserve source depth: 8-bit stays 8-bit, 10/12/16-bit rides u16,
    // float stays float.
    let pixels = match &img {
        DynamicImage::ImageRgb32F(_) | DynamicImage::ImageRgba32F(_) => {
            RawPixels::RgbaF32(img.into_rgba32f().into_raw())
        }
        DynamicImage::ImageLuma16(_)
        | DynamicImage::ImageLumaA16(_)
        | DynamicImage::ImageRgb16(_)
        | DynamicImage::ImageRgba16(_) => RawPixels::Rgba16(img.into_rgba16().into_raw()),
        _ => RawPixels::Rgba8(img.into_rgba8().into_raw()),
    };

    Ok(RawImage {
        width,
        height,
        pixels,
        color,
    })
}

fn resolve_color(data: &[u8], format: Format, icc: Option<Vec<u8>>) -> Result<RawColor> {
    match format {
        Format::Jpeg | Format::WebP => Ok(match icc {
            Some(icc) => RawColor::Icc(icc),
            None => RawColor::Encoding(ColorEncoding::SRGB),
        }),
        Format::Avif => {
            // Trust the container's colr box over whatever the decoder
            // surfaced (it surfaces nothing as of image 0.25).
            match parse_avif_colr(data) {
                Some(Colr::Nclx {
                    primaries,
                    tf,
                    full_range,
                }) => match encoding_from_cicp(primaries, tf, full_range) {
                    Some(enc) => Ok(RawColor::Encoding(enc)),
                    None => bail!(
                        "AVIF nclx names CICP primaries={primaries} tf={tf} which prism-bg \
                         can't express (limited-range or exotic encoding)"
                    ),
                },
                Some(Colr::Icc(icc)) => Ok(RawColor::Icc(icc)),
                None => Ok(match icc {
                    Some(icc) => RawColor::Icc(icc),
                    None => RawColor::Encoding(ColorEncoding::SRGB),
                }),
            }
        }
        Format::Exr | Format::Hdr => Ok(RawColor::Encoding(ColorEncoding {
            tf: Tf::Linear,
            primaries: PrimaryVolume::Srgb,
            luminances: Some(Luminances {
                min: 0.0,
                max: 10000.0,
                reference: SCENE_LINEAR_REF_NITS,
            }),
        })),
        Format::Png | Format::Jxl | Format::Jxr => unreachable!(),
    }
}

enum Colr {
    Nclx {
        primaries: u8,
        tf: u8,
        full_range: bool,
    },
    Icc(Vec<u8>),
}

/// Minimal ISO-BMFF walk to the first `colr` box: top-level `meta` →
/// `iprp` → `ipco` → `colr`. Single-image AVIFs keep their one color
/// property there; we don't chase item↔property associations.
fn parse_avif_colr(data: &[u8]) -> Option<Colr> {
    let meta = find_box(data, b"meta")?;
    // `meta` is a FullBox: 4 bytes version/flags before child boxes.
    let iprp = find_box(meta.get(4..)?, b"iprp")?;
    let ipco = find_box(iprp, b"ipco")?;
    let colr = find_box(ipco, b"colr")?;
    match colr.get(..4)? {
        b"nclx" => {
            let primaries = u16::from_be_bytes([*colr.get(4)?, *colr.get(5)?]);
            let tf = u16::from_be_bytes([*colr.get(6)?, *colr.get(7)?]);
            // [8..10] = matrix coefficients (irrelevant post-decode: dav1d
            // already converted YUV→RGB); byte 10 bit 7 = full range.
            let full_range = colr.get(10)? & 0x80 != 0;
            Some(Colr::Nclx {
                primaries: u8::try_from(primaries).ok()?,
                tf: u8::try_from(tf).ok()?,
                full_range,
            })
        }
        b"prof" | b"rICC" => Some(Colr::Icc(colr.get(4..)?.to_vec())),
        _ => None,
    }
}

/// Test hook: expose the colr parse result as plain data.
#[cfg(test)]
pub(super) fn test_parse_avif_colr(data: &[u8]) -> Option<(u8, u8, bool)> {
    match parse_avif_colr(data)? {
        Colr::Nclx {
            primaries,
            tf,
            full_range,
        } => Some((primaries, tf, full_range)),
        Colr::Icc(_) => None,
    }
}

/// Scan sibling boxes in `data` for `kind`, returning its payload.
fn find_box<'a>(data: &'a [u8], kind: &[u8; 4]) -> Option<&'a [u8]> {
    let mut rest = data;
    while rest.len() >= 8 {
        let size = u32::from_be_bytes(rest[0..4].try_into().unwrap()) as usize;
        let name = &rest[4..8];
        // size 1 = 64-bit largesize, size 0 = "to end of file"; neither
        // shows up in the boxes we care about, so treat as unparseable.
        if size < 8 || size > rest.len() {
            return None;
        }
        if name == kind {
            return Some(&rest[8..size]);
        }
        rest = &rest[size..];
    }
    None
}
