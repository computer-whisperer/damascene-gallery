//! PNG via the `png` crate directly (the `image` facade hides the color
//! chunks). Metadata priority per the PNG 3 spec: `cICP` > `iCCP` > `sRGB`
//! > `gAMA`+`cHRM` > assumed sRGB.

use anyhow::{bail, Context, Result};

use super::{RawColor, RawImage, RawPixels};
use crate::color::{encoding_from_cicp, Chromaticities, ColorEncoding, PrimaryVolume, Tf};

pub fn decode(data: &[u8]) -> Result<RawImage> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(data));
    // Expand palette / sub-byte depths; keep 16-bit as 16-bit.
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().context("reading PNG header")?;

    let mut buf = vec![
        0u8;
        reader
            .output_buffer_size()
            .context("PNG dimensions overflow")?
    ];
    let frame = reader.next_frame(&mut buf).context("decoding PNG frame")?;
    buf.truncate(frame.buffer_size());
    let (color_type, bit_depth) = reader.output_color_type();

    let info = reader.info();
    let color = resolve_color(info)?;

    let width = frame.width;
    let height = frame.height;
    let n = (width as usize) * (height as usize);

    // Normalize to straight RGBA at the source depth.
    let pixels = match bit_depth {
        png::BitDepth::Eight => RawPixels::Rgba8(to_rgba::<u8>(&buf, n, color_type, 255)?),
        png::BitDepth::Sixteen => {
            // PNG 16-bit samples are big-endian.
            let samples: Vec<u16> = buf
                .chunks_exact(2)
                .map(|c| u16::from_be_bytes([c[0], c[1]]))
                .collect();
            RawPixels::Rgba16(to_rgba::<u16>(&samples, n, color_type, u16::MAX)?)
        }
        other => bail!("unsupported PNG bit depth after expansion: {other:?}"),
    };

    Ok(RawImage {
        width,
        height,
        pixels,
        color,
    })
}

/// Expand Gray / GrayAlpha / RGB / RGBA samples to straight RGBA.
fn to_rgba<T: Copy>(
    samples: &[T],
    n: usize,
    color_type: png::ColorType,
    opaque: T,
) -> Result<Vec<T>> {
    let channels = match color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => bail!("indexed PNG not expanded (decoder bug)"),
    };
    if samples.len() < n * channels {
        bail!("PNG buffer shorter than expected");
    }
    let mut out = Vec::with_capacity(n * 4);
    for px in samples[..n * channels].chunks_exact(channels) {
        match color_type {
            png::ColorType::Grayscale => out.extend_from_slice(&[px[0], px[0], px[0], opaque]),
            png::ColorType::GrayscaleAlpha => out.extend_from_slice(&[px[0], px[0], px[0], px[1]]),
            png::ColorType::Rgb => out.extend_from_slice(&[px[0], px[1], px[2], opaque]),
            png::ColorType::Rgba => out.extend_from_slice(px),
            png::ColorType::Indexed => unreachable!(),
        }
    }
    Ok(out)
}

fn resolve_color(info: &png::Info<'_>) -> Result<RawColor> {
    // cICP wins when we can express it.
    if let Some(cicp) = &info.coding_independent_code_points {
        if let Some(enc) = encoding_from_cicp(
            cicp.color_primaries,
            cicp.transfer_function,
            cicp.is_video_full_range_image,
        ) {
            return Ok(RawColor::Encoding(enc));
        }
        tracing::warn!(
            primaries = cicp.color_primaries,
            tf = cicp.transfer_function,
            "PNG cICP names an encoding we can't express; falling back"
        );
    }
    if let Some(icc) = &info.icc_profile {
        return Ok(RawColor::Icc(icc.to_vec()));
    }
    if info.srgb.is_some() {
        return Ok(RawColor::Encoding(ColorEncoding::SRGB));
    }
    // gAMA/cHRM fallback (the decoder surfaces the raw chunks; the
    // `source_*` fields are encoder-side). gAMA is the *encoding* exponent
    // (sample = intensity^gamma), so decode gamma is its reciprocal.
    if info.gama_chunk.is_some() || info.chrm_chunk.is_some() {
        let tf = match info.gama_chunk.map(|g| 1.0 / g.into_value() as f64) {
            // The classic 45455 gAMA value is sRGB-era 2.2 content.
            Some(g) if (g - 2.2).abs() < 0.05 => Tf::Gamma22,
            Some(g) if (g - 2.4).abs() < 0.05 => Tf::Bt1886,
            Some(g) if (g - 1.0).abs() < 0.01 => Tf::Linear,
            Some(g) => {
                // Odd gamma; nearest of our named TFs would visibly shift
                // midtones, so flag it. Could linearize client-side later.
                tracing::warn!(gamma = g, "PNG gAMA has no named TF match; assuming 2.2");
                Tf::Gamma22
            }
            None => Tf::Srgb,
        };
        let primaries = match info.chrm_chunk {
            Some(c) => {
                let xy = |s: (png::ScaledFloat, png::ScaledFloat)| {
                    (s.0.into_value() as f64, s.1.into_value() as f64)
                };
                PrimaryVolume::Custom(Chromaticities {
                    r: xy(c.red),
                    g: xy(c.green),
                    b: xy(c.blue),
                    w: xy(c.white),
                })
                .snap_to_named(2e-3)
            }
            None => PrimaryVolume::Srgb,
        };
        return Ok(RawColor::Encoding(ColorEncoding {
            tf,
            primaries,
            luminances: None,
        }));
    }
    Ok(RawColor::Encoding(ColorEncoding::SRGB))
}
