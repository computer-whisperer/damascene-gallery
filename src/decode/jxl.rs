//! JPEG XL via jxl-oxide. The richest source of color metadata we handle:
//! enum color encodings map straight to CICP (PQ HDR included), and
//! `intensity_target` gives content peak luminance for the description's
//! mastering metadata.
//!
//! HLG is the one encoding prism doesn't advertise — jxl-oxide can render
//! into a requested encoding, so HLG content is re-rendered as BT.2100 PQ
//! before upload.

use anyhow::{bail, Result};
use jxl_oxide::{EnumColourEncoding, HdrType, JxlImage, RenderingIntent};

use super::{RawColor, RawImage, RawPixels};
use crate::color::{encoding_from_cicp, Luminances, Tf};

pub fn decode(data: &[u8]) -> Result<RawImage> {
    let mut image = JxlImage::builder()
        .read(std::io::Cursor::new(data))
        .map_err(|e| anyhow::anyhow!("reading JXL header: {e}"))?;

    if image.hdr_type() == Some(HdrType::Hlg) {
        // prism has no HLG TF; ask jxl-oxide for PQ instead.
        image.request_color_encoding(EnumColourEncoding::bt2100_pq(RenderingIntent::Relative));
    }

    let render = image
        .render_frame(0)
        .map_err(|e| anyhow::anyhow!("rendering JXL frame: {e}"))?;
    let mut stream = render.stream();
    let (width, height) = (stream.width(), stream.height());
    let channels = stream.channels() as usize;
    if channels != 3 && channels != 4 {
        bail!("unsupported JXL channel count {channels} (CMYK?)");
    }
    let n = width as usize * height as usize;
    let mut samples = vec![0f32; n * channels];
    stream.write_to_buffer(&mut samples);

    // Expand to RGBA straight alpha.
    let rgba: Vec<f32> = if channels == 4 {
        samples
    } else {
        let mut out = Vec::with_capacity(n * 4);
        for px in samples.chunks_exact(3) {
            out.extend_from_slice(&[px[0], px[1], px[2], 1.0]);
        }
        out
    };

    let metadata = &image.image_header().metadata;
    let bits = metadata.bit_depth.bits_per_sample();
    let intensity_target = metadata.tone_mapping.intensity_target;

    let color = match image.rendered_cicp() {
        Some([primaries, tf, _matrix, full_range]) => {
            match encoding_from_cicp(primaries, tf, full_range != 0) {
                Some(mut enc) => {
                    // PQ values are absolute; intensity_target declares the
                    // content peak. Pass it as the primary volume max so
                    // tone mapping knows the real ceiling.
                    if enc.tf == Tf::Pq && intensity_target > 0.0 {
                        enc.luminances = Some(Luminances {
                            min: 0.0,
                            max: intensity_target as f64,
                            reference: 203.0,
                        });
                    }
                    RawColor::Encoding(enc)
                }
                None => {
                    tracing::warn!(
                        primaries,
                        tf,
                        "JXL CICP not expressible; resolving via synthesized ICC"
                    );
                    RawColor::Icc(image.rendered_icc())
                }
            }
        }
        None => RawColor::Icc(image.rendered_icc()),
    };

    // SDR content from ≤8-bit sources doesn't need fp16 on the wire.
    let display_referred_sdr = matches!(
        &color,
        RawColor::Encoding(e) if matches!(e.tf, Tf::Srgb | Tf::Gamma22 | Tf::Bt1886)
    );
    let pixels = if bits <= 8 && display_referred_sdr {
        RawPixels::Rgba8(
            rgba.iter()
                .map(|&v| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8)
                .collect(),
        )
    } else {
        RawPixels::RgbaF32(rgba)
    };

    Ok(RawImage {
        width,
        height,
        pixels,
        color,
    })
}
