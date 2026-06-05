//! ICC profile resolution. prism does not advertise `IccV2V4`, so embedded
//! profiles must be expressed parametrically — exactly, not approximately:
//!
//! 1. Profile carries a `cicp` tag → map it directly (v4.4 profiles).
//! 2. Matrix-shaper profile whose TRC matches a named TF (sRGB curve,
//!    γ2.2, γ2.4, linear) within an 8-bit half-step → pixels untouched,
//!    primaries from the colorants (snapped to a named set when they
//!    match). Covers the sRGB/Display-P3/Adobe-RGB profiles that make up
//!    nearly all tagged wallpapers.
//! 3. Matrix-shaper with an odd TRC (ProPhoto γ1.8, lab-made curves) →
//!    re-encode pixels through moxcms to the *same primaries* with the
//!    sRGB TF. Gamut untouched, only the curve is rewritten; result rides
//!    fp16 so the re-quantization is lossless.
//! 4. LUT-based profile → full moxcms conversion to Display-P3-sized
//!    fallback? No: BT.2020 + sRGB TF, fp16. Wide enough for any real
//!    content, and display-referred so the compositor anchors it like all
//!    other SDR content (deliberately NOT ExtLinear: prism takes ExtLinear
//!    luminance literally, which would un-anchor SDR content).

use anyhow::{bail, Context, Result};
use moxcms::{ColorProfile, Layout, ToneReprCurve, TransformOptions};

use crate::color::{
    encoding_from_cicp, Chromaticities, ColorEncoding, PrimaryVolume, Tf, BT2020_CHROMA,
};
use crate::decode::RawPixels;

/// Resolve `icc` against prism's parametric vocabulary, rewriting pixels
/// when the profile can't be expressed by tagging alone.
pub fn resolve_icc(icc: &[u8], pixels: RawPixels) -> Result<(RawPixels, ColorEncoding)> {
    let profile = ColorProfile::new_from_slice(icc).context("parsing ICC profile")?;

    // Case 1: v4.4 cicp tag. The range flag is ignored: pixel data behind
    // an ICC profile is full-range by construction (the flag describes the
    // original video signal, not these samples).
    if let Some(cicp) = &profile.cicp {
        let (p, t) = (
            cicp.color_primaries as u8,
            cicp.transfer_characteristics as u8,
        );
        if let Some(enc) = encoding_from_cicp(p, t, true) {
            return Ok((pixels, enc));
        }
        tracing::debug!(
            primaries = p,
            tf = t,
            "ICC cicp tag not expressible; continuing"
        );
    }

    if profile.is_matrix_shaper() && !has_luts(&profile) {
        let primaries = PrimaryVolume::Custom(profile_chromaticities(&profile)).snap_to_named(5e-3);

        // Case 2: named TRC → tag only.
        if let Some(tf) = classify_trc(&profile) {
            return Ok((
                pixels,
                ColorEncoding {
                    tf,
                    primaries,
                    luminances: None,
                },
            ));
        }

        // Case 3: rewrite the curve, keep the gamut.
        tracing::debug!("ICC matrix-shaper with unnamed TRC; re-encoding through sRGB TF");
        let mut dst = profile.clone();
        let srgb = ColorProfile::new_srgb();
        dst.red_trc = srgb.red_trc.clone();
        dst.green_trc = srgb.green_trc.clone();
        dst.blue_trc = srgb.blue_trc.clone();
        dst.cicp = None;
        let pixels = convert(&profile, &dst, pixels)?;
        return Ok((
            pixels,
            ColorEncoding {
                tf: Tf::Srgb,
                primaries,
                luminances: None,
            },
        ));
    }

    // Case 4: LUT profile → BT.2020 / sRGB-TF.
    tracing::debug!("LUT-based ICC profile; converting to BT.2020 + sRGB TF");
    let mut dst = ColorProfile::new_bt2020();
    let srgb = ColorProfile::new_srgb();
    dst.red_trc = srgb.red_trc.clone();
    dst.green_trc = srgb.green_trc.clone();
    dst.blue_trc = srgb.blue_trc.clone();
    dst.cicp = None;
    let pixels = convert(&profile, &dst, pixels)?;
    Ok((
        pixels,
        ColorEncoding {
            tf: Tf::Srgb,
            primaries: PrimaryVolume::Custom(BT2020_CHROMA).snap_to_named(5e-3),
            luminances: None,
        },
    ))
}

fn has_luts(p: &ColorProfile) -> bool {
    p.lut_a_to_b_perceptual.is_some()
        || p.lut_a_to_b_colorimetric.is_some()
        || p.lut_a_to_b_saturation.is_some()
}

/// PCS illuminant (D50) in XYZ.
const D50: [f64; 3] = [0.9642, 1.0, 0.8249];

/// Bradford cone-response matrix (the standard ICC CHAD basis).
const BRADFORD: Mat3 = Mat3([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0296],
]);

/// Recover the source chromaticities from the profile's colorants.
///
/// ICC stores colorants relative to the D50 PCS, with the actual→D50
/// adaptation in the `chad` tag (v4) or implicit (v2; some v2 profiles
/// store unadapted colorants outright). Conventions in the wild are messy,
/// so decide from the data:
/// - colorants summing to the media white → unadapted, use directly;
/// - a plausible `chad` tag → invert it (and recover the actual white from
///   it when the white tag is D50, the v4 layout);
/// - otherwise → undo a self-computed Bradford D50→media-white adaptation.
///
/// "Plausible" matters: synthesized profiles (moxcms's own defaults) have
/// been seen carrying the raw Bradford cone matrix in `chad`, which is not
/// an adaptation matrix at all.
fn profile_chromaticities(p: &ColorProfile) -> Chromaticities {
    let r = [p.red_colorant.x, p.red_colorant.y, p.red_colorant.z];
    let g = [p.green_colorant.x, p.green_colorant.y, p.green_colorant.z];
    let b = [p.blue_colorant.x, p.blue_colorant.y, p.blue_colorant.z];
    let media = p.media_white_point.map(|w| [w.x, w.y, w.z]).unwrap_or(D50);

    let sum = [r[0] + g[0] + b[0], r[1] + g[1] + b[1], r[2] + g[2] + b[2]];
    let near = |a: [f64; 3], b: [f64; 3]| {
        (a[0] - b[0]).abs() < 0.02 && (a[1] - b[1]).abs() < 0.02 && (a[2] - b[2]).abs() < 0.02
    };

    let (r, g, b, w) = if !near(media, D50) && near(sum, media) {
        // Colorants are unadapted (v2 style).
        (r, g, b, media)
    } else if let Some(chad) = p.chromatic_adaptation.map(mat3_from) {
        if plausible_chad(&chad) {
            let inv = chad.inverse();
            // v4: white tag is D50 and the real white sits behind chad.
            let w = if near(media, D50) {
                inv.mul_vec(D50)
            } else {
                media
            };
            (inv.mul_vec(r), inv.mul_vec(g), inv.mul_vec(b), w)
        } else {
            unadapt_via_bradford(r, g, b, media)
        }
    } else {
        unadapt_via_bradford(r, g, b, media)
    };

    Chromaticities {
        r: xy(r),
        g: xy(g),
        b: xy(b),
        w: xy(w),
    }
}

/// Undo a D50-adaptation by computing the Bradford matrix media→D50
/// ourselves and inverting it. If the media white IS D50 there is nothing
/// to undo.
fn unadapt_via_bradford(
    r: [f64; 3],
    g: [f64; 3],
    b: [f64; 3],
    media: [f64; 3],
) -> ([f64; 3], [f64; 3], [f64; 3], [f64; 3]) {
    let src_cone = BRADFORD.mul_vec(media);
    let dst_cone = BRADFORD.mul_vec(D50);
    let scale = Mat3([
        [dst_cone[0] / src_cone[0], 0.0, 0.0],
        [0.0, dst_cone[1] / src_cone[1], 0.0],
        [0.0, 0.0, dst_cone[2] / src_cone[2]],
    ]);
    let adapt = BRADFORD.inverse().mul(&scale).mul(&BRADFORD);
    let inv = adapt.inverse();
    (inv.mul_vec(r), inv.mul_vec(g), inv.mul_vec(b), media)
}

/// A real ICC `chad` is a near-identity-with-scaled-Z affair; the raw
/// Bradford cone matrix (large off-diagonals) is not.
fn plausible_chad(m: &Mat3) -> bool {
    (0..3).all(|i| {
        (0..3).all(|j| {
            if i == j {
                (0.5..=1.5).contains(&m.0[i][j])
            } else {
                m.0[i][j].abs() < 0.3
            }
        })
    })
}

fn mat3_from(m: moxcms::Matrix3d) -> Mat3 {
    Mat3(m.v)
}

fn xy(v: [f64; 3]) -> (f64, f64) {
    let sum = v[0] + v[1] + v[2];
    if sum.abs() < 1e-9 {
        return (0.0, 0.0);
    }
    (v[0] / sum, v[1] / sum)
}

/// Minimal row-major 3×3 — self-contained so we depend on moxcms's data,
/// not its linear-algebra conventions.
#[derive(Debug, Clone, Copy)]
struct Mat3([[f64; 3]; 3]);

impl Mat3 {
    fn mul_vec(&self, v: [f64; 3]) -> [f64; 3] {
        let m = &self.0;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    fn mul(&self, o: &Mat3) -> Mat3 {
        let mut out = [[0.0; 3]; 3];
        for (i, row) in out.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                *cell = (0..3).map(|k| self.0[i][k] * o.0[k][j]).sum();
            }
        }
        Mat3(out)
    }

    fn inverse(&self) -> Mat3 {
        let m = &self.0;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        let inv_det = 1.0 / det;
        Mat3([
            [
                (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * inv_det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * inv_det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * inv_det,
            ],
            [
                (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * inv_det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * inv_det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * inv_det,
            ],
            [
                (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * inv_det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * inv_det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * inv_det,
            ],
        ])
    }
}

/// Match the profile's TRC against the named TFs prism advertises.
/// Tolerance is half an 8-bit step (max deviation over a dense sample),
/// so tagging instead of converting can never shift a displayed 8-bit
/// value by more than the quantization it already carries.
fn classify_trc(p: &ColorProfile) -> Option<Tf> {
    let (r, g, b) = (
        p.red_trc.as_ref()?,
        p.green_trc.as_ref()?,
        p.blue_trc.as_ref()?,
    );

    const TOL: f32 = 0.5 / 255.0;
    type Eotf = fn(f32) -> f32;
    let candidates: [(Tf, Eotf); 4] = [
        (Tf::Srgb, srgb_eotf),
        (Tf::Gamma22, |x| x.powf(2.2)),
        (Tf::Bt1886, |x| x.powf(2.4)),
        (Tf::Linear, |x| x),
    ];
    'cand: for (tf, eotf) in candidates {
        for curve in [r, g, b] {
            for i in 0..=128 {
                let x = i as f32 / 128.0;
                if (eval_trc(curve, x) - eotf(x)).abs() > TOL {
                    continue 'cand;
                }
            }
        }
        return Some(tf);
    }
    None
}

fn srgb_eotf(x: f32) -> f32 {
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// Evaluate an ICC tone curve (encoded → linear) at `x` ∈ [0, 1].
fn eval_trc(curve: &ToneReprCurve, x: f32) -> f32 {
    match curve {
        ToneReprCurve::Lut(lut) => match lut.len() {
            // Empty curveType means identity; single entry is a u8.8
            // fixed-point gamma exponent (ICC curv special cases).
            0 => x,
            1 => x.powf(lut[0] as f32 / 256.0),
            n => {
                let pos = x.clamp(0.0, 1.0) * (n - 1) as f32;
                let i = (pos as usize).min(n - 2);
                let frac = pos - i as f32;
                let a = lut[i] as f32 / 65535.0;
                let b = lut[i + 1] as f32 / 65535.0;
                a + (b - a) * frac
            }
        },
        ToneReprCurve::Parametric(params) => eval_parametric(params, x),
    }
}

/// ICC `parametricCurveType` (the five `para` function types, keyed by
/// parameter count).
fn eval_parametric(p: &[f32], x: f32) -> f32 {
    let x = x.clamp(0.0, 1.0);
    match *p {
        [g] => x.powf(g),
        [g, a, b] => {
            if x >= -b / a {
                (a * x + b).powf(g)
            } else {
                0.0
            }
        }
        [g, a, b, c] => {
            if x >= -b / a {
                (a * x + b).powf(g) + c
            } else {
                c
            }
        }
        [g, a, b, c, d] => {
            if x >= d {
                (a * x + b).powf(g)
            } else {
                c * x
            }
        }
        [g, a, b, c, d, e, f] => {
            if x >= d {
                (a * x + b).powf(g) + e
            } else {
                c * x + f
            }
        }
        _ => x,
    }
}

/// Run pixels through a moxcms f32 transform (straight alpha preserved).
fn convert(src: &ColorProfile, dst: &ColorProfile, pixels: RawPixels) -> Result<RawPixels> {
    let input: Vec<f32> = match &pixels {
        RawPixels::Rgba8(d) => d.iter().map(|&v| v as f32 / 255.0).collect(),
        RawPixels::Rgba16(d) => d.iter().map(|&v| v as f32 / 65535.0).collect(),
        RawPixels::RgbaF32(d) => d.clone(),
    };
    let transform = src
        .create_transform_f32(Layout::Rgba, dst, Layout::Rgba, TransformOptions::default())
        .map_err(|e| anyhow::anyhow!("creating ICC transform: {e:?}"))?;
    let mut output = vec![0f32; input.len()];
    transform
        .transform(&input, &mut output)
        .map_err(|e| anyhow::anyhow!("applying ICC transform: {e:?}"))?;
    if output.len() != input.len() {
        bail!("ICC transform changed pixel count");
    }
    Ok(RawPixels::RgbaF32(output))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn srgb_profile_classifies_as_srgb_tagging_only() {
        let p = ColorProfile::new_srgb();
        assert_eq!(classify_trc(&p), Some(Tf::Srgb));
        let prim = PrimaryVolume::Custom(profile_chromaticities(&p)).snap_to_named(5e-3);
        assert_eq!(prim, PrimaryVolume::Srgb);
    }

    #[test]
    fn display_p3_profile_snaps_to_named_p3() {
        let p = ColorProfile::new_display_p3();
        assert_eq!(classify_trc(&p), Some(Tf::Srgb));
        let prim = PrimaryVolume::Custom(profile_chromaticities(&p)).snap_to_named(5e-3);
        assert_eq!(prim, PrimaryVolume::DisplayP3);
    }

    #[test]
    fn adobe_rgb_is_gamma22_with_custom_primaries() {
        let p = ColorProfile::new_adobe_rgb();
        // Adobe gamma is 563/256 ≈ 2.19921875 — inside the half-8-bit-step
        // tolerance of pure 2.2.
        assert_eq!(classify_trc(&p), Some(Tf::Gamma22));
        let prim = PrimaryVolume::Custom(profile_chromaticities(&p)).snap_to_named(5e-3);
        // Adobe's green primary (0.21, 0.71) must NOT snap to anything.
        assert!(matches!(prim, PrimaryVolume::Custom(_)), "got {prim:?}");
    }

    #[test]
    fn bt2020_profile_snaps_to_named() {
        let p = ColorProfile::new_bt2020();
        let prim = PrimaryVolume::Custom(profile_chromaticities(&p)).snap_to_named(5e-3);
        assert_eq!(prim, PrimaryVolume::Bt2020);
    }

    #[test]
    fn parametric_curves_evaluate() {
        // Pure gamma.
        assert!((eval_parametric(&[2.2], 0.5) - 0.5f32.powf(2.2)).abs() < 1e-6);
        // sRGB as ICC type 3 (g, a, b, c, d).
        let srgb_para = [2.4, 1.0 / 1.055, 0.055 / 1.055, 1.0 / 12.92, 0.04045];
        for i in 0..=64 {
            let x = i as f32 / 64.0;
            assert!(
                (eval_parametric(&srgb_para, x) - srgb_eotf(x)).abs() < 1e-4,
                "mismatch at {x}"
            );
        }
    }
}
