//! Internal color description types — the renderer-/protocol-independent
//! middle ground between image metadata (cICP chunks, ICC profiles, format
//! conventions) and the `wp_color_management_v1` parametric description we
//! hand to the compositor.
//!
//! The vocabulary deliberately mirrors what prism advertises (see
//! `prism-protocols/src/color_management.rs`): named TFs {sRGB, gamma 2.2,
//! BT.1886, PQ, extended linear}, named primaries {sRGB, Display-P3,
//! BT.2020} plus custom chromaticities, and optional luminances.

/// CIE 1931 xy chromaticity coordinates for an RGB primary set + white point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Chromaticities {
    pub r: (f64, f64),
    pub g: (f64, f64),
    pub b: (f64, f64),
    pub w: (f64, f64),
}

/// D65 white point (used by all primary sets we name).
pub const D65: (f64, f64) = (0.3127, 0.3290);

pub const SRGB_CHROMA: Chromaticities = Chromaticities {
    r: (0.640, 0.330),
    g: (0.300, 0.600),
    b: (0.150, 0.060),
    w: D65,
};

/// Display-P3 = DCI-P3 primaries with D65 white.
pub const DISPLAY_P3_CHROMA: Chromaticities = Chromaticities {
    r: (0.680, 0.320),
    g: (0.265, 0.690),
    b: (0.150, 0.060),
    w: D65,
};

pub const BT2020_CHROMA: Chromaticities = Chromaticities {
    r: (0.708, 0.292),
    g: (0.170, 0.797),
    b: (0.131, 0.046),
    w: D65,
};

impl Chromaticities {
    /// True if all eight coordinates are within `tol` of `other`'s.
    pub fn approx_eq(&self, other: &Chromaticities, tol: f64) -> bool {
        let pairs = [
            (self.r, other.r),
            (self.g, other.g),
            (self.b, other.b),
            (self.w, other.w),
        ];
        pairs
            .iter()
            .all(|(a, b)| (a.0 - b.0).abs() <= tol && (a.1 - b.1).abs() <= tol)
    }
}

/// Transfer function of the encoded pixel data. Restricted to what prism
/// advertises via `supported_tf_named`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tf {
    /// The piecewise sRGB curve (IEC 61966-2-1).
    Srgb,
    /// Pure power 2.2.
    Gamma22,
    /// BT.1886 (pure power 2.4 against the protocol's reference display).
    Bt1886,
    /// SMPTE ST 2084 perceptual quantizer.
    Pq,
    /// Extended linear (scene/display-linear values, 1.0 = reference white
    /// unless luminances say otherwise; negative/extended values allowed).
    Linear,
}

/// Primary color volume: a named set the protocol knows, or custom
/// chromaticities (prism advertises `set_primaries`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PrimaryVolume {
    Srgb,
    DisplayP3,
    Bt2020,
    Custom(Chromaticities),
}

impl PrimaryVolume {
    /// Collapse custom chromaticities onto a named set when they match
    /// within `tol` (named sets travel better and hit compositor fast paths).
    pub fn snap_to_named(self, tol: f64) -> PrimaryVolume {
        let PrimaryVolume::Custom(c) = self else {
            return self;
        };
        for (named, reference) in [
            (PrimaryVolume::Srgb, &SRGB_CHROMA),
            (PrimaryVolume::DisplayP3, &DISPLAY_P3_CHROMA),
            (PrimaryVolume::Bt2020, &BT2020_CHROMA),
        ] {
            if c.approx_eq(reference, tol) {
                return named;
            }
        }
        self
    }
}

/// Primary color volume luminance + reference white, in cd/m².
/// Maps to `set_luminances`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Luminances {
    pub min: f64,
    pub max: f64,
    pub reference: f64,
}

/// A complete description of how pixel values encode color — everything the
/// compositor needs to decode the buffer correctly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorEncoding {
    pub tf: Tf,
    pub primaries: PrimaryVolume,
    /// `None` = protocol defaults for the TF.
    pub luminances: Option<Luminances>,
}

impl ColorEncoding {
    /// What an untagged image is assumed to be.
    pub const SRGB: ColorEncoding = ColorEncoding {
        tf: Tf::Srgb,
        primaries: PrimaryVolume::Srgb,
        luminances: None,
    };
}

impl Tf {
    /// Decode an electrical value to linear light. Only defined for the
    /// display-referred TFs (+identity); PQ needs luminance context and is
    /// never converted client-side.
    pub fn eotf(self, x: f32) -> f32 {
        match self {
            Tf::Srgb => {
                if x <= 0.04045 {
                    x / 12.92
                } else {
                    ((x + 0.055) / 1.055).powf(2.4)
                }
            }
            Tf::Gamma22 => x.powf(2.2),
            Tf::Bt1886 => x.powf(2.4),
            Tf::Linear => x,
            Tf::Pq => unreachable!("PQ is never converted client-side"),
        }
    }

    /// Encode linear light to an electrical value (inverse of [`Self::eotf`]).
    pub fn oetf(self, x: f32) -> f32 {
        match self {
            Tf::Srgb => {
                if x <= 0.0031308 {
                    x * 12.92
                } else {
                    1.055 * x.powf(1.0 / 2.4) - 0.055
                }
            }
            Tf::Gamma22 => x.powf(1.0 / 2.2),
            Tf::Bt1886 => x.powf(1.0 / 2.4),
            Tf::Linear => x,
            Tf::Pq => unreachable!("PQ is never converted client-side"),
        }
    }
}

/// ST 2084 PQ inverse EOTF. `y` is display luminance normalized to 10000
/// cd/m² (i.e. nits/10000), result is the PQ electrical signal in [0, 1].
/// Used to repack linear HDR content into integer buffers on compositors
/// without fp16 shm — PQ is perceptually transparent at ≥12 bits.
pub fn pq_oetf(y: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;
    let y = y.clamp(0.0, 1.0);
    let ym = y.powf(M1);
    ((C1 + C2 * ym) / (1.0 + C3 * ym)).powf(M2)
}

/// User-requested luminance shaping for HDR content (some HDR wallpapers
/// declare absurd peaks). Values are target nits; applied to linear and PQ
/// sources in absolute luminance, before anything else sees the pixels.
///
/// The stages compose in order: `scale_max` normalizes the whole image so
/// its measured peak lands at most there (preserving highlight structure),
/// `tone_map` compresses what remains above the target through the
/// BT.2390 EETF (knee + roll-off, hue-preserving), and `cap` hard-clips
/// anything still left. The "color is sane but the white peaks are crazy"
/// recipe is a generous scale plus a tight cap; the automatic remaster is
/// `tone_map` alone.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LuminanceControl {
    /// Scale linearly so the content peak lands at most here (no-op when
    /// already below).
    pub scale_max: Option<f64>,
    /// Compress to this display peak via the BT.2390 EETF (resolved nits;
    /// `--tone-map auto` resolves from the output's preferred description
    /// before this struct is built).
    pub tone_map: Option<f64>,
    /// Hard-clip channels above this many nits (last).
    pub cap: Option<f64>,
}

impl LuminanceControl {
    pub fn is_empty(&self) -> bool {
        self.scale_max.is_none() && self.tone_map.is_none() && self.cap.is_none()
    }

    /// Stable hash key (f64 isn't Eq) for image deduplication. Zero bits
    /// can't collide with a real value — nits are validated positive.
    pub fn key(&self) -> (u64, u64, u64) {
        (
            self.scale_max.map_or(0, f64::to_bits),
            self.tone_map.map_or(0, f64::to_bits),
            self.cap.map_or(0, f64::to_bits),
        )
    }
}

/// BT.2390 EETF: map luminance mastered up to `src_peak` nits onto a
/// display peaking at `target` nits. Identity below the knee
/// (KS = 1.5·maxLum − 0.5 in normalized PQ), Hermite-spline roll-off
/// above it. Returns a nits → nits function; identity when
/// `target ≥ src_peak`.
pub fn bt2390_eetf(src_peak: f32, target: f32) -> impl Fn(f32) -> f32 {
    let src_max_pq = pq_oetf(src_peak / 10000.0);
    let max_lum = (pq_oetf(target / 10000.0) / src_max_pq).min(1.0);
    let ks = 1.5 * max_lum - 0.5;
    let noop = target >= src_peak;
    move |nits: f32| {
        if noop {
            return nits;
        }
        let e1 = pq_oetf(nits.max(0.0) / 10000.0) / src_max_pq;
        let e2 = if e1 < ks {
            e1
        } else {
            // Hermite spline P(E1) per BT.2390-8 §5.4.1.
            let t = (e1 - ks) / (1.0 - ks);
            let (t2, t3) = (t * t, t * t * t);
            (2.0 * t3 - 3.0 * t2 + 1.0) * ks
                + (t3 - 2.0 * t2 + t) * (1.0 - ks)
                + (-2.0 * t3 + 3.0 * t2) * max_lum
        };
        pq_eotf(e2 * src_max_pq) * 10000.0
    }
}

/// ST 2084 PQ EOTF (inverse of [`pq_oetf`]).
pub fn pq_eotf(e: f32) -> f32 {
    const M1: f32 = 2610.0 / 16384.0;
    const M2: f32 = 2523.0 / 4096.0 * 128.0;
    const C1: f32 = 3424.0 / 4096.0;
    const C2: f32 = 2413.0 / 4096.0 * 32.0;
    const C3: f32 = 2392.0 / 4096.0 * 32.0;
    let e = e.clamp(0.0, 1.0);
    let em = e.powf(1.0 / M2);
    ((em - C1).max(0.0) / (C2 - C3 * em)).powf(1.0 / M1)
}

/// Map CICP code points (H.273) to a [`ColorEncoding`], for sources that
/// carry them (PNG `cICP`, JXL, ICC v4.4 `cicp` tag). Returns `None` when
/// the code points name something we can't express losslessly (we'd rather
/// fall back to the ICC/assumed path than silently misrender).
pub fn encoding_from_cicp(primaries: u8, tf: u8, full_range: bool) -> Option<ColorEncoding> {
    // Wallpaper buffers are RGB; limited-range RGB is rare and we don't
    // rescale for it. Reject so the caller falls back.
    if !full_range {
        return None;
    }
    let primaries = match primaries {
        1 => PrimaryVolume::Srgb,
        9 => PrimaryVolume::Bt2020,
        12 => PrimaryVolume::DisplayP3,
        _ => return None,
    };
    let tf = match tf {
        // 13 = sRGB; 1/6/14/15 = BT.709/601/2020 camera OETF, which media
        // convention displays through BT.1886.
        13 => Tf::Srgb,
        1 | 6 | 14 | 15 => Tf::Bt1886,
        4 => Tf::Gamma22,
        8 => Tf::Linear,
        16 => Tf::Pq,
        _ => return None,
    };
    Some(ColorEncoding {
        tf,
        primaries,
        luminances: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cicp_maps_the_common_cases() {
        // sRGB still image.
        let e = encoding_from_cicp(1, 13, true).unwrap();
        assert_eq!((e.tf, e.primaries), (Tf::Srgb, PrimaryVolume::Srgb));
        // HDR10 still.
        let e = encoding_from_cicp(9, 16, true).unwrap();
        assert_eq!((e.tf, e.primaries), (Tf::Pq, PrimaryVolume::Bt2020));
        // P3 + sRGB TF.
        let e = encoding_from_cicp(12, 13, true).unwrap();
        assert_eq!(e.primaries, PrimaryVolume::DisplayP3);
        // Narrow range refused.
        assert!(encoding_from_cicp(1, 13, false).is_none());
        // Unknown primaries refused.
        assert!(encoding_from_cicp(22, 13, true).is_none());
    }

    #[test]
    fn snapping_tolerates_quantized_chromaticities() {
        // Slightly-off sRGB (as you'd get from ICC s15Fixed16 colorants).
        let c = Chromaticities {
            r: (0.6401, 0.3299),
            g: (0.3001, 0.6001),
            b: (0.1499, 0.0601),
            w: (0.3128, 0.3291),
        };
        assert_eq!(
            PrimaryVolume::Custom(c).snap_to_named(2e-3),
            PrimaryVolume::Srgb
        );
    }
}
