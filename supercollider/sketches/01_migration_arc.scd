/*
================================================================================
SONIFICACIO OCEANICA — Sketch 01: The Migration Arc
================================================================================

The Diel Vertical Migration (DVM) is the largest synchronized animal movement
on Earth. Every dusk, billions of mesopelagic organisms ascend from their deep
refuge (~550m) to feed near the surface (~180m). Every dawn, they descend back
into the dark. This sketch makes that movement audible as a single FM voice
whose pitch traces the DVM depth over 24 hours — extracted from the ratio of
shallow/deep acoustic energy, which tracks the spatial redistribution of
scatterers between the surface feeding zone and the deep refuge.

Everything you hear corresponds directly to a measurement from the water column.

MAPPING:
  dvm_depth      -> pitch (45-650 Hz, ~3.8 octaves: F#1 to E5, non-linear)
  entropy        -> FM timbre (pure sine <-> complex inharmonic)
  velocity       -> brightness (ascending = brighter, descending = darker)
  anomaly        -> detuning (beating/tension when depth deviates from expected)
  intensity      -> amplitude + phrase gate (recedes/silences when energy is low)
  onset          -> timbral transient (FM index spike + amplitude swell + re-attack)
  ipc            -> granularity + phrase gate (turbulent = shimmer, stable = rest)
  histogram_diff -> micro-pitch (subtle ±4 Hz inflections from zone energy shifts)
  histogram[8]   -> spectral partials (detuned pairs, spectral width ~ zone energy)
  phraseGate     -> articulation (carrier voice breathes; sub-harmonic persists)

DATA: sonification_sc_v8_20110126.json
      MALASPINA 2010, Leg 2, 38 kHz, South Atlantic (~25 S)
      January 26, 2011 — sunrise 05:45 UTC, sunset 20:15 UTC

NARRATIVE (at 1x = 8 minutes):
  0:00-1:55  NIGHT   — organisms at ~180m, feeding. Warm, clear tone.
  1:55-2:50  DAWN    — descent (~2h compressed). Pitch darkens, sinks.
  2:50-6:30  DAY     — deep refuge ~500m. Lower drone, rich harmonics.
  6:30-7:00  DUSK    — ascent (~2h). Pitch sweeps up, clarifies.
  7:00-8:00  NIGHT   — return to surface. Resolution. Full circle.

Listen for the asymmetry: the dawn descent is reluctant and slow (organisms
stop swimming, sink passively). The dusk ascent is urgent and fast (organisms
actively swim upward, burning energy, racing toward food).

HOW TO USE:
  1. Execute sections 1-5 in order (Cmd+Enter each block)
  2. Press PLAY or spacebar
  3. Listen. Just this one voice. Get to know it.

Oceanica (Alex Cabrer & Joan Sala)
CLT019 — Generalitat de Catalunya, Departament de Cultura
================================================================================
*/


// ============================================================================
// SECTION 1: SERVER BOOT
// ============================================================================

(
s.options.numOutputBusChannels = 2;
s.options.numInputBusChannels  = 0;
s.options.memSize     = 2.pow(19);
s.options.sampleRate  = 48000;

// Path: sketches/ -> supercollider/ -> project root -> output/data/
~dataPath = thisProcess.nowExecutingPath.dirname.dirname.dirname +/+ "output/data/";
~dateStr  = "20110126";

s.waitForBoot({
    "Server ready.".postln;
});
)


// ============================================================================
// SECTION 2: LOAD DATA
// ============================================================================

(
var jsonFile, jsonString;

jsonFile   = File(~dataPath ++ "sonification_sc_v8_" ++ ~dateStr ++ ".json", "r");
jsonString = jsonFile.readAllString;
jsonFile.close;
~jsonData  = jsonString.parseJSON;

// --- Only the features this voice needs ---
~times       = ~jsonData["38kHz"]["time_seconds"].asArray.collect(_.asFloat);
~depth       = ~jsonData["38kHz"]["dvm_depth_norm"].asArray.collect(_.asFloat);
~entropy     = ~jsonData["38kHz"]["acoustic_entropy"].asArray.collect(_.asFloat);
~velocity    = ~jsonData["38kHz"]["velocity_norm"].asArray.collect(_.asFloat);
~anomaly     = ~jsonData["38kHz"]["anomaly_norm"].asArray.collect(_.asFloat);
~intensity   = ~jsonData["38kHz"]["intensity_norm"].asArray.collect(_.asFloat);
~hour        = ~jsonData["38kHz"]["hour_norm"].asArray.collect(_.asFloat);
~depthRaw    = ~jsonData["38kHz"]["dvm_depth_m"].asArray.collect(_.asFloat);
~velocityRaw = ~jsonData["38kHz"]["dvm_velocity_m_h"].asArray.collect(_.asFloat);
// Keep center-of-mass depth for reference (anomaly is relative to this)
~depthCoM    = ~jsonData["38kHz"]["depth_m"].asArray.collect(_.asFloat);

// --- Histogram: 8-band depth-zone energy (locally normalized [0,1]) ---
// Zone 0: 10-50m (surface), Zone 1: 50-150m, Zone 2: 150-300m,
// Zone 3: 300-500m, Zone 4: 500-700m, Zone 5: 700-850m,
// Zone 6: 850-1000m, Zone 7: DVM corridor (280-580m)
~hist = Array.fill(8, { |b|
    ~jsonData["38kHz"]["histogram_8band_local_" ++ b.asString].asArray.collect(_.asFloat);
});

// --- Onset strength (smoothed transient events, [0,1]) ---
~onset = ~jsonData["38kHz"]["onset_strength_norm"].asArray.collect(_.asFloat);

// --- Inter-ping correlation (texture continuity, [0,1]; 0=turbulent, 1=stable) ---
~ipc = ~jsonData["38kHz"]["inter_ping_correlation"].asArray.collect(_.asFloat);

// --- Histogram short-term differential (0.5=stable, >0.5=gaining, <0.5=losing) ---
~histDiffShort = Array.fill(8, { |b|
    ~jsonData["38kHz"]["histogram_diff_short_" ++ b.asString].asArray.collect(_.asFloat);
});

// --- Auto-detected timing from extractor (hours UTC) ---
~sunriseH   = ~jsonData["info"]["sunrise_h"].asFloat   ? 5.75;
~sunsetH    = ~jsonData["info"]["sunset_h"].asFloat    ? 20.25;
~dawnEndH   = ~jsonData["info"]["dawn_end_h"].asFloat  ? 8.5;
~duskStartH = ~jsonData["info"]["dusk_start_h"].asFloat ? 19.5;
// Convert to hour fractions [0,1] for comparison with hourFrac/hourVal
~sunriseFrac   = ~sunriseH / 24;
~sunsetFrac    = ~sunsetH / 24;
~dawnEndFrac   = ~dawnEndH / 24;
~duskStartFrac = ~duskStartH / 24;

// --- Playback timing ---
~speed          = 1.0;
~totalDuration  = 480.0;   // 8 minutes at 1x
~compression    = ~times.last / ~totalDuration;
~masterVol      = 0.75;

// --- Visualization arrays (500 columns) ---
// Real-time trail: filled during playback from heardDepthSmoothed.
// Guarantees the line matches the dot exactly (same data, same pipeline).
// Raw depth preview: pre-computed from dvm_depth_m for unplayed future portion.
~vizN = 500;
~vizHeardTrail = Array.fill(~vizN, { nil });  // nil = not yet played
~vizLastIdx = 0;  // last column written by composition engine

// Raw depth preview: direct from dvm_depth_m (dim line showing unplayed data)
~vizRawPreview = Array.fill(~vizN, { |i|
    var idx = ((i / ~vizN) * (~depthRaw.size - 1)).round.asInteger.clip(0, ~depthRaw.size - 1);
    ~depthRaw[idx] ? 400;
});

// Pre-compute hour fraction per viz column (for section coloring)
~vizHourTrace = Array.fill(~vizN, { |i|
    var idx = ((i / ~vizN) * (~hour.size - 1)).round.asInteger.clip(0, ~hour.size - 1);
    ~hour[idx];
});

// --- Shared GUI state ---
~currentTime = 0.0;
~depthRawVal = 400.0;
~heardDepthM = 400.0;   // windowed depth value driving the audio (matches pitch)
~phraseActive = false;   // whether the phrase gate is open
~phraseAlpha = 1.0;      // smoothed phrase gate for visual fade (0.0-1.0)
~playing     = false;

"".postln;
"=== MIGRATION ARC DATA LOADED (v8) ===".postln;
("Points: " ++ ~times.size).postln;
("Duration: " ++ ~times.last.round(1) ++ "s (" ++ (~times.last / 3600).round(0.1) ++ "h)").postln;
("Compression: " ++ ~compression.round(1) ++ "x -> " ++ ~totalDuration ++ "s").postln;
("Sections: sunrise=" ++ ~sunriseH.round(0.1) ++ "h, dawn_end=" ++ ~dawnEndH.round(0.1) ++ "h, dusk_start=" ++ ~duskStartH.round(0.1) ++ "h, sunset=" ++ ~sunsetH.round(0.1) ++ "h").postln;
"Features loaded: dvm_depth, entropy, velocity, anomaly, intensity, hour, histogram[8], onset, ipc, histDiff[8]".postln;
)


// ============================================================================
// SECTION 3: SYNTHDEFS
// ============================================================================

(
// -------------------------------------------------------------------------
// migrationArc — FM voice tracking DVM depth (dual-zone energy ratio)
//
// Two slightly detuned FM carriers create natural richness without complexity.
//
// Night (low entropy): near-pure sine, tight unison, warm and clear.
//   Like a cello section playing the same sustained note.
// Day (high entropy): complex inharmonic FM, wider detuning, shimmering.
//   Like looking down into deep water — you sense vast depth below.
// Transitions: velocity-driven vibrato and brightness.
//   Dawn descent: slow darkening. Dusk ascent: sudden clarification.
//
// v2 additions (day-section animation, wider pitch, richer spectrum):
//   - Onset transients spike FM index (timbral flash, not just volume)
//   - IPC granularity: turbulent water column → amplitude shimmer (8-14 Hz)
//   - Micro-pitch: histogram differential → ±4 Hz inflections in static phases
//   - Histogram partials: boosted (0.22), detuned pairs for spectral width
//   - Sub-harmonic: extended threshold (300 Hz), stronger (0.3)
//   - Pitch range: 45-650 Hz (non-linear mapping, ~3.8 octaves)
//   - Dynamic range: wider floor (0.15 from 0.35)
//
// v3 additions (articulation, GUI sync):
//   - Phrase gating: carrier breathes (on/off from intensity+IPC+onset)
//   - Sub-harmonic bypasses gate (persists as depth anchor during rests)
//   - Reverb increases during rests (dissolve-into-space effect)
//   - Hysteresis on phrase gate (2s min ON, 1s min OFF, no flicker)
//
// v4 fixes (data-driven thresholds, display clarity):
//   - Phrase gate: data-driven thresholds (intensity>0.78, ipc<0.5, onset>0.15)
//     Old thresholds (0.3/0.7/0.3) never triggered rest — 100% always ON
//   - "Heard" vs "Data" display: Heard = smoothed depth matching audio pitch,
//     Data = raw DVM depth from the dual-zone log-ratio algorithm.
//
// v5 fixes (click artifact, metronomic gate, IPC split):
//   - Phrase envelope: 0.08s→0.4s attack, 1.8s→2.5s release (eliminates click)
//   - IPC split: raw feeds synth texture (ring-mod), windowed feeds gate decision
//     Raw IPC is bimodal (0.99 or 0.0) — single noisy pings retriggered gate
//   - Hysteresis: 2s/1s → 5s/3s hold times (organic phrases, not metronomic pulse)
//
// v6 fixes (visual-audio coherence, click refinement):
//   - Unified line+dot: both use heard-depth pipeline (norm→pitch→meters→smoother)
//     Eliminates confusing divergence between line (raw meters) and dot (heard depth)
//   - Reverb mix smoothed (Lag.kr 0.5s) to prevent click on ±20% wet jump
//   - Phrase envelope attack: 0.4s→0.6s for cleaner FM re-entry
//   - Dot alpha smoothed (tau 0.5s) — no visual pop on gate transitions
//
// v7 fixes (trail sync, dawn descent voice):
//   - Real-time trail: line is written from heardDepthSmoothed during playback
//     (pre-computed trace diverged due to different windowing; now line ≡ dot)
//   - Raw depth preview: dim line from dvm_depth_m shows unplayed future data
//   - Velocity phrase gate: |velocity_norm - 0.5| > 0.15 keeps voice active
//     during dawn descent and dusk ascent (column acoustics read "quiet" but
//     organisms are actively migrating — the movement itself is the signal)
//
// v8 (corridor CoM + auto-detect):
//   - DVM depth: corridor center-of-mass (50-600m) replaces dual-zone log-ratio
//     Excludes DSL (>600m) which biased all ratio approaches at 38 kHz
//   - Night/day depth auto-detected from bimodal CoM distribution
//   - Sunrise/sunset auto-detected from velocity extrema (no hardcoded hours)
//   - Section boundaries (night/dawn/day/dusk) read from JSON metadata
//   - JSON filename: sonification_sc_v8_*.json
// -------------------------------------------------------------------------
SynthDef(\migrationArc, { |out=0, freq=200, amp=0.3,
    entropy=0.65, velocity=0.5, anomaly=0.5, intensity=0.5,
    onset=0.0, ipc=1.0, microPitch=0.0,
    phraseGate=1.0,
    h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0,
    gate=1, lagTime=2.0|

    var carrierSig, subSig, env;
    var entScaled, velAbs;
    var modRatio, modIndex, modIndexLive, mod;
    var carrier1, carrier2, detune;
    var beatVoice, beatAmp;
    var subAmp;
    var vibRate, vibDepth, vibrato;
    var lpfFreq, breath, ampMod, pan;
    var histPartials, histAmp, histAmps, partialRatios, histGain, histSpread;
    var onsetEnv;
    var ipcGrain, microPitchHz, freqLive;
    var turbulence;
    var phraseEnv, finalSig;

    // Smooth all inputs — biological processes are continuous
    freq      = Lag.kr(freq, lagTime);
    amp       = Lag.kr(amp, lagTime);
    entropy   = Lag.kr(entropy, lagTime * 0.7);  // timbre responds slightly faster
    velocity  = Lag.kr(velocity, lagTime);
    anomaly   = Lag.kr(anomaly, lagTime);
    intensity = Lag.kr(intensity, lagTime);
    ipc       = Lag.kr(ipc, lagTime * 0.3);      // texture responds fast
    microPitch = Lag.kr(microPitch, lagTime * 0.4);

    // Phrase gate: asymmetric smoothing — gentle attack, slow release.
    // Attack 0.6s: carrier fades in smoothly (avoids click on complex FM signal)
    // Release 2.5s: phrase dissolves gracefully, leaving reverb tail + sub behind
    phraseEnv = Lag2UD.kr(phraseGate, 0.6, 2.5);

    // Histogram partials — slightly faster response than main timbre
    h0 = Lag.kr(h0, lagTime * 0.5);
    h1 = Lag.kr(h1, lagTime * 0.5);
    h2 = Lag.kr(h2, lagTime * 0.5);
    h3 = Lag.kr(h3, lagTime * 0.5);
    h4 = Lag.kr(h4, lagTime * 0.5);
    h5 = Lag.kr(h5, lagTime * 0.5);
    h6 = Lag.kr(h6, lagTime * 0.5);
    h7 = Lag.kr(h7, lagTime * 0.5);

    // Onset: asymmetric envelope — fast attack (50ms), slow release (~3s)
    onsetEnv = Lag2UD.kr(onset, 0.05, lagTime * 1.5);

    // Derived controls
    // Entropy: actual data range [0.60, 0.90] -> [0, 1]
    // Night ~0.65 (organized dense layers) -> entScaled ~0.17
    // Day   ~0.87 (diffuse scattered)      -> entScaled ~0.90
    entScaled = entropy.linlin(0.6, 0.9, 0, 1).clip(0, 1);

    // Velocity magnitude (for vibrato — movement in either direction)
    // velocity_norm: 0 = fast descent, 0.5 = still, 1 = fast ascent
    velAbs = (velocity - 0.5).abs * 2;

    // === MICRO-PITCH (histogram differential) ===
    // During static phases (day/night), the histogram differential provides
    // subtle pitch inflections (±4 Hz) that keep the tone alive.
    // microPitch: 0.5=stable, >0.5=shallow zones gaining, <0.5=deep zones gaining
    microPitchHz = (microPitch - 0.5) * 8.0;  // ±4 Hz range
    freqLive = freq + microPitchHz;

    // === FM MODULATOR ===
    // Night: ratio near 1 -> pure unison (the carrier and modulator reinforce)
    // Day: ratio -> sqrt(2) -> inharmonic, bell-like overtones
    // sqrt(2) is the tritone — the most ambiguous interval — for diffuse biomass
    modRatio = entScaled.linlin(0, 1, 1.002, 1.4142);
    modIndex = entScaled.linexp(0, 1, 0.08, 5.0);

    // Onset events briefly spike FM index — timbral flash, not just volume
    // Creates momentary harmonic complexity on acoustic transients
    modIndexLive = modIndex + (onsetEnv * 3.0);

    // Velocity-driven vibrato
    // Still water: near-zero vibrato. Fast transitions: shimmering instability.
    // The dusk ascent (~45 min, max velocity) will have the most vibrato —
    // organisms physically straining upward against gravity.
    vibRate  = velAbs.linlin(0, 0.7, 0.3, 5.5);
    vibDepth = velAbs.linlin(0, 0.7, 0.0004, 0.005) * freqLive;
    vibrato  = SinOsc.kr(vibRate) * vibDepth;

    mod = SinOsc.ar((freqLive + vibrato) * modRatio) * ((freqLive + vibrato) * modIndexLive);

    // === DUAL CARRIER (detuned pair) ===
    // Night: 0.15 Hz detuning -> slow, warm beating (like a cello section)
    // Day: 1.5 Hz detuning -> wider, shimmering chorus
    detune   = entScaled.linlin(0, 1, 0.15, 1.5);
    carrier1 = SinOsc.ar(freqLive + vibrato + mod);
    carrier2 = SinOsc.ar(freqLive + vibrato + detune + (mod * 0.85));
    carrierSig = (carrier1 + carrier2) * 0.5;

    // === IPC TEXTURE (turbulence granularity) ===
    // When inter-ping correlation drops (turbulent water column), the tone
    // develops subtle amplitude granularity — a shimmer that conveys instability.
    // ipc: 1.0=stable (no grain), 0.0=turbulent (maximum grain)
    // Uses ring-modulation at 8-14 Hz, depth controlled by inverse IPC.
    turbulence = (1.0 - ipc).clip(0, 1);
    ipcGrain = SinOsc.kr(
        turbulence.linlin(0, 0.6, 8, 14),  // grain rate: faster when more turbulent
        0,
        turbulence.linlin(0, 0.6, 0, 0.15),  // grain depth: 0-15% amplitude modulation
        1.0
    );
    carrierSig = carrierSig * ipcGrain;

    // === ANOMALY BEATING ===
    // When center-of-mass deviates from expected DVM curve -> tension
    // anomaly_norm: 0.5 = on track, 0 or 1 = strong deviation
    // Creates subtle interference patterns — biological unease made audible
    beatAmp   = anomaly.linlin(0.35, 0.65, 0, 0.25).clip(0, 0.25);
    beatVoice = SinOsc.ar(freqLive * (1 + anomaly.linlin(0.35, 0.65, 0, 0.006)));
    carrierSig = carrierSig + (beatVoice * beatAmp);

    // === HISTOGRAM PARTIALS ===
    // 8 partials whose amplitudes are driven by depth-zone energy.
    // Each partial is a detuned pair — spread widens with zone energy,
    // creating spectral width proportional to biomass presence.
    // The timbre literally IS the vertical distribution of biomass.
    //
    // Partial mapping (inverted: shallow zones → high partials, deep → low):
    //   Zone 0 (10-50m, surface)      → 4.0x  (bright overtone)
    //   Zone 1 (50-150m)              → 3.0x  (upper partial)
    //   Zone 2 (150-300m, DVM night)  → 2.0x  (octave above)
    //   Zone 3 (300-500m, mid)        → 1.5x  (fifth above)
    //   Zone 4 (500-700m, DVM day)    → 1.0x  (unison, reinforces fundamental)
    //   Zone 5 (700-850m)             → 0.75x (fourth below)
    //   Zone 6 (850-1000m, abyss)     → 0.5x  (octave below)
    //   Zone 7 (DVM corridor 280-580) → 1.25x (major third — warm, swells during transit)
    partialRatios = #[4.0, 3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 1.25];
    histAmps = [h0, h1, h2, h3, h4, h5, h6, h7];
    histGain = 0.22;

    histPartials = 8.collect({ |i|
        // Each zone becomes a detuned pair: spread proportional to amplitude.
        // Active zones shimmer; silent zones cost nothing.
        histSpread = histAmps[i] * 1.2;  // max ±1.2 Hz spread per partial
        (SinOsc.ar(freqLive * partialRatios[i] + histSpread)
        + SinOsc.ar(freqLive * partialRatios[i] - histSpread)) * 0.5 * histAmps[i];
    });
    histAmp = Mix.ar(histPartials) * histGain;
    carrierSig = carrierSig + histAmp;

    // === VELOCITY -> BRIGHTNESS ===
    // Ascending (velocity > 0.5): LPF opens -> brighter, like rising toward light
    // Descending (velocity < 0.5): LPF closes -> darker, sinking into the deep
    // Static (velocity ~ 0.5): moderate brightness
    lpfFreq = velocity.linexp(0.15, 0.85, freqLive * 2, freqLive * 14);
    carrierSig = LPF.ar(carrierSig, lpfFreq.clip(80, 18000));
    carrierSig = HPF.ar(carrierSig, 28);

    // === SUB-HARMONIC (bypasses phrase gate) ===
    // Octave below the fundamental — adds physical weight when deep.
    // Persists during phrase rests as a depth-awareness anchor.
    // The listener always knows "where" in the water column we are.
    subAmp = freqLive.explin(40, 300, 0.3, 0.0).clip(0, 0.3);
    subSig = SinOsc.ar(freqLive * 0.5) * subAmp;

    // === ORGANIC BREATH ===
    // Very slow amplitude modulation (~15s cycle, +/- 4%)
    // The ocean breathes. The tone is alive, not mechanical.
    breath = SinOsc.kr(0.065, 0, 0.04, 1.0);

    // === DYNAMICS ===
    // Intensity (overall acoustic energy) modulates presence
    // Lower floor (0.15) allows wider dynamic range across the 24h cycle
    ampMod = intensity.linlin(0.15, 0.85, 0.15, 1.0).clip(0.15, 1.0);
    // Onset events: amplitude swell (+35%) plus the FM index spike above
    // Together these create compound timbral transients — brighter AND louder
    ampMod = ampMod * (1.0 + (onsetEnv * 0.35));

    // === PHRASE GATING ===
    // The carrier signal (FM + histogram + anomaly) passes through the phrase gate.
    // The sub-harmonic BYPASSES the gate — it persists as a depth anchor during rests.
    //
    // Scientific meaning:
    //   Phrase ON  = acoustic activity (organisms moving, layers interacting)
    //   Phrase OFF = acoustic stillness (stable column, no events)
    //   The listener hears silence as "nothing is changing in the water column."
    //   The sub-harmonic reminds them where the organisms are hiding.
    finalSig = (carrierSig * phraseEnv) + subSig;

    // === SPATIAL ===
    // Ascending: slight right pan. Descending: slight left.
    // Subtle, not distracting — just a sense of directionality.
    pan = velocity.linlin(0.2, 0.8, -0.3, 0.3);
    finalSig = Pan2.ar(finalSig, pan);

    // === OUTPUT ===
    env = EnvGen.kr(Env.asr(4, 1, 5), gate, doneAction: 2);
    Out.ar(out, finalSig * env * amp * ampMod * breath);
}).add;


// -------------------------------------------------------------------------
// migrationReverb — Entropy-driven spatial processor
//
// The reverb size tells you about the biological state of the water column:
// Night (low entropy, organisms in dense organized layers): small room,
//   intimate, close — as if the biomass is right there beside you.
// Day (high entropy, organisms scattered diffusely): vast cathedral,
//   distant — the organisms are spread thin across hundreds of meters.
//
// The space itself is a sonification parameter.
// -------------------------------------------------------------------------
SynthDef(\migrationReverb, { |in, out=0, mix=0.3, room=0.5, damp=0.3,
    masterVol=0.75|
    var sig;
    // Smooth mix to prevent clicks when phrase gate toggles reverb wet ±20%
    mix  = Lag.kr(mix, 0.5);
    sig = In.ar(in, 2);
    sig = FreeVerb2.ar(sig[0], sig[1], mix, room, damp);
    sig = HPF.ar(sig, 35);   // prevent sub-bass mud from reverb tail
    sig = Limiter.ar(sig, 0.92);
    Out.ar(out, sig * masterVol);
}).add;

"SynthDefs loaded: migrationArc (v6), migrationReverb.".postln;
)


// ============================================================================
// SECTION 4: GUI
// ============================================================================

(
~createGUI = {
    var win;

    if(~guiWindow.notNil) { ~guiWindow.close };

    ~guiWindow = Window("Sketch 01: Migration Arc", Rect(80, 50, 740, 420))
        .background_(Color.gray(0.06))
        .front;
    win = ~guiWindow;

    // --- Header ---
    StaticText(win, Rect(15, 8, 550, 20))
        .string_("THE MIGRATION ARC  —  DVM Depth, 24 Hours")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 13));

    // --- Depth trace visualization (700 x 230) ---
    ~arcView = UserView(win, Rect(15, 32, 710, 230))
        .background_(Color.black)
        .animate_(true)
        .frameRate_(15);

    ~arcView.drawFunc = { |uv|
        var w = uv.bounds.width;
        var h = uv.bounds.height;
        var pct = (~currentTime / ~totalDuration).clip(0, 1);
        var depthMax = 1000;  // meters (y-axis: 0 at top, 1000 at bottom)

        // --- Section background bands ---
        // Subtle color fills that indicate the biological period
        ~vizN.do { |i|
            var x = (i / ~vizN) * w;
            var colW = (w / ~vizN).ceil + 1;
            var hourFrac = ~vizHourTrace[i] ? 0;
            var bgColor = case
                { hourFrac < ~sunriseFrac }  { Color.new255(10, 18, 45, 50) }   // night: deep blue
                { hourFrac < ~dawnEndFrac }  { Color.new255(90, 55, 15, 45) }   // dawn: amber
                { hourFrac < ~duskStartFrac }{ Color.new255(65, 60, 20, 30) }   // day: pale gold
                { hourFrac < ~sunsetFrac }   { Color.new255(90, 35, 55, 50) }   // dusk: warm pink
                { true }                     { Color.new255(10, 18, 45, 50) };  // night

            Pen.fillColor_(bgColor);
            Pen.fillRect(Rect(x, 0, colW, h));
        };

        // --- Depth grid ---
        [0, 200, 400, 600, 800, 1000].do { |depthM|
            var y = (depthM / depthMax) * h;
            Pen.strokeColor_(Color.gray(0.13));
            Pen.width_(0.5);
            Pen.line(0 @ y, w @ y);
            Pen.stroke;
            Pen.stringAtPoint(depthM.asString ++ "m", 3 @ (y + 2),
                Font("Helvetica", 8), Color.gray(0.28));
        };

        // --- Time labels ---
        [0, 3, 6, 9, 12, 15, 18, 21, 24].do { |hr|
            var x = (hr / 24) * w;
            Pen.stringAtPoint(hr.asString.padLeft(2, "0") ++ ":00",
                (x - 12).max(2) @ (h - 14),
                Font("Helvetica", 8), Color.gray(0.3));
        };

        // --- Section labels ---
        [
            [2.88/24,   "NIGHT"],
            [7.125/24,  "DAWN"],
            [14.0/24,   "DAY"],
            [19.875/24, "DUSK"],
            [22.125/24, "NIGHT"]
        ].do { |pair|
            var x = pair[0] * w;
            Pen.stringAtPoint(pair[1], (x - 15).max(2) @ 4,
                Font("Helvetica Bold", 9), Color.gray(0.22));
        };

        // --- DVM corridor markers (120m and 620m) ---
        [120, 620].do { |depthM|
            var y = (depthM / depthMax) * h;
            Pen.strokeColor_(Color.new255(100, 160, 220, 30));
            Pen.width_(0.5);
            Pen.line(0 @ y, w @ y);
            Pen.stroke;
        };

        // --- Raw depth preview (dim, full timeline, shows unplayed data) ---
        Pen.strokeColor_(Color.new255(100, 160, 220, 25));
        Pen.width_(1);
        Pen.moveTo(0 @ ((~vizRawPreview[0] / depthMax).clip(0, 1) * h));
        ~vizN.do { |i|
            var x = (i / (~vizN - 1)) * w;
            var y = ((~vizRawPreview[i] ? 400) / depthMax).clip(0, 1) * h;
            Pen.lineTo(x @ y);
        };
        Pen.stroke;

        // --- Real-time heard-depth trail (played portion only) ---
        // Glow layer
        if((~vizLastIdx ? 0) > 1) {
            var lastIdx = ~vizLastIdx.min(~vizN - 1);
            Pen.strokeColor_(Color.new255(100, 170, 255, 40));
            Pen.width_(4);
            Pen.moveTo(0 @ (((~vizHeardTrail[0] ? 400) / depthMax).clip(0, 1) * h));
            (lastIdx + 1).do { |i|
                var x = (i / (~vizN - 1)) * w;
                var y = ((~vizHeardTrail[i] ? 400) / depthMax).clip(0, 1) * h;
                Pen.lineTo(x @ y);
            };
            Pen.stroke;

            // Main line
            Pen.strokeColor_(Color.new255(160, 200, 240, 150));
            Pen.width_(1.2);
            Pen.moveTo(0 @ (((~vizHeardTrail[0] ? 400) / depthMax).clip(0, 1) * h));
            (lastIdx + 1).do { |i|
                var x = (i / (~vizN - 1)) * w;
                var y = ((~vizHeardTrail[i] ? 400) / depthMax).clip(0, 1) * h;
                Pen.lineTo(x @ y);
            };
            Pen.stroke;
        };

        // --- Depth marker (rides on the real-time trail) ---
        // The dot and trail are written from the same value (heardDepthSmoothed),
        // so the dot always sits exactly on the trail line at the playhead.
        if(pct > 0.001) {
            var cx = pct * w;
            var heardY = ((~heardDepthM ? 400) / depthMax).clip(0, 1) * h;
            // Smoothed alpha: fades gradually with audio envelope (no visual pop)
            var dotAlpha = ((~phraseAlpha ? 1.0) * 150 + 80).asInteger.clip(80, 230);

            // Outer glow
            Pen.fillColor_(Color.new255(255, 255, 255, (dotAlpha * 0.35).asInteger));
            Pen.fillOval(Rect(cx - 7, heardY - 7, 14, 14));
            // Inner dot
            Pen.fillColor_(Color.new255(255, 255, 255, dotAlpha));
            Pen.fillOval(Rect(cx - 3.5, heardY - 3.5, 7, 7));
        };

        // --- Green playhead ---
        Pen.strokeColor_(Color.new255(0, 220, 100, 180));
        Pen.width_(1.5);
        Pen.line((pct * w) @ 0, (pct * w) @ h);
        Pen.stroke;
    };

    // --- Info display ---
    ~infoDisplay = StaticText(win, Rect(15, 270, 430, 50))
        .string_("UTC: --:--   Depth: ---m   Velocity: --- m/h\nEntropy: ----   Anomaly: ----")
        .stringColor_(Color.new255(170, 195, 215))
        .font_(Font("Monaco", 10))
        .background_(Color.gray(0.09));

    ~sectionDisplay = StaticText(win, Rect(455, 270, 270, 50))
        .string_("---\nAwaiting playback")
        .stringColor_(Color.gray(0.4))
        .align_(\center)
        .font_(Font("Helvetica Bold", 12))
        .background_(Color.gray(0.09));

    // --- Transport ---
    ~playBtn = Button(win, Rect(15, 335, 90, 32))
        .states_([
            ["PLAY", Color.black, Color.new255(50, 220, 80)],
            ["STOP", Color.white, Color.new255(200, 50, 50)]
        ])
        .font_(Font("Helvetica Bold", 12))
        .action_({ |btn|
            if(btn.value == 1) { ~playComposition.value } { ~stopAll.value };
        });

    // --- Volume ---
    StaticText(win, Rect(125, 340, 30, 20))
        .string_("Vol")
        .stringColor_(Color.gray(0.5))
        .font_(Font("Helvetica", 10));
    Slider(win, Rect(155, 340, 110, 20))
        .value_(~masterVol)
        .background_(Color.gray(0.18))
        .knobColor_(Color.white)
        .action_({ |sl| ~masterVol = sl.value });

    // --- Speed ---
    StaticText(win, Rect(290, 340, 45, 20))
        .string_("Speed")
        .stringColor_(Color.gray(0.5))
        .font_(Font("Helvetica", 10));

    ~speedBtns = Array.new(4);
    [0.5, 1.0, 2.0, 4.0].do { |spd, i|
        var lbl = if(spd < 1) { "0.5x" } { spd.asInteger.asString ++ "x" };
        ~speedBtns = ~speedBtns.add(
            Button(win, Rect(340 + (i * 52), 335, 46, 28))
                .states_([
                    [lbl, Color.white, Color.gray(0.22)],
                    [lbl, Color.black, Color.yellow]
                ])
                .font_(Font("Helvetica Bold", 10))
                .value_(if(spd == 1.0) { 1 } { 0 })
                .action_({ |btn|
                    if(btn.value == 1) {
                        ~speedBtns.do { |b| if(b !== btn) { b.value_(0) } };
                        ~speed = spd;
                        ~totalDuration = ~times.last / (~times.last / 480) / ~speed;
                        ~compression = ~times.last / ~totalDuration;
                        ("Speed: " ++ lbl ++ " -> " ++ ~totalDuration.round(1) ++ "s").postln;
                    };
                })
        );
    };

    // --- REC button ---
    Button(win, Rect(560, 335, 60, 28))
        .states_([["REC", Color.white, Color.gray(0.25)]])
        .font_(Font("Helvetica Bold", 10))
        .action_({
            if(s.isRecording)
                { s.stopRecording; "Stopped recording.".postln }
                { s.record;        "Recording...".postln };
        });

    // --- Credits ---
    StaticText(win, Rect(15, 385, 710, 25))
        .string_(
            "Sonificacio Oceanica — Sketch 01: Migration Arc  |  " ++
            "MALASPINA 2010 — 38 kHz — Jan 26, 2011 — 25 S Atlantic  |  " ++
            "Oceanica (Cabrer & Sala) — CLT019"
        )
        .stringColor_(Color.gray(0.25))
        .font_(Font("Helvetica", 8));

    // --- Spacebar shortcut ---
    win.view.keyDownAction = { |view, char|
        if(char == $ ) {
            if(~playing) {
                ~stopAll.value;
                { ~playBtn.value_(0) }.defer;
            } {
                ~playComposition.value;
                { ~playBtn.value_(1) }.defer;
            };
        };
    };

    win.onClose = {
        ~stopAll.value;
        "Window closed.".postln;
    };

    "GUI created. Press PLAY or spacebar.".postln;
};

~createGUI.value;
)


// ============================================================================
// SECTION 5: COMPOSITION ENGINE
// ============================================================================

(
// --- Data access functions ---

// Windowed mean: smooths over a time window (for continuous parameters)
~getValue = { |dataArray, timeArray, compTime, compression, windowSec=2|
    var realTime, startIdx, endIdx, values;
    realTime = compTime * compression;
    startIdx = (timeArray.indexOfGreaterThan(realTime - (windowSec * compression * 0.5)) ? 0).max(0);
    endIdx   = (timeArray.indexOfGreaterThan(realTime + (windowSec * compression * 0.5)) ? (timeArray.size - 1)).min(timeArray.size - 1);
    values   = dataArray.copyRange(startIdx, endIdx);
    values   = values.reject({ |v| v.isNil or: { v.isNaN } });
    if(values.size > 0) { values.mean } { 0.5 };
};

// Nearest-sample lookup: preserves dynamics (for discrete/fast parameters)
~getRaw = { |dataArray, timeArray, compTime, compression|
    var realTime, idx;
    realTime = compTime * compression;
    idx = (timeArray.indexOfGreaterThan(realTime) ? (timeArray.size - 1)).max(0).min(dataArray.size - 1);
    dataArray[idx] ? 0;
};

// --- FX bus ---
~fxBus = Bus.audio(s, 2);
~playing = false;

// --- The composition: one voice, one story ---
~composition = Routine({
    var updateRate = 0.05;  // 50ms = 20 Hz
    var currentTime = 0;
    var migSynth, fxSynth;

    // Data values
    var depthVal, entropyVal, velocityVal, anomalyVal, intensityVal;
    var depthRawVal, velRawVal, hourVal;
    var heardDepthM;  // windowed depth in meters (matches audio pitch)
    var heardDepthSmoothed = 400;  // exponential smoother to approximate Lag.kr(freq, 2.0)
    var smoothAlpha = 1 - (-0.05 / 2.0).exp;  // dt=50ms, tau=2.0s -> ~0.0247
    var entScaled;
    // Histogram + onset + IPC + micro-pitch + adaptive window
    var h0val, h1val, h2val, h3val, h4val, h5val, h6val, h7val;
    var onsetVal, ipcVal, ipcRawVal, microPitchVal;
    var velPreview, velMag, adaptiveWindow;

    // Phrase gate state (with hysteresis)
    var phraseGateVal = 1.0;
    var phraseOnTimer = 0;    // frames since last gate-ON
    var phraseOffTimer = 0;   // frames since last gate-OFF
    var phraseRaw = 1.0;      // raw gate decision before hysteresis
    var minOnFrames, minOffFrames;

    // Section tracking
    var section = \night1, lastSection = \none;
    var sectionName = "---", bioState = "";
    var sectionHoldCount = 0, candidateSection = \none;

    "".postln;
    "=======================================================".postln;
    "   SKETCH 01: THE MIGRATION ARC".postln;
    "   The largest animal movement on Earth, in sound.".postln;
    "=======================================================".postln;
    "".postln;

    // Start reverb (tail of signal chain)
    fxSynth = Synth(\migrationReverb, [
        \in, ~fxBus, \out, 0, \masterVol, ~masterVol
    ], addAction: \addToTail);

    // Start migration voice (silent, will fade in via data)
    migSynth = Synth(\migrationArc, [
        \out, ~fxBus, \amp, 0, \gate, 1
    ]);

    0.5.wait;
    ~playing = true;

    // Phrase gate hysteresis: minimum hold times to create musical phrases.
    // Short holds (2s/1s) created metronomic 3.1s pulsing from IPC noise.
    // Longer holds produce organic phrases: 5s of voice, 3s of sub + reverb.
    // At 20 Hz update rate: 100 frames = 5s ON, 60 frames = 3s OFF.
    minOnFrames  = (5.0 / updateRate).round.asInteger;   // stay ON at least 5s
    minOffFrames = (3.0 / updateRate).round.asInteger;    // stay OFF at least 3s

    while({ currentTime < ~totalDuration and: { ~playing } }, {

        if(~playing.not) {
            updateRate.wait;
            currentTime = currentTime + updateRate;
        } {

        // ===== READ DATA =====
        // Velocity-adaptive window: fast migration → responsive, stillness → smooth
        velPreview = ~getValue.(~velocity, ~times, currentTime, ~compression, 1.5 * ~speed);
        velMag = (velPreview - 0.5).abs * 2;
        adaptiveWindow = velMag.linlin(0, 0.7, 3.0, 1.0).clip(1.0, 3.5) * ~speed;

        depthVal     = ~getValue.(~depth,     ~times, currentTime, ~compression, adaptiveWindow);
        entropyVal   = ~getValue.(~entropy,   ~times, currentTime, ~compression, adaptiveWindow);
        velocityVal  = ~getValue.(~velocity,  ~times, currentTime, ~compression, adaptiveWindow);
        anomalyVal   = ~getValue.(~anomaly,   ~times, currentTime, ~compression, adaptiveWindow);
        intensityVal = ~getValue.(~intensity, ~times, currentTime, ~compression, adaptiveWindow);
        hourVal      = ~getValue.(~hour,      ~times, currentTime, ~compression, 1.5 * ~speed);

        depthRawVal = ~getRaw.(~depthRaw,    ~times, currentTime, ~compression);
        velRawVal   = ~getRaw.(~velocityRaw, ~times, currentTime, ~compression);

        // Histogram bands (windowed mean, smooth)
        h0val = ~getValue.(~hist[0], ~times, currentTime, ~compression, 2.0 * ~speed);
        h1val = ~getValue.(~hist[1], ~times, currentTime, ~compression, 2.0 * ~speed);
        h2val = ~getValue.(~hist[2], ~times, currentTime, ~compression, 2.0 * ~speed);
        h3val = ~getValue.(~hist[3], ~times, currentTime, ~compression, 2.0 * ~speed);
        h4val = ~getValue.(~hist[4], ~times, currentTime, ~compression, 2.0 * ~speed);
        h5val = ~getValue.(~hist[5], ~times, currentTime, ~compression, 2.0 * ~speed);
        h6val = ~getValue.(~hist[6], ~times, currentTime, ~compression, 2.0 * ~speed);
        h7val = ~getValue.(~hist[7], ~times, currentTime, ~compression, 2.0 * ~speed);

        // Onset strength (nearest-sample to preserve transient spikes)
        onsetVal = ~getRaw.(~onset, ~times, currentTime, ~compression);

        // Inter-ping correlation: two reads for different purposes.
        // Raw (nearest-sample): feeds synth ring-mod grain — preserves fast texture
        // Windowed mean: feeds phrase gate — smooths bimodal spikes so individual
        // noisy pings don't retrigger the gate every 3 seconds
        ipcRawVal = ~getRaw.(~ipc, ~times, currentTime, ~compression);
        ipcVal = ~getValue.(~ipc, ~times, currentTime, ~compression, 2.0 * ~speed);

        // Micro-pitch from histogram differential: weighted sum across zones.
        // Shallow zones gaining energy (>0.5) push pitch slightly up;
        // deep zones gaining push pitch down. Creates subtle life in static phases.
        microPitchVal = 8.collect({ |b|
            var diffVal = ~getValue.(~histDiffShort[b], ~times, currentTime, ~compression, 1.5 * ~speed);
            var weight = #[0.8, 0.6, 0.4, 0.2, -0.2, -0.4, -0.6, 0.1][b];  // shallow=positive, deep=negative
            (diffVal - 0.5) * weight;
        }).sum;
        // Re-center to 0.5 and clip to [0, 1]
        microPitchVal = (microPitchVal + 0.5).clip(0, 1);

        entScaled = entropyVal.linlin(0.6, 0.9, 0, 1).clip(0, 1);

        // ===== HEARD DEPTH (for GUI sync) =====
        // "Heard depth" = the depth implied by the pitch the listener hears.
        // The pitch mapping is non-linear (linexp), so this differs from the raw
        // linear depth, especially at the extremes where linexp compresses.
        //
        // Pipeline: depthVal -> linexp(0.15,1.0,45,650) -> freq Hz
        //           freq Hz  -> explin(45,650,620,120)   -> "heard" meters
        //
        // This reveals where the pitch mapping distorts: if heard != raw,
        // the listener is perceiving a different depth than the data shows.
        heardDepthM = depthVal.linexp(0.15, 1.0, 45, 650).clip(40, 700)
            .explin(45, 650, 620, 120).clip(50, 1000);

        // Exponential smoother: approximates Lag.kr(freq, 2.0) in the synth.
        // This makes heardDepthSmoothed track what the listener actually hears,
        // including the ~2s lag that smooths pitch transitions.
        heardDepthSmoothed = heardDepthSmoothed + (smoothAlpha * (heardDepthM - heardDepthSmoothed));

        // ===== PHRASE GATE =====
        // Determines whether the carrier voice is active or resting.
        //
        // Scientific logic:
        //   Active (gate=1): intensity is notably high, OR the column is turbulent
        //                    (low IPC), OR an onset transient is firing.
        //   Rest (gate=0):   intensity is moderate-to-low AND column is stable
        //                    AND no onsets — "the water column is quiet."
        //
        // Thresholds are data-driven (day intensity: p10=0.74, p50=0.77, p90=0.80):
        //   intensity > 0.78  → above-median energy, something notable
        //   ipc < 0.5         → genuine turbulence (bimodal: mostly ~0.99 or ~0.0)
        //   onset > 0.15      → meaningful transient (p90 of day onset ~0.07)
        //   |velocity - 0.5| > 0.15 → organisms are actively migrating (dawn/dusk)
        //     velocity_norm 0.5 = stationary; deviations = ascending or descending
        //     0.15 threshold catches the gradual dawn descent that column acoustics miss
        //
        // Hysteresis prevents rapid flickering between states.
        phraseRaw = if(
            (intensityVal > 0.78) or: { ipcVal < 0.5 } or: { onsetVal > 0.15 }
            or: { (velocityVal - 0.5).abs > 0.15 }
        ) { 1.0 } { 0.0 };

        // Apply hysteresis: enforce minimum hold times
        if(phraseRaw > 0.5) {
            // Wants to be ON
            if(phraseGateVal < 0.5) {
                // Currently OFF -> switch ON only if OFF long enough
                if(phraseOffTimer >= minOffFrames) {
                    phraseGateVal = 1.0;
                    phraseOnTimer = 0;
                    phraseOffTimer = 0;
                } {
                    phraseOffTimer = phraseOffTimer + 1;
                };
            } {
                // Already ON -> stay ON, reset timer
                phraseOnTimer = phraseOnTimer + 1;
                phraseOffTimer = 0;
            };
        } {
            // Wants to be OFF
            if(phraseGateVal > 0.5) {
                // Currently ON -> switch OFF only if ON long enough
                if(phraseOnTimer >= minOnFrames) {
                    phraseGateVal = 0.0;
                    phraseOffTimer = 0;
                    phraseOnTimer = 0;
                } {
                    phraseOnTimer = phraseOnTimer + 1;
                };
            } {
                // Already OFF -> stay OFF
                phraseOffTimer = phraseOffTimer + 1;
                phraseOnTimer = 0;
            };
        };

        // ===== SECTION DETECTION =====
        candidateSection = case
            { hourVal < ~sunriseFrac }   { \night1 }
            { hourVal < ~dawnEndFrac }   { \dawn   }
            { hourVal < ~duskStartFrac } { \day    }
            { hourVal < ~sunsetFrac }    { \dusk   }
            { true }                     { \night2 };

        if(candidateSection != section) {
            sectionHoldCount = sectionHoldCount + 1;
            if(sectionHoldCount >= 3) {
                section = candidateSection;
                sectionHoldCount = 0;
            };
        } { sectionHoldCount = 0 };

        #sectionName, bioState = case
            { section == \night1 } {
                ["NIGHT",  "Surface feeding — organisms at 180-230m"]
            }
            { section == \dawn } {
                ["DAWN",   "Slow descent to depth — ~2 hours sinking"]
            }
            { section == \day } {
                ["DAY",    "Deep refuge — hiding from predators at 430-550m"]
            }
            { section == \dusk } {
                ["DUSK",   "Rapid ascent — 350 meters in ~2 hours"]
            }
            { true } {
                ["NIGHT",  "Return to surface — the cycle completes"]
            };

        // Announce section changes
        if(section != lastSection) {
            "".postln;
            (">>> " ++ sectionName).postln;
            ("    " ++ bioState).postln;
            lastSection = section;
        };

        // ===== UPDATE MIGRATION VOICE =====
        // Pitch: non-linear mapping that stretches the transition range.
        // depthVal is dvm_depth_norm: 0=deep(800m), 1=shallow(100m).
        // Actual data range: night ~0.86, day ~0.37.
        //
        // Piecewise exponential: the mid-range (0.35-0.75) where dawn/dusk
        // transitions happen gets expanded perceptually, while the endpoints
        // (where pitch is near-static) are compressed.
        //   depthVal 0.20 ->  45 Hz (F#1)  — lowest usable fundamental
        //   depthVal 0.37 ->  90 Hz (F#2)  — day refuge
        //   depthVal 0.55 -> 200 Hz (G3)   — mid-transition
        //   depthVal 0.75 -> 380 Hz (F#4)  — approaching surface
        //   depthVal 0.90 -> 560 Hz (C#5)  — night feeding
        //   depthVal 1.00 -> 650 Hz (E5)   — extreme shallow
        // Total usable range: ~2.6 octaves (F#2 to C#5), with 3.8 octaves available.
        migSynth.set(
            \freq,      depthVal.linexp(0.15, 1.0, 45, 650).clip(40, 700),
            \amp,       ~masterVol * 0.5,
            \entropy,   entropyVal,
            \velocity,  velocityVal,
            \anomaly,   anomalyVal,
            \intensity, intensityVal,
            \onset,     onsetVal,
            \ipc,       ipcRawVal,  // raw IPC for texture granularity
            \microPitch, microPitchVal,
            \phraseGate, phraseGateVal,
            \h0, h0val, \h1, h1val, \h2, h2val, \h3, h3val,
            \h4, h4val, \h5, h5val, \h6, h6val, \h7, h7val
        );

        // ===== UPDATE REVERB =====
        // Night: intimate room. Day: vast cathedral.
        // During phrase rests: reverb mix increases — sound dissolves into space.
        // This creates a natural "the signal diffuses" effect as the carrier fades.
        fxSynth.set(
            \mix,       entScaled.linlin(0, 1, 0.12, 0.45)
                        + ((1.0 - phraseGateVal) * 0.2),  // +20% wet during rests
            \room,      entScaled.linlin(0, 1, 0.2, 0.82),
            \damp,      entScaled.linlin(0, 1, 0.55, 0.18),
            \masterVol, ~masterVol
        );

        // ===== GUI UPDATE =====
        ~currentTime = currentTime;
        ~depthRawVal = depthRawVal;
        ~heardDepthM = heardDepthSmoothed;  // smoothed to match Lag.kr in synth
        ~phraseActive = phraseGateVal > 0.5;

        // Write heard depth into the real-time trail (guarantees line = dot)
        {
            var vizIdx = ((currentTime / ~totalDuration) * ~vizN).floor.asInteger.clip(0, ~vizN - 1);
            ~vizHeardTrail[vizIdx] = heardDepthSmoothed;
            ~vizLastIdx = vizIdx;
        }.value;
        // Smooth visual alpha to match audio envelope (no instant pop)
        // tau=0.5s for both attack and release — fast enough to track, smooth enough to not pop
        ~phraseAlpha = ~phraseAlpha + ((1 - (-0.05 / 0.5).exp) * (phraseGateVal - ~phraseAlpha));

        {
            var rHour = (hourVal * 24).floor.asInteger;
            var rMin  = ((hourVal * 24 * 60) % 60).floor.asInteger;
            var velDir = if(velRawVal > 0) { " asc" } { " dsc" };
            var phraseStr = if(phraseGateVal > 0.5) { "" } { " [rest]" };

            if(~infoDisplay.notNil and: { ~infoDisplay.isClosed.not }) {
                ~infoDisplay.string_(
                    "UTC: " ++ rHour.asString.padLeft(2, "0") ++ ":" ++
                    rMin.asString.padLeft(2, "0") ++
                    "   Heard: " ++ heardDepthSmoothed.round(1) ++ "m" ++
                    "   Data: " ++ depthRawVal.round(1) ++ "m" ++
                    phraseStr ++
                    "\nEntropy: " ++ entropyVal.round(0.001) ++
                    "   Intensity: " ++ intensityVal.round(0.01) ++
                    "   IPC: " ++ ipcVal.round(0.01) ++
                    "   Gate: " ++ if(phraseGateVal > 0.5) { "ON" } { "off" }
                );
            };

            if(~sectionDisplay.notNil and: { ~sectionDisplay.isClosed.not }) {
                ~sectionDisplay.string_(sectionName ++ "\n" ++ bioState);
                ~sectionDisplay.stringColor_(case
                    { section == \night1 or: { section == \night2 } } {
                        Color.new255(80, 170, 255)
                    }
                    { section == \dawn } { Color.new255(255, 155, 50) }
                    { section == \day  } { Color.new255(255, 225, 90) }
                    { section == \dusk } { Color.new255(255, 95, 155) }
                    { true } { Color.white }
                );
            };
        }.defer;

        }; // end ~playing guard

        updateRate.wait;
        currentTime = currentTime + updateRate;
    });

    // === FADE OUT ===
    if(~playing) {
        "".postln;
        "The cycle is complete.".postln;
        migSynth.set(\gate, 0);
    };
    ~playing = false;

    5.wait;  // let the envelope release and reverb tail decay
    fxSynth.free;

    { if(~sectionDisplay.notNil and: { ~sectionDisplay.isClosed.not }) {
        ~sectionDisplay.string_("COMPLETE\n24-hour cycle finished");
        ~sectionDisplay.stringColor_(Color.white);
    } }.defer;

    "Done.".postln;
});

// --- Transport controls ---
~playComposition = {
    ~playing = false;
    // Reset real-time trail for fresh playback
    ~vizN.do { |i| ~vizHeardTrail[i] = nil };
    ~vizLastIdx = 0;
    ~composition.reset;
    ~composition.play;
    "Playing migration arc...".postln;
};

~stopAll = {
    ~playing = false;
    ~composition.stop;
    s.freeAll;
    "Stopped.".postln;
};

"Composition engine ready.".postln;
)


// ============================================================================
// END
// ============================================================================

/*
WHAT TO LISTEN FOR:

Night (0:00-1:55):
  A warm, clear tone around C#5 (560 Hz). Near-pure sine with gentle
  chorus from the detuned carrier pair. Intimate reverb. The organisms
  are packed in dense layers near the surface (~180m) — organized,
  purposeful. Entropy is low, so the FM index is minimal. Histogram
  partials shimmer faintly — biomass concentrated in the upper zones.
  The phrase gate is mostly open — the voice is sustained and present.

Dawn descent (1:55-2:50):
  The pitch drops dramatically — about 2.6 octaves over a minute. As
  entropy rises, the timbre complexifies: new overtones bloom, the
  sound becomes richer and more ambiguous. The reverb opens up. The
  organisms are dispersing, sinking, spreading thin. The LPF closes
  as they descend — the sound darkens. Listen for timbral flashes on
  onset events (brief FM index spikes) as layers reorganize. The voice
  may briefly rest during calm moments in the transition — only the
  sub-harmonic and reverb tail persist, reminding you of the depth.

Day (2:50-6:30):
  A deep drone around F#2 (90 Hz) with strong sub-harmonic weight.
  Complex FM harmonics shift slowly. Vast reverb. The voice BREATHES:
  during quiet stretches (high IPC, low intensity), the carrier fades
  to silence, leaving only the sub-harmonic octave below — you feel the
  ocean's weight without hearing the voice. When intensity rises or
  the column becomes turbulent (IPC drops), the voice re-enters with a
  timbral bloom. Listen for:
    - Micro-pitch inflections (±4 Hz) from histogram differential
    - Amplitude shimmer from IPC drops (turbulent column events)
    - FM index flashes on onset transients (layer reorganizations)
    - The reverb swelling during rests (sound dissolves into space)
  Silence means "nothing is changing." Sound means "something is happening."

Dusk ascent (6:30-7:00):
  The pitch sweeps up as organisms race toward the surface. The timbre
  clarifies — complex FM resolves toward pure sine. Vibrato intensifies.
  The LPF opens wide — brightness floods in. Reverb contracts. The
  sub-harmonic fades as pitch rises above 300 Hz. The voice is fully
  active — the dusk ascent is the most energetic part of the cycle.

Night return (7:00-8:00):
  Resolution. The warm, clear tone returns. The cycle is complete.
  Tomorrow it happens again.

GUI NOTES:
  The line and dot both show "heard depth" — the depth implied by the audio
  pitch, smoothed to match the synth's Lag.kr. The dot rides on the line at
  the playhead position, fading gradually with the phrase envelope.

  "Heard" = smoothed depth matching the audio pitch (120-620m range).
  "Data" = raw DVM depth from the dual-zone log-ratio algorithm (meters).
  "[rest]" = phrase gate is closed, carrier is silent, sub persists.

  Rest periods occur when the water column is quiet: intensity below the
  day median (~0.78), IPC stable (>0.5), and no onset transients (<0.15).

NEXT SKETCHES:
  02 — Layer events (Klank bells on scattering layer birth/death)
  03 — Zone energy (8-band BPF filter bank from depth histogram)
  04 — Onset pulses (high-percussion on acoustic transients)
  05 — Composed (all voices combined with intentional arrangement)
*/
