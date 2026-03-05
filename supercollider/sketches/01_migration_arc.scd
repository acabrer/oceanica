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
  dvm_depth -> pitch (80-500 Hz, ~2.6 octaves: E2 to B4)
  entropy   -> FM timbre (pure sine <-> complex inharmonic)
  velocity  -> brightness (ascending = brighter, descending = darker)
  anomaly   -> detuning (beating/tension when depth deviates from expected)
  intensity -> amplitude (recedes when acoustic energy is low)

DATA: sonification_sc_v7_20110126.json
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

jsonFile   = File(~dataPath ++ "sonification_sc_v7_" ++ ~dateStr ++ ".json", "r");
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

// --- Playback timing ---
~speed          = 1.0;
~totalDuration  = 480.0;   // 8 minutes at 1x
~compression    = ~times.last / ~totalDuration;
~masterVol      = 0.75;

// --- Pre-compute depth trace for visualization (500 points, bin-averaged) ---
~vizN = 500;
~vizDepthTrace = Array.fill(~vizN, { |i|
    var idxLo = ((i / ~vizN) * (~depthRaw.size - 1)).round.asInteger.max(0);
    var idxHi = (((i + 1) / ~vizN) * (~depthRaw.size - 1)).round.asInteger.min(~depthRaw.size - 1);
    var slice = ~depthRaw.copyRange(idxLo, idxHi);
    slice = slice.reject({ |v| v.isNil or: { v.isNaN } });
    if(slice.size > 0) { slice.mean } { 400 };
});

// Pre-compute hour fraction per viz column (for section coloring)
~vizHourTrace = Array.fill(~vizN, { |i|
    var idx = ((i / ~vizN) * (~hour.size - 1)).round.asInteger.clip(0, ~hour.size - 1);
    ~hour[idx];
});

// --- Shared GUI state ---
~currentTime = 0.0;
~depthRawVal = 400.0;
~playing     = false;

"".postln;
"=== MIGRATION ARC DATA LOADED ===".postln;
("Points: " ++ ~times.size).postln;
("Duration: " ++ ~times.last.round(1) ++ "s (" ++ (~times.last / 3600).round(0.1) ++ "h)").postln;
("Compression: " ++ ~compression.round(1) ++ "x -> " ++ ~totalDuration ++ "s").postln;
"Features loaded: dvm_depth, entropy, velocity, anomaly, intensity, hour, histogram[8], onset".postln;
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
// The sub-harmonic reinforces the fundamental when deep — you feel
// the ocean's weight. The anomaly beating creates interference when
// depth deviates from expected — biological tension made audible.
// -------------------------------------------------------------------------
SynthDef(\migrationArc, { |out=0, freq=200, amp=0.3,
    entropy=0.65, velocity=0.5, anomaly=0.5, intensity=0.5,
    onset=0.0,
    h0=0.0, h1=0.0, h2=0.0, h3=0.0, h4=0.0, h5=0.0, h6=0.0, h7=0.0,
    gate=1, lagTime=2.0|

    var sig, env;
    var entScaled, velAbs;
    var modRatio, modIndex, mod;
    var carrier1, carrier2, detune;
    var beatVoice, beatAmp;
    var sub, subAmp;
    var vibRate, vibDepth, vibrato;
    var lpfFreq, breath, ampMod, pan;
    var histPartials, histAmp, histAmps, partialRatios, histGain;
    var onsetEnv;

    // Smooth all inputs — biological processes are continuous
    freq      = Lag.kr(freq, lagTime);
    amp       = Lag.kr(amp, lagTime);
    entropy   = Lag.kr(entropy, lagTime * 0.7);  // timbre responds slightly faster
    velocity  = Lag.kr(velocity, lagTime);
    anomaly   = Lag.kr(anomaly, lagTime);
    intensity = Lag.kr(intensity, lagTime);

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

    // === FM MODULATOR ===
    // Night: ratio near 1 -> pure unison (the carrier and modulator reinforce)
    // Day: ratio -> sqrt(2) -> inharmonic, bell-like overtones
    // sqrt(2) is the tritone — the most ambiguous interval — for diffuse biomass
    modRatio = entScaled.linlin(0, 1, 1.002, 1.4142);
    modIndex = entScaled.linexp(0, 1, 0.08, 5.0);

    // Velocity-driven vibrato
    // Still water: near-zero vibrato. Fast transitions: shimmering instability.
    // The dusk ascent (~45 min, max velocity) will have the most vibrato —
    // organisms physically straining upward against gravity.
    vibRate  = velAbs.linlin(0, 0.7, 0.3, 5.5);
    vibDepth = velAbs.linlin(0, 0.7, 0.0004, 0.005) * freq;
    vibrato  = SinOsc.kr(vibRate) * vibDepth;

    mod = SinOsc.ar((freq + vibrato) * modRatio) * ((freq + vibrato) * modIndex);

    // === DUAL CARRIER (detuned pair) ===
    // Night: 0.15 Hz detuning -> slow, warm beating (like a cello section)
    // Day: 1.5 Hz detuning -> wider, shimmering chorus
    detune   = entScaled.linlin(0, 1, 0.15, 1.5);
    carrier1 = SinOsc.ar(freq + vibrato + mod);
    carrier2 = SinOsc.ar(freq + vibrato + detune + (mod * 0.85));
    sig = (carrier1 + carrier2) * 0.5;

    // === ANOMALY BEATING ===
    // When center-of-mass deviates from expected DVM curve -> tension
    // anomaly_norm: 0.5 = on track, 0 or 1 = strong deviation
    // Creates subtle interference patterns — biological unease made audible
    beatAmp   = anomaly.linlin(0.35, 0.65, 0, 0.25).clip(0, 0.25);
    beatVoice = SinOsc.ar(freq * (1 + anomaly.linlin(0.35, 0.65, 0, 0.006)));
    sig = sig + (beatVoice * beatAmp);

    // === SUB-HARMONIC ===
    // Octave below the fundamental — adds physical weight when deep
    // Below ~180 Hz: organisms are at depth, you feel the ocean's mass
    // Above ~180 Hz: organisms are shallow, tone is lighter and clearer
    subAmp = freq.explin(55, 180, 0.2, 0.0).clip(0, 0.2);
    sub    = SinOsc.ar(freq * 0.5) * subAmp;
    sig = sig + sub;

    // === HISTOGRAM PARTIALS ===
    // 8 sine partials whose amplitudes are driven by depth-zone energy.
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
    histGain = 0.12;

    histPartials = 8.collect({ |i|
        SinOsc.ar(freq * partialRatios[i]) * histAmps[i];
    });
    histAmp = Mix.ar(histPartials) * histGain;
    sig = sig + histAmp;

    // === VELOCITY -> BRIGHTNESS ===
    // Ascending (velocity > 0.5): LPF opens -> brighter, like rising toward light
    // Descending (velocity < 0.5): LPF closes -> darker, sinking into the deep
    // Static (velocity ~ 0.5): moderate brightness
    lpfFreq = velocity.linexp(0.15, 0.85, freq * 2, freq * 14);
    sig = LPF.ar(sig, lpfFreq.clip(80, 18000));
    sig = HPF.ar(sig, 28);

    // === ORGANIC BREATH ===
    // Very slow amplitude modulation (~15s cycle, +/- 4%)
    // The ocean breathes. The tone is alive, not mechanical.
    breath = SinOsc.kr(0.065, 0, 0.04, 1.0);

    // === DYNAMICS ===
    // Intensity (overall acoustic energy) modulates presence
    // Low energy: tone recedes but never disappears (floor at 0.35)
    // High energy: full presence
    ampMod = intensity.linlin(0.15, 0.85, 0.35, 1.0).clip(0.35, 1.0);
    // Onset events create brief amplitude swells (+25% max)
    // The asymmetric envelope (fast attack, slow release) makes these organic
    ampMod = ampMod * (1.0 + (onsetEnv * 0.25));

    // === SPATIAL ===
    // Ascending: slight right pan. Descending: slight left.
    // Subtle, not distracting — just a sense of directionality.
    pan = velocity.linlin(0.2, 0.8, -0.3, 0.3);
    sig = Pan2.ar(sig, pan);

    // === OUTPUT ===
    env = EnvGen.kr(Env.asr(4, 1, 5), gate, doneAction: 2);
    Out.ar(out, sig * env * amp * ampMod * breath);
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
    sig = In.ar(in, 2);
    sig = FreeVerb2.ar(sig[0], sig[1], mix, room, damp);
    sig = HPF.ar(sig, 35);   // prevent sub-bass mud from reverb tail
    sig = Limiter.ar(sig, 0.92);
    Out.ar(out, sig * masterVol);
}).add;

"SynthDefs loaded: migrationArc, migrationReverb.".postln;
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
                { hourFrac < (5.75 / 24) }  { Color.new255(10, 18, 45, 50) }   // night: deep blue
                { hourFrac < (8.5 / 24) }   { Color.new255(90, 55, 15, 45) }   // dawn: amber
                { hourFrac < (19.5 / 24) }  { Color.new255(65, 60, 20, 30) }   // day: pale gold
                { hourFrac < (20.25 / 24) } { Color.new255(90, 35, 55, 50) }   // dusk: warm pink
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

        // --- Depth trace: glow layer (thick, soft) ---
        Pen.strokeColor_(Color.new255(100, 170, 255, 50));
        Pen.width_(5);
        Pen.moveTo(0 @ ((~vizDepthTrace[0] / depthMax) * h));
        ~vizN.do { |i|
            var x = (i / (~vizN - 1)) * w;
            var y = ((~vizDepthTrace[i] ? 400) / depthMax).clip(0, 1) * h;
            Pen.lineTo(x @ y);
        };
        Pen.stroke;

        // --- Depth trace: main line (thin, bright) ---
        Pen.strokeColor_(Color.new255(180, 215, 255, 220));
        Pen.width_(1.5);
        Pen.moveTo(0 @ ((~vizDepthTrace[0] / depthMax) * h));
        ~vizN.do { |i|
            var x = (i / (~vizN - 1)) * w;
            var y = ((~vizDepthTrace[i] ? 400) / depthMax).clip(0, 1) * h;
            Pen.lineTo(x @ y);
        };
        Pen.stroke;

        // --- Current depth marker (bright dot on the arc) ---
        // Interpolate vizDepthTrace so dot rides exactly on the drawn line
        if(pct > 0.001) {
            var cx = pct * w;
            var vizIdx = (pct * (~vizN - 1)).clip(0, ~vizN - 1);
            var vizLo = vizIdx.floor.asInteger;
            var vizHi = (vizLo + 1).min(~vizN - 1);
            var vizFrac = vizIdx - vizLo;
            var dotDepth = (~vizDepthTrace[vizLo] * (1 - vizFrac)) + (~vizDepthTrace[vizHi] * vizFrac);
            var cy = (dotDepth / depthMax).clip(0, 1) * h;
            // Outer glow
            Pen.fillColor_(Color.new255(255, 255, 255, 80));
            Pen.fillOval(Rect(cx - 7, cy - 7, 14, 14));
            // Inner dot
            Pen.fillColor_(Color.new255(255, 255, 255, 230));
            Pen.fillOval(Rect(cx - 3.5, cy - 3.5, 7, 7));
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
    var entScaled;
    // Histogram + onset + adaptive window
    var h0val, h1val, h2val, h3val, h4val, h5val, h6val, h7val;
    var onsetVal;
    var velPreview, velMag, adaptiveWindow;

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

        entScaled = entropyVal.linlin(0.6, 0.9, 0, 1).clip(0, 1);

        // ===== SECTION DETECTION =====
        candidateSection = case
            { hourVal < (5.75  / 24) } { \night1 }
            { hourVal < (8.5   / 24) } { \dawn   }
            { hourVal < (19.5  / 24) } { \day    }
            { hourVal < (20.25 / 24) } { \dusk   }
            { true }                    { \night2 };

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
        // Pitch: dvm_depth_norm -> 80-500 Hz (E2 to B4)
        // Night (dvm ~0.86, 180m): ~410 Hz (Ab4) — warm, present
        // Day (dvm ~0.37, 550m):   ~140 Hz (C#3) — deep, felt
        // Dawn descent: ~410 -> 140 Hz over ~55s — ~1.6 octave slide down
        // Dusk ascent:  ~140 -> 410 Hz in ~15s — rapid sweep up
        migSynth.set(
            \freq,      depthVal.linexp(0.2, 1.0, 65, 520).clip(60, 560),
            \amp,       ~masterVol * 0.5,
            \entropy,   entropyVal,
            \velocity,  velocityVal,
            \anomaly,   anomalyVal,
            \intensity, intensityVal,
            \onset,     onsetVal,
            \h0, h0val, \h1, h1val, \h2, h2val, \h3, h3val,
            \h4, h4val, \h5, h5val, \h6, h6val, \h7, h7val
        );

        // ===== UPDATE REVERB =====
        // Night: intimate room. Day: vast cathedral.
        fxSynth.set(
            \mix,       entScaled.linlin(0, 1, 0.12, 0.45),
            \room,      entScaled.linlin(0, 1, 0.2, 0.82),
            \damp,      entScaled.linlin(0, 1, 0.55, 0.18),
            \masterVol, ~masterVol
        );

        // ===== GUI UPDATE =====
        ~currentTime = currentTime;
        ~depthRawVal = depthRawVal;

        {
            var rHour = (hourVal * 24).floor.asInteger;
            var rMin  = ((hourVal * 24 * 60) % 60).floor.asInteger;
            var velDir = if(velRawVal > 0) { " asc" } { " dsc" };

            if(~infoDisplay.notNil and: { ~infoDisplay.isClosed.not }) {
                ~infoDisplay.string_(
                    "UTC: " ++ rHour.asString.padLeft(2, "0") ++ ":" ++
                    rMin.asString.padLeft(2, "0") ++
                    "   Depth: " ++ depthRawVal.round(1) ++ "m" ++
                    "   Vel: " ++ velRawVal.round(0.1) ++ " m/h" ++ velDir ++
                    "\nEntropy: " ++ entropyVal.round(0.001) ++
                    "   Anomaly: " ++ anomalyVal.round(0.01) ++
                    "   Intensity: " ++ intensityVal.round(0.01)
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
  A warm, clear tone around Ab4 (410 Hz). Near-pure sine with gentle
  chorus from the detuned carrier pair. Intimate reverb. The organisms
  are packed in dense layers near the surface (~180m) — organized,
  purposeful. Entropy is low, so the FM index is minimal.

Dawn descent (1:55-2:50):
  The pitch falls steadily — about 1.6 octaves over a minute. As
  entropy rises, the timbre complexifies: new overtones bloom, the
  sound becomes richer and more ambiguous. The reverb opens up. The
  organisms are dispersing, sinking, spreading thin. The LPF closes
  as they descend — the sound darkens.

Day (2:50-6:30):
  A lower, rich drone around C#3 (140 Hz). Complex FM harmonics shift
  slowly. Vast reverb. This is the longest section — near-stillness.
  The sub-harmonic (octave below) adds physical weight. The FM ratio
  near sqrt(2) creates bell-like inharmonic overtones that wander.
  The organisms are hiding ~500m below the surface.

Dusk ascent (6:30-7:00):
  The pitch sweeps up as organisms race toward the surface. The timbre
  clarifies — complex FM resolves toward pure sine. Vibrato intensifies.
  The LPF opens wide — brightness floods in. Reverb contracts.

Night return (7:00-8:00):
  Resolution. The warm, clear tone returns. The cycle is complete.
  Tomorrow it happens again.

NEXT SKETCHES:
  02 — Layer events (Klank bells on scattering layer birth/death)
  03 — Zone energy (8-band BPF filter bank from depth histogram)
  04 — Onset pulses (high-percussion on acoustic transients)
  05 — Composed (all voices combined with intentional arrangement)
*/
