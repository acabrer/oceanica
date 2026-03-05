/*
================================================================================
SONIFICACIÓ OCEÀNICA - Version 8: DATA-DRIVEN MINIMAL COMPOSITION
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya

DESIGN PRINCIPLES:
  1. RESTRAINT: 4 voices, each explainable in one sentence
  2. THRESHOLD-GATED: Silence when data says silence — not everything plays always
  3. 1:1 DATA MAPPING: Every sound = a biological phenomenon
  4. CONTRAST: Night is alive (dense layers), Day is still (diffuse), Bridges are events

VOICE ARCHITECTURE:
  Migration  (80-800 Hz)  — FM tone tracking center-of-mass depth; entropy modulates timbre
  Pulse      (70-600 Hz)  — Resonant percussion on acoustic onset events; threshold-gated
  Layers     (80-800 Hz)  — Klank pings on layer birth/death events; discrete triggers
  Zones      (60-4000 Hz) — 8-band BPF filter bank; soloable per depth zone

SIGNAL ROUTING:
  All voices → fxBus → FreeVerb2 → Limiter → Out

DATA: sonification_sc_v6_{date}.json (v6: shorter smoothing, event features, 8-band histogram)
================================================================================
*/

// ============================================================================
// SECTION 1: SERVER BOOT
// ============================================================================

(
s.options.numOutputBusChannels = 2;
s.options.numInputBusChannels  = 0;
s.options.memSize     = 2.pow(20);
s.options.sampleRate  = 48000;
s.options.numBuffers  = 8192;

~dataPath = thisProcess.nowExecutingPath.dirname.dirname +/+ "output/data/";
~dateStr = "20110126";

s.waitForBoot({
    "Server ready.".postln;
});
)

// ============================================================================
// SECTION 2: LOAD V6 DATA
// ============================================================================

(
var jsonFile, jsonString;

jsonFile   = File(~dataPath ++ "sonification_sc_v6_" ++ ~dateStr ++ ".json", "r");
jsonString = jsonFile.readAllString;
jsonFile.close;
~jsonData  = jsonString.parseJSON;

// --- Core features ---
~times      = ~jsonData["38kHz"]["time_seconds"].asArray.collect(_.asFloat);
~depth      = ~jsonData["38kHz"]["depth_norm"].asArray.collect(_.asFloat);
~intensity  = ~jsonData["38kHz"]["intensity_norm"].asArray.collect(_.asFloat);
~velocity   = ~jsonData["38kHz"]["velocity_norm"].asArray.collect(_.asFloat);
~anomaly    = ~jsonData["38kHz"]["anomaly_norm"].asArray.collect(_.asFloat);
~hour       = ~jsonData["38kHz"]["hour_norm"].asArray.collect(_.asFloat);
~entropy    = ~jsonData["38kHz"]["acoustic_entropy"].asArray.collect(_.asFloat);
~onset      = ~jsonData["38kHz"]["onset_strength_norm"].asArray.collect(_.asFloat);
~kurtosis   = ~jsonData["38kHz"]["kurtosis_norm"].asArray.collect(_.asFloat);
~spread     = ~jsonData["38kHz"]["spread_norm"].asArray.collect(_.asFloat);
~depthRaw   = ~jsonData["38kHz"]["depth_m"].asArray.collect(_.asFloat);
~velocityRaw= ~jsonData["38kHz"]["velocity_m_h"].asArray.collect(_.asFloat);
~regimeScore = ~jsonData["38kHz"]["regime_change_score"].asArray.collect(_.asFloat);

// --- Layer events ---
~layerEventType  = ~jsonData["38kHz"]["layer_event_type"].asArray.collect(_.asFloat);
~layerEventDepth = ~jsonData["38kHz"]["layer_event_depth_norm"].asArray.collect(_.asFloat);

// --- 8-band histogram + differential ---
~hist8 = Array.fill(8, { |i|
    ~jsonData["38kHz"]["histogram_8band_" ++ i].asArray.collect(_.asFloat);
});
~histDiff8 = Array.fill(8, { |i|
    ~jsonData["38kHz"]["histogram_diff_" ++ i].asArray.collect(_.asFloat);
});

// --- Playback timing ---
~speed         = 1.0;
~totalDuration = 480.0;   // 8 minutes at 1x
~compression   = ~times.last / ~totalDuration;

// --- Pre-compute 400-point display traces ---
~vizN = 400;
~vizDepthTrace = Array.fill(~vizN, { |i|
    var idx = ((i / (~vizN - 1)) * (~depthRaw.size - 1)).round.asInteger.clip(0, ~depthRaw.size - 1);
    ~depthRaw[idx] ? 400;
});

// --- Pre-compute 8-band echogram (400 columns × 8 rows) ---
~vizEchoW = 400;
~vizEchogram8 = Array.fill(~vizEchoW, { |col|
    var idx = ((col / (~vizEchoW - 1)) * (~times.size - 1)).round.asInteger.clip(0, ~times.size - 1);
    Array.fill(8, { |band| ~hist8[band][idx] ? 0 });
});

// --- Per-band normalization (2nd-98th percentile per band) ---
// This ensures each depth zone uses its full dynamic range independently.
// Without this, Band 0 (surface, mean=0.58) dominates and DVM bands are inaudible.
~hist8Min = Array.fill(8, 0.0);
~hist8Max = Array.fill(8, 1.0);
8.do { |i|
    var sorted = ~hist8[i].copy.sort;
    var lo = sorted[(sorted.size * 0.02).asInteger];
    var hi = sorted[(sorted.size * 0.98).asInteger];
    ~hist8Min[i] = lo;
    ~hist8Max[i] = hi.max(lo + 0.001);  // avoid division by zero
    ("Band " ++ i ++ ": p2=" ++ lo.round(0.001) ++ "  p98=" ++ hi.round(0.001)).postln;
};

// Band amplitude scaling: surface (Band 0) and broadband (Band 7) are attenuated
// because they're near-constant — scientifically valid signal but sonically drowns DVM dynamics.
// DVM bands (2-6) are boosted to make the migration audible.
~zoneBandScale = [0.25, 0.6, 1.0, 1.0, 1.0, 1.0, 0.7, 0.15];

// --- Ring buffer for onset + event markers ---
~recentOnsets = Array.new(20);
~recentEvents = Array.new(30);

// --- Shared GUI state ---
~currentTime   = 0.0;
~depthRawVal   = 400.0;
~currentHist8  = Array.fill(8, 0.0);

"".postln;
"=== V6 DATA LOADED ===".postln;
("Points: " ++ ~times.size).postln;
("Duration: " ++ ~times.last.round(1) ++ " s (" ++ (~times.last / 3600).round(0.1) ++ " h)").postln;
("Compression: " ++ ~compression.round(1) ++ "x -> " ++ ~totalDuration ++ " s composition").postln;
"Features: 8-band histogram, histogram differential, layer events, regime changes".postln;
)

// ============================================================================
// SECTION 3: CONTROL BUSES
// ============================================================================

(
~ctrl = IdentityDictionary.new;

// Master
~ctrl[\masterVol] = Bus.control(s, 1).set(0.75);
~ctrl[\reverbMix] = Bus.control(s, 1).set(0.3);
~ctrl[\onsetThreshold] = Bus.control(s, 1).set(0.15);

// 4 voices
~voiceNames = [\migration, \pulse, \layers, \zones];

~voiceVol  = IdentityDictionary.new;
~voiceMute = IdentityDictionary.new;
~voiceSolo = IdentityDictionary.new;

~voiceNames.do { |name|
    ~voiceVol[name]  = Bus.control(s, 1).set(0.5);
    ~voiceMute[name] = Bus.control(s, 1).set(0);
    ~voiceSolo[name] = Bus.control(s, 1).set(0);
};

// Default voice levels
~voiceVol[\migration].set(0.6);
~voiceVol[\pulse].set(0.5);
~voiceVol[\layers].set(0.45);
~voiceVol[\zones].set(0.4);

// 8 zone mute buttons (for the filter bank)
~zoneMute = Array.fill(8, { Bus.control(s, 1).set(0) });
~zoneSolo = Array.fill(8, { Bus.control(s, 1).set(0) });

// Helper: resolve solo/mute/volume logic for a voice
~getVoiceAmp = { |voiceName, baseAmp|
    var vol   = ~voiceVol[voiceName].getSynchronous;
    var muted = ~voiceMute[voiceName].getSynchronous;
    var soloed = ~voiceSolo[voiceName].getSynchronous;
    var anySolo = ~voiceNames.any { |n| ~voiceSolo[n].getSynchronous > 0 };

    if(muted > 0) { 0.0 } {
        if(anySolo) {
            if(soloed > 0) { baseAmp * vol } { 0.0 }
        } { baseAmp * vol }
    };
};

// Helper: resolve zone solo/mute for a single band
~getZoneAmp = { |bandIdx, baseAmp|
    var muted = ~zoneMute[bandIdx].getSynchronous;
    var soloed = ~zoneSolo[bandIdx].getSynchronous;
    var anySolo = 8.collect({ |i| ~zoneSolo[i].getSynchronous }).any({ |v| v > 0 });

    if(muted > 0) { 0.0 } {
        if(anySolo) {
            if(soloed > 0) { baseAmp } { 0.0 }
        } { baseAmp }
    };
};

"Control buses ready (4 voices + 8 zone solo/mute).".postln;
)

// ============================================================================
// SECTION 4: SYNTHDEFS (v8 — 4 voices + FX)
// ============================================================================

(
// -------------------------------------------------------------------------
// migrationTone — FM synthesis tracking center-of-mass depth
// Biology: Where the bulk of biomass sits right now
// Low entropy (night, dense layers) = clean sine
// High entropy (day, diffuse) = complex FM timbre
//
// DATA RANGES (measured from v6 extraction):
//   depth_norm:  0.15-0.95 (most time in 0.3-0.6)
//   entropy:     0.60-0.90 (night ~0.65, day ~0.87)
// Remapping uses these actual ranges so full musical range is exploited.
// -------------------------------------------------------------------------
SynthDef(\migrationTone, { |out=0, freq=200, amp=0.3,
    entropy=0.5, anomaly=0.5, velocity=0.5, gate=1, lagTime=1.5|
    var modIndex, modFreq, carrier, sig, env, entScaled;

    freq    = Lag.kr(freq, lagTime);
    amp     = Lag.kr(amp, lagTime);
    entropy = Lag.kr(entropy, lagTime);
    anomaly = Lag.kr(anomaly, lagTime);
    velocity = Lag.kr(velocity, lagTime);

    // Remap entropy from actual data range [0.6, 0.9] → [0, 1]
    // Night (dense organized layers) → 0 → clean sine
    // Day (diffuse scattered organisms) → 1 → complex FM
    entScaled = entropy.linlin(0.6, 0.9, 0, 1).clip(0, 1);

    // FM: remapped entropy controls modulation depth
    modIndex = entScaled.linlin(0, 1, 0.1, 4.0);
    modFreq  = SinOsc.ar(
        freq * entScaled.linlin(0, 1, 1.0, 2.01),  // ratio 1:1 (pure) to ~2 (octave shimmer)
        0,
        freq * modIndex
    );
    carrier = SinOsc.ar(freq + modFreq);

    // Anomaly adds subtle detuned second voice (tension when depth deviates from expected)
    carrier = carrier + (SinOsc.ar(
        freq * anomaly.linlin(0.3, 0.7, 0.998, 1.015),
        0,
        anomaly.linlin(0.3, 0.7, 0.0, 0.25)
    ));

    sig = carrier;

    // Gentle LPF to tame FM harshness — opens with entropy
    sig = LPF.ar(sig, freq * entScaled.linlin(0, 1, 3, 10));

    // Velocity → subtle pan (descending = left, ascending = right)
    sig = Pan2.ar(sig, velocity.linlin(0, 1, -0.4, 0.4));

    env = EnvGen.kr(Env.asr(2.0, 1, 3.0), gate, doneAction: 2);
    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// pulseTrigger — resonant percussion on acoustic onset events
// Biology: Sudden changes in the water column (onset_strength threshold crossing)
// Kurtosis controls resonance sharpness
// One-shot: spawned per event, doneAction:2
// -------------------------------------------------------------------------
SynthDef(\pulseTrigger, { |out=0, freq=200, amp=0.3, sustain=0.2,
    kurtosis=0.5, depthNorm=0.5|
    var trig, body, click, sig, env, pan, rq;

    // Resonant body — kurtosis sharpens resonance
    rq = kurtosis.linlin(0, 1, 0.4, 0.03);
    body = Ringz.ar(
        Impulse.ar(0),
        freq,
        sustain * 2
    ) * 0.3;

    // Click attack — short noise burst
    click = BPF.ar(
        WhiteNoise.ar(0.5),
        (freq * 2).clip(100, 12000),
        0.15
    ) * EnvGen.ar(Env.perc(0.001, 0.02), doneAction: 0);

    sig = body + click;

    // Depth panning: shallow = wider stereo, deep = center
    pan = Rand(-1, 1) * depthNorm.linlin(0, 1, 0.15, 0.65);
    sig = Pan2.ar(sig, pan);

    env = EnvGen.kr(Env.perc(0.001, sustain), doneAction: 2);
    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// layerBirth — Klank resonant ping for new scattering layer forming
// Biology: Organisms aggregating into a new layer — ascending bell
// One-shot: doneAction:2
// -------------------------------------------------------------------------
SynthDef(\layerBirth, { |out=0, freq=300, amp=0.2|
    var sig, env, pan;

    sig = Klank.ar(
        `[freq * [1, 2.2, 3.5, 5.1], [0.3, 0.2, 0.12, 0.06], [0.3, 0.2, 0.15, 0.1]],
        Impulse.ar(0)
    );
    sig = LPF.ar(sig, 6000);

    pan = Rand(-0.6, 0.6);
    sig = Pan2.ar(sig, pan);

    env = EnvGen.kr(Env.perc(0.005, 0.4), doneAction: 2);
    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// layerDeath — Klank decay for scattering layer dissolving
// Biology: Organisms dispersing — descending subharmonic decay
// One-shot: doneAction:2
// -------------------------------------------------------------------------
SynthDef(\layerDeath, { |out=0, freq=300, amp=0.2|
    var sig, env, pan;

    sig = Klank.ar(
        `[freq * [1, 0.75, 0.5, 0.33], [0.3, 0.2, 0.12, 0.06], [0.6, 0.5, 0.4, 0.3]],
        Impulse.ar(0)
    );
    sig = LPF.ar(sig, 2000);

    pan = Rand(-0.6, 0.6);
    sig = Pan2.ar(sig, pan);

    env = EnvGen.kr(Env.perc(0.005, 0.8), doneAction: 2);
    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// zoneFilter — 8-band BPF filter bank (soloable depth zones)
// Biology: Energy distribution across 8 oceanographic depth zones (10-1000m)
// Each band opens when its zone has energy; differential modulates Q
// Persistent voice, amplitude controlled per band
// -------------------------------------------------------------------------
SynthDef(\zoneFilter, { |out=0, amp=0.3, gate=1, lagTime=0.5|
    var bands = 8;
    var input, sig, env;
    // Band amplitudes (from histogram_8band)
    var b0 = \b0.kr(0), b1 = \b1.kr(0), b2 = \b2.kr(0), b3 = \b3.kr(0);
    var b4 = \b4.kr(0), b5 = \b5.kr(0), b6 = \b6.kr(0), b7 = \b7.kr(0);
    // Differential (energy flow direction per band)
    var d0 = \d0.kr(0.5), d1 = \d1.kr(0.5), d2 = \d2.kr(0.5), d3 = \d3.kr(0.5);
    var d4 = \d4.kr(0.5), d5 = \d5.kr(0.5), d6 = \d6.kr(0.5), d7 = \d7.kr(0.5);

    var bandAmps = [b0, b1, b2, b3, b4, b5, b6, b7];
    var bandDiffs = [d0, d1, d2, d3, d4, d5, d6, d7];
    // Frequencies: 60 Hz (deep, band 6-7) to 4000 Hz (surface, band 0)
    // Mapped to oceanographic zones: surface=high freq, deep=low freq
    var bandFreqs = [3200, 2000, 1200, 700, 400, 200, 100, 60];
    // Stereo position: surface wide, deep center
    var bandPans = [0.6, 0.45, 0.3, 0.15, -0.15, -0.3, -0.45, -0.5];

    amp = Lag.kr(amp, lagTime);

    // Excitation: pink noise (organic broadband)
    input = PinkNoise.ar(0.5);

    sig = Mix.fill(bands, { |i|
        var bandAmp = Lag.kr(bandAmps[i], lagTime);
        var diff = Lag.kr(bandDiffs[i], lagTime);
        // Differential modulates Q: gaining energy (diff>0.5) = sharper resonance
        var rq = diff.linlin(0.3, 0.7, 0.4, 0.08).clip(0.02, 0.5);
        var filtered = BPF.ar(input, bandFreqs[i], rq) * bandAmp * 4.0;
        Pan2.ar(filtered, bandPans[i]);
    });

    env = EnvGen.kr(Env.asr(2, 1, 3), gate, doneAction: 2);
    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// masterFXv8 — Simplified: reverb + limiter only
// -------------------------------------------------------------------------
SynthDef(\masterFXv8, { |in=0, out=0, reverbMix=0.3, reverbRoom=0.6,
    entropy=0.5, masterVol=0.75, muteAll=0|
    var sig, reverbed, muteGate;

    muteGate = 1 - Lag.kr(muteAll, 0.5);
    sig = In.ar(in, 2);

    // Entropy-driven reverb: high entropy (diffuse) = larger room
    reverbed = FreeVerb2.ar(sig[0], sig[1],
        mix: reverbMix,
        room: entropy.linlin(0.3, 0.9, 0.3, 0.75),
        damp: 0.35
    );

    reverbed = HPF.ar(reverbed, 40);
    reverbed = Limiter.ar(reverbed, 0.92);

    Out.ar(out, reverbed * masterVol * muteGate);
}).add;

"SynthDefs loaded (v8: migrationTone, pulseTrigger, layerBirth, layerDeath, zoneFilter, masterFXv8).".postln;
)

// ============================================================================
// SECTION 5: GUI (v8 — Simplified)
// ============================================================================

(
~createGUI = {
    var win, makeKnob;

    if(~guiWindow.notNil) { ~guiWindow.close };

    ~guiWindow = Window("Sonificacio Oceanica v8 — Data-Driven Minimal", Rect(40, 30, 1050, 580))
        .background_(Color.gray(0.08))
        .front;

    win = ~guiWindow;

    // --- Helper: standard knob ---
    makeKnob = { |parent, bounds, label, bus, lo=0, hi=1, color|
        var spec = ControlSpec(lo, hi);
        var col = color ? Color.cyan;
        StaticText(parent, Rect(bounds.left, bounds.top, bounds.width, 14))
            .string_(label)
            .stringColor_(col)
            .align_(\center)
            .font_(Font("Helvetica", 9));
        Knob(parent, Rect(bounds.left + 4, bounds.top + 15, bounds.width - 8, bounds.width - 8))
            .value_(spec.unmap(bus.getSynchronous))
            .color_([Color.gray(0.2), col, col.alpha_(0.5), Color.white])
            .action_({ |k| bus.set(spec.map(k.value)) });
    };

    // ========= HEADER BAR =========
    StaticText(win, Rect(10, 6, 500, 22))
        .string_("SONIFICACIO OCEANICA  v8  —  Data-Driven Minimal Composition")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 14));

    // Transport
    Button(win, Rect(740, 5, 75, 26))
        .states_([
            ["PLAY", Color.black, Color.new255(50, 220, 80)],
            ["STOP", Color.white, Color.new255(200, 50, 50)]
        ])
        .font_(Font("Helvetica Bold", 11))
        .action_({ |btn|
            if(btn.value == 1) { ~playComposition.value } { ~stopAll.value };
        });

    Button(win, Rect(820, 5, 75, 26))
        .states_([["REC", Color.white, Color.gray(0.3)]])
        .font_(Font("Helvetica Bold", 11))
        .action_({
            if(s.isRecording)
                { s.stopRecording; "Stopped recording.".postln }
                { s.record;        "Recording...".postln };
        });

    Button(win, Rect(900, 5, 75, 26))
        .states_([["REFRESH", Color.white, Color.gray(0.3)]])
        .font_(Font("Helvetica", 10))
        .action_({ ~createGUI.value });

    // ========= ECHOGRAM VISUALIZATION (550 x 260) =========
    StaticText(win, Rect(10, 34, 550, 14))
        .string_("VERTICAL STRUCTURE  (8-band energy x time; white = depth of mass; green = playhead)")
        .stringColor_(Color.gray(0.55))
        .font_(Font("Helvetica", 9));

    ~echoView = UserView(win, Rect(10, 48, 550, 260))
        .background_(Color.black)
        .animate_(true)
        .frameRate_(15);

    ~echoView.drawFunc = { |uv|
        var w = uv.bounds.width;
        var h = uv.bounds.height;
        var pct = (~currentTime / ~totalDuration).clip(0, 1);
        var bandH = h / 8;

        // Draw 8-band energy columns
        ~vizEchoW.do { |col|
            var x = (col / (~vizEchoW - 1)) * w;
            var hist = ~vizEchogram8[col];
            if(hist.notNil) {
                8.do { |row|
                    var val = hist[row] ? 0;
                    var y = row * bandH;  // band 0 (surface) at top, band 7 (deep) at bottom
                    var hue = 0.63 - (val.clip(0, 1) * 0.63);
                    Pen.fillColor_(Color.hsv(hue.clip(0, 1), 0.85, val.linlin(0, 0.3, 0.05, 0.9).clip(0.05, 0.9), 0.85));
                    Pen.fillRect(Rect(x, y, (w / ~vizEchoW).ceil + 1, bandH.ceil + 1));
                };
            };
        };

        // Depth-of-mass trace (white)
        Pen.strokeColor_(Color.new255(255, 255, 255, 200));
        Pen.width_(1.5);
        Pen.moveTo(0 @ (((~vizDepthTrace[0] ? 400) / 1000) * h));
        ~vizN.do { |i|
            var x = (i / (~vizN - 1)) * w;
            var depthM = (~vizDepthTrace[i] ? 400).clip(0, 1000);
            var y = (depthM / 1000) * h;
            Pen.lineTo(x @ y);
        };
        Pen.stroke;

        // Zone boundary labels
        ["10-50m", "50-150m", "150-300m", "300-500m", "500-700m", "700-850m", "850-1000m", "broad"].do { |lbl, i|
            var y = (i * bandH) + (bandH * 0.4);
            Pen.stringAtPoint(lbl, 3 @ y, Font("Helvetica", 7), Color.gray(0.4));
        };

        // Event markers (cyan dots for layer events)
        ~recentEvents.do { |evtPct|
            var ox = evtPct * w;
            Pen.fillColor_(Color.cyan.alpha_(0.6));
            Pen.fillOval(Rect(ox - 2, 2, 4, 4));
        };

        // Onset markers (orange at top)
        ~recentOnsets.do { |evtPct|
            var ox = evtPct * w;
            Pen.strokeColor_(Color.new255(255, 140, 0, 180));
            Pen.width_(1.0);
            Pen.line(ox @ 0, ox @ (h * 0.08));
            Pen.stroke;
        };

        // Green playhead
        Pen.strokeColor_(Color.new255(0, 220, 100));
        Pen.width_(2.0);
        Pen.line((pct * w) @ 0, (pct * w) @ h);
        Pen.stroke;
    };

    // ========= ZONE LEVEL METERS (8 bars, 180 x 260) =========
    StaticText(win, Rect(570, 34, 180, 14))
        .string_("DEPTH ZONES (8-band energy)")
        .stringColor_(Color.gray(0.55))
        .font_(Font("Helvetica", 9));

    ~zoneView = UserView(win, Rect(570, 48, 180, 260))
        .background_(Color.gray(0.1))
        .animate_(true)
        .frameRate_(15);

    ~zoneView.drawFunc = { |uv|
        var w = uv.bounds.width;
        var h = uv.bounds.height;
        var bandH = h / 8;
        var barMaxW = w - 55;
        var zoneLabels = ["10-50m", "50-150m", "150-300m", "300-500m",
            "500-700m", "700-850m", "850-1000m", "broad"];
        var zoneColors = [
            Color.new255(50, 220, 80),    // surface green
            Color.new255(100, 200, 150),   // light green
            Color.new255(0, 200, 255),     // cyan (DVM night)
            Color.new255(100, 150, 255),   // blue
            Color.new255(200, 100, 255),   // purple (DVM day)
            Color.new255(150, 80, 200),    // dark purple
            Color.new255(80, 60, 150),     // deep blue
            Color.gray(0.5)                // broadband
        ];

        8.do { |i|
            var val = ~currentHist8[i] ? 0;
            var y = i * bandH;
            var barW = val.clip(0, 1) * barMaxW;

            // Zone label
            Pen.stringAtPoint(zoneLabels[i], 0 @ (y + (bandH * 0.35)),
                Font("Helvetica", 8), zoneColors[i]);

            // Energy bar
            Pen.fillColor_(zoneColors[i].alpha_(0.8));
            Pen.fillRect(Rect(50, y + 4, barW, bandH - 8));
        };
    };

    // ========= INFO PANEL =========
    ~infoDisplay = StaticText(win, Rect(760, 48, 280, 80))
        .string_("UTC: --:--    Section: --------\n" ++
                 "Depth: --- m   Velocity: --- m/h\n" ++
                 "Entropy: ----  Regime: ----")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 10))
        .background_(Color.gray(0.12));

    ~sectionDisplay = StaticText(win, Rect(760, 132, 280, 24))
        .string_("SECTION: ---")
        .stringColor_(Color.cyan)
        .align_(\center)
        .font_(Font("Helvetica Bold", 13))
        .background_(Color.gray(0.18));

    // ========= VOICE CONTROLS (4 voices) =========
    StaticText(win, Rect(10, 318, 200, 14))
        .string_("VOICES")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 11));

    [\migration, \pulse, \layers, \zones].do { |name, i|
        var x = 10 + (i * 120);
        var labels = ["MIGRATION", "PULSE", "LAYERS", "ZONES"];
        var colors = [
            Color.new255(0, 200, 255),     // migration = cyan (DVM depth tracking)
            Color.new255(255, 160, 50),     // pulse = orange (onset events)
            Color.new255(50, 220, 80),      // layers = green (births/deaths)
            Color.new255(200, 100, 255)     // zones = purple (filter bank)
        ];

        StaticText(win, Rect(x, 334, 110, 14))
            .string_(labels[i])
            .stringColor_(colors[i])
            .align_(\center)
            .font_(Font("Helvetica Bold", 10));

        makeKnob.(win, Rect(x + 25, 348, 55, 55), "Vol", ~voiceVol[name], 0, 1, colors[i]);

        // Solo
        Button(win, Rect(x + 10, 414, 42, 20))
            .states_([
                ["S", Color.white, Color.gray(0.25)],
                ["S", Color.black, Color.yellow]
            ])
            .font_(Font("Helvetica Bold", 10))
            .action_({ |btn| ~voiceSolo[name].set(btn.value) });

        // Mute
        Button(win, Rect(x + 58, 414, 42, 20))
            .states_([
                ["M", Color.white, Color.gray(0.25)],
                ["M", Color.white, Color.red(0.6)]
            ])
            .font_(Font("Helvetica Bold", 10))
            .action_({ |btn| ~voiceMute[name].set(btn.value) });
    };

    // ========= ZONE SOLO/MUTE (8 buttons) =========
    StaticText(win, Rect(500, 318, 200, 14))
        .string_("ZONE SOLO / MUTE")
        .stringColor_(Color.new255(200, 100, 255))
        .font_(Font("Helvetica Bold", 10));

    8.do { |i|
        var x = 500 + (i * 62);
        var zoneLabels = ["Surf", "Epi", "Night", "Mid", "Day", "Deep", "Bathy", "Broad"];
        var y = 338;

        StaticText(win, Rect(x, y, 58, 12))
            .string_(zoneLabels[i])
            .stringColor_(Color.gray(0.6))
            .align_(\center)
            .font_(Font("Helvetica", 8));

        // Solo
        Button(win, Rect(x, y + 14, 26, 18))
            .states_([
                ["S", Color.white, Color.gray(0.25)],
                ["S", Color.black, Color.yellow]
            ])
            .font_(Font("Helvetica Bold", 8))
            .action_({ |btn| ~zoneSolo[i].set(btn.value) });

        // Mute
        Button(win, Rect(x + 30, y + 14, 26, 18))
            .states_([
                ["M", Color.white, Color.gray(0.25)],
                ["M", Color.white, Color.red(0.6)]
            ])
            .font_(Font("Helvetica Bold", 8))
            .action_({ |btn| ~zoneMute[i].set(btn.value) });
    };

    // ========= MASTER + REVERB + ONSET THRESHOLD =========
    StaticText(win, Rect(10, 446, 150, 14))
        .string_("MASTER")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 10));

    makeKnob.(win, Rect(10, 460, 50, 50), "Master", ~ctrl[\masterVol], 0, 1, Color.white);
    makeKnob.(win, Rect(70, 460, 50, 50), "Reverb", ~ctrl[\reverbMix], 0, 1, Color.magenta);

    // Onset threshold
    StaticText(win, Rect(140, 446, 90, 14))
        .string_("Onset Thr")
        .stringColor_(Color.new255(255, 160, 50))
        .font_(Font("Helvetica", 9));
    Slider(win, Rect(140, 468, 80, 20))
        .value_(ControlSpec(0.03, 0.5).unmap(~ctrl[\onsetThreshold].getSynchronous))
        .background_(Color.gray(0.2))
        .knobColor_(Color.new255(255, 160, 50))
        .action_({ |sl|
            ~ctrl[\onsetThreshold].set(ControlSpec(0.03, 0.5).map(sl.value));
        });

    // ========= SPEED BUTTONS =========
    StaticText(win, Rect(250, 446, 60, 14))
        .string_("SPEED")
        .stringColor_(Color.gray(0.6))
        .font_(Font("Helvetica", 9));

    ~speedBtns = Array.new(4);
    [0.5, 1.0, 2.0, 4.0].do { |spd, i|
        var lbl = if(spd < 1) { "0.5x" } { spd.asInteger.asString ++ "x" };
        ~speedBtns = ~speedBtns.add(
            Button(win, Rect(250 + (i * 52), 462, 46, 24))
                .states_([
                    [lbl, Color.white, Color.gray(0.28)],
                    [lbl, Color.black, Color.yellow]
                ])
                .font_(Font("Helvetica Bold", 10))
                .value_(if(spd == 1.0) { 1 } { 0 })
                .action_({ |btn|
                    if(btn.value == 1) {
                        ~speedBtns.do { |b| if(b !== btn) { b.value_(0) } };
                        ~speed = spd;
                        ~totalDuration = ~times.last / (~times.last / 480) / ~speed;
                        ~compression   = ~times.last / ~totalDuration;
                        ("Speed: " ++ lbl ++ " — compression: " ++ ~compression.round(1) ++ "x").postln;
                    };
                })
        );
    };

    // ========= CREDITS =========
    StaticText(win, Rect(10, 530, 1030, 30))
        .string_(
            "SONIFICACIO OCEANICA v8 — Oceanica (Alex Cabrer & Joan Sala) | CLT019 Generalitat de Catalunya\n" ++
            "MALASPINA 2010 — Atlantic Ocean — January 26, 2011 — 38 kHz — 29,997 pings — 8-band zones — v6 data"
        )
        .stringColor_(Color.gray(0.4))
        .font_(Font("Helvetica", 9));

    win.onClose = {
        ~stopAll.value;
        "Window closed.".postln;
    };

    "GUI v8 created.".postln;
};

~createGUI.value;
)

// ============================================================================
// SECTION 6: DATA ACCESS FUNCTIONS
// ============================================================================

(
// Windowed mean (reused from v7)
~getValue = { |dataArray, timeArray, compTime, compression, windowSec=2|
    var realTime, startIdx, endIdx, values;
    realTime = compTime * compression;
    startIdx = (timeArray.indexOfGreaterThan(realTime - (windowSec * compression * 0.5)) ? 0).max(0);
    endIdx   = (timeArray.indexOfGreaterThan(realTime + (windowSec * compression * 0.5)) ? (timeArray.size - 1)).min(timeArray.size - 1);
    values   = dataArray.copyRange(startIdx, endIdx);
    values   = values.reject({ |v| v.isNil or: { v.isNaN } });
    if(values.size > 0) { values.mean } { 0.5 };
};

// Nearest-sample lookup
~getRaw = { |dataArray, timeArray, compTime, compression|
    var realTime, idx;
    realTime = compTime * compression;
    idx = (timeArray.indexOfGreaterThan(realTime) ? (timeArray.size - 1)).max(0).min(dataArray.size - 1);
    dataArray[idx] ? 0;
};

"Data functions ready.".postln;
)

// ============================================================================
// SECTION 7: COMPOSITION ENGINE (v8 — Minimal, Threshold-Gated)
// ============================================================================

(
~fxBus = Bus.audio(s, 2);
~playing = false;

~composition = Routine({
    var updateRate = 0.05;
    var currentTime = 0;

    // Synth references
    var synths = IdentityDictionary.new;
    var fxSynth;

    // Data values
    var depthVal, entropyVal, velocityVal, anomalyVal, onsetVal;
    var kurtosisVal, spreadVal, hourVal, regimeVal;
    var depthRawVal, velRawVal;
    var histVals = Array.fill(8, 0.0);
    var diffVals = Array.fill(8, 0.5);
    var eventTypeVal, eventDepthVal;

    // Section tracking
    var section = \night1, lastSection = \none;
    var sectionName, bioState;
    var sectionHoldCount = 0, candidateSection = \none;

    // Onset cooldown
    var lastOnsetTime = -1.0;
    var onsetCooldown;

    // Layer event cooldown (prevent flooding)
    var lastEventTime = -1.0;
    var eventCooldown;

    // Regime duck state
    var regimeDucking = false;
    var duckEndTime = 0;
    var lastRegimeHigh = false;

    // Mute-all
    var allMuted = false;

    "".postln;
    "===========================================================".postln;
    "   SONIFICACIO OCEANICA v8 — Data-Driven Minimal".postln;
    "   4 voices - 8 zones - 8-min cycle".postln;
    "===========================================================".postln;

    // Start FX chain
    fxSynth = Synth(\masterFXv8, [\in, ~fxBus, \out, 0], addAction: \addToTail);

    // Persistent voices
    synths[\migration] = Synth(\migrationTone, [\out, ~fxBus, \amp, 0, \gate, 1]);
    synths[\zones]     = Synth(\zoneFilter, [\out, ~fxBus, \amp, 0, \gate, 1]);

    0.3.wait;
    ~playing = true;

    onsetCooldown = (0.12 / ~speed) / (~compression.sqrt);
    eventCooldown = (0.05 / ~speed) / (~compression.sqrt);

    while({ currentTime < ~totalDuration and: { ~playing } }, {

        // ===== GUARD: skip all server messages if stopped =====
        if(~playing.not) {
            updateRate.wait;
            currentTime = currentTime + updateRate;
        } {

        // ===== READ DATA =====
        depthVal    = ~getValue.(~depth,    ~times, currentTime, ~compression, 1.5 * ~speed);
        entropyVal  = ~getValue.(~entropy,  ~times, currentTime, ~compression, 1.5 * ~speed);
        velocityVal = ~getValue.(~velocity, ~times, currentTime, ~compression, 1.5 * ~speed);
        anomalyVal  = ~getValue.(~anomaly,  ~times, currentTime, ~compression, 1.5 * ~speed);
        onsetVal    = ~getValue.(~onset,    ~times, currentTime, ~compression, 0.5 * ~speed);
        kurtosisVal = ~getValue.(~kurtosis, ~times, currentTime, ~compression, 1.5 * ~speed);
        spreadVal   = ~getValue.(~spread,   ~times, currentTime, ~compression, 1.5 * ~speed);
        hourVal     = ~getValue.(~hour,     ~times, currentTime, ~compression, 1.0 * ~speed);
        regimeVal   = ~getValue.(~regimeScore, ~times, currentTime, ~compression, 0.5 * ~speed);

        depthRawVal = ~getRaw.(~depthRaw, ~times, currentTime, ~compression);
        velRawVal   = ~getRaw.(~velocityRaw, ~times, currentTime, ~compression);

        // 8-band histogram + differential
        // Use getRaw (nearest sample) to preserve dynamics — getValue smoothing kills band variance
        // Apply per-band normalization so each zone uses its full 0-1 range
        8.do { |i|
            var rawHist = ~getRaw.(~hist8[i], ~times, currentTime, ~compression);
            histVals[i] = rawHist.linlin(~hist8Min[i], ~hist8Max[i], 0, 1).clip(0, 1) * ~zoneBandScale[i];
            diffVals[i] = ~getRaw.(~histDiff8[i], ~times, currentTime, ~compression);
        };

        // Layer events (nearest sample — discrete)
        eventTypeVal  = ~getRaw.(~layerEventType, ~times, currentTime, ~compression);
        eventDepthVal = ~getRaw.(~layerEventDepth, ~times, currentTime, ~compression);

        // ===== SECTION LOGIC (with hysteresis) =====
        candidateSection = case
            { hourVal < (5.75  / 24) } { \night1 }
            { hourVal < (8.5   / 24) } { \dawn   }
            { hourVal < (19.5  / 24) } { \day    }
            { hourVal < (20.25 / 24) } { \dusk   }
            { true }                   { \night2 };

        if(candidateSection != section) {
            sectionHoldCount = sectionHoldCount + 1;
            if(sectionHoldCount >= 3) {
                section = candidateSection;
                sectionHoldCount = 0;
            };
        } { sectionHoldCount = 0 };

        #sectionName, bioState = case
            { section == \night1 } { ["NIGHT (Early)", "Surface feeding"] }
            { section == \dawn   } { ["DAWN",          "Descending to depth"] }
            { section == \day    } { ["DAY",           "Deep refuge"] }
            { section == \dusk   } { ["DUSK",          "Ascending to surface"] }
            { true                } { ["NIGHT (Late)", "Surface activity"] };

        lastSection = section;

        // ===== VOICE 1: MIGRATION (persistent FM tone) =====
        // Remap depth from actual data range [0.15, 0.95] → full pitch range
        // depth_norm: 0=deep(580m), 1=shallow(250m) → low freq to high freq
        synths[\migration].set(
            \freq,     depthVal.linexp(0.15, 0.95, 80, 800).clip(80, 800),
            \amp,      ~getVoiceAmp.(\migration, 0.35),
            \entropy,  entropyVal,
            \anomaly,  anomalyVal,
            \velocity, velocityVal
        );

        // ===== VOICE 2: PULSE (threshold-gated percussion) =====
        if((onsetVal > ~ctrl[\onsetThreshold].getSynchronous) and:
           { (currentTime - lastOnsetTime) > onsetCooldown }) {
            var pulseAmp = ~getVoiceAmp.(\pulse, onsetVal.linlin(0.1, 1.0, 0.15, 0.5));
            if(pulseAmp > 0) {
                lastOnsetTime = currentTime;
                Synth(\pulseTrigger, [
                    \out,       ~fxBus,
                    \freq,      depthVal.clip(0.001, 1).linexp(0.001, 1, 70, 600),
                    \amp,       pulseAmp,
                    \sustain,   onsetVal.linlin(0.1, 1.0, 0.08, 0.35),
                    \kurtosis,  kurtosisVal,
                    \depthNorm, depthVal
                ]);
                ~recentOnsets = (~recentOnsets.add(currentTime / ~totalDuration)).keep(-20);
            };
        };

        // ===== VOICE 3: LAYERS (discrete birth/death events) =====
        if(((eventTypeVal > 0.7) or: { eventTypeVal < 0.3 }) and:
           { (currentTime - lastEventTime) > eventCooldown }) {
            var layerAmp = ~getVoiceAmp.(\layers,
                if(eventTypeVal > 0.7) { 0.25 } { 0.18 });
            if(layerAmp > 0) {
                var synthName = if(eventTypeVal > 0.7) { \layerBirth } { \layerDeath };
                lastEventTime = currentTime;
                Synth(synthName, [
                    \out,  ~fxBus,
                    \freq, eventDepthVal.clip(0.001, 1).linexp(0.001, 1, 80, 800),
                    \amp,  layerAmp
                ]);
                ~recentEvents = (~recentEvents.add(currentTime / ~totalDuration)).keep(-30);
            };
        };

        // ===== VOICE 4: ZONES (8-band filter bank) =====
        8.do { |i|
            var zoneAmp = ~getZoneAmp.(i, histVals[i]);
            synths[\zones].set(
                ("b" ++ i).asSymbol, zoneAmp,
                ("d" ++ i).asSymbol, diffVals[i]
            );
        };
        synths[\zones].set(
            \amp, ~getVoiceAmp.(\zones, 0.35)
        );

        // ===== REGIME TRANSITIONS (structural breaths) =====
        if((regimeVal > 0.7) and: { lastRegimeHigh.not }) {
            regimeDucking = true;
            duckEndTime = currentTime + 0.5;
            (">>> REGIME CHANGE at " ++ sectionName ++ " <<<").postln;
        };
        lastRegimeHigh = regimeVal > 0.7;

        if(regimeDucking) {
            if(currentTime > duckEndTime) {
                regimeDucking = false;
                fxSynth.set(\masterVol, ~ctrl[\masterVol].getSynchronous);
            } {
                fxSynth.set(\masterVol, ~ctrl[\masterVol].getSynchronous * 0.7);
            };
        };

        // ===== FX UPDATE =====
        if(regimeDucking.not) {
            fxSynth.set(
                \reverbMix, ~ctrl[\reverbMix].getSynchronous,
                \entropy,   entropyVal,
                \masterVol, ~ctrl[\masterVol].getSynchronous
            );
        };

        // ===== MUTE-ALL DETECTION =====
        allMuted = ~voiceNames.every { |n| ~getVoiceAmp.(n, 1.0) == 0 };
        fxSynth.set(\muteAll, allMuted.asInteger);

        }; // end ~playing guard

        // ===== GUI UPDATE =====
        {
            var mins  = (currentTime / 60).floor.asInteger;
            var secs  = (currentTime % 60).floor.asInteger;
            var rHour = (hourVal * 24).floor.asInteger;
            var rMin  = ((hourVal * 24 * 60) % 60).floor.asInteger;
            var velDir = if(velRawVal > 0) { " (asc)" } { " (dsc)" };

            ~currentTime = currentTime;
            ~depthRawVal = depthRawVal;

            // Smooth 8-band display
            ~currentHist8 = ~currentHist8.collect({ |old, i|
                (old * 0.7) + ((histVals[i] ? 0) * 0.3)
            });

            if(~infoDisplay.notNil and: { ~infoDisplay.isClosed.not }) {
                ~infoDisplay.string_(
                    "UTC: "  ++ rHour.asString.padLeft(2, "0") ++ ":"
                              ++ rMin.asString.padLeft(2,  "0")
                    ++ "    Section: " ++ sectionName ++ "\n"
                    ++ "Depth: " ++ depthRawVal.round(1) ++ " m"
                    ++ "   Velocity: " ++ velRawVal.round(1) ++ " m/h" ++ velDir ++ "\n"
                    ++ "Entropy: " ++ entropyVal.round(0.001)
                    ++ "  Regime: " ++ regimeVal.round(0.01)
                );
            };

            if(~sectionDisplay.notNil and: { ~sectionDisplay.isClosed.not }) {
                ~sectionDisplay.string_(sectionName ++ "  —  " ++ bioState);
                ~sectionDisplay.stringColor_(case
                    { section == \night1 or: { section == \night2 } } { Color.new255(80, 180, 255) }
                    { section == \dawn   } { Color.new255(255, 160, 50) }
                    { section == \day    } { Color.yellow }
                    { section == \dusk   } { Color.new255(255, 100, 180) }
                    { true } { Color.white }
                );
            };
        }.defer;

        updateRate.wait;
        currentTime = currentTime + updateRate;
    });

    // === FADE OUT ===
    if(~playing) {
        "Composition ending — fading out...".postln;
        [synths[\migration], synths[\zones]].do { |syn|
            if(syn.notNil) { try { syn.set(\gate, 0) } };
        };
    };
    ~playing = false;

    4.wait;
    fxSynth.free;

    { if(~sectionDisplay.notNil and: { ~sectionDisplay.isClosed.not }) {
        ~sectionDisplay.string_("COMPLETE  —  24h cycle finished");
        ~sectionDisplay.stringColor_(Color.white);
    } }.defer;

    "Done.".postln;
});

~playComposition = {
    ~playing = false;
    ~composition.reset;
    ~composition.play;
    "Composition started (v8, 8 min).".postln;
};

~stopAll = {
    ~playing = false;  // signal routine to exit before freeing nodes
    ~composition.stop;
    s.freeAll;
    ~recentOnsets = Array.new(20);
    ~recentEvents = Array.new(30);
    "Stopped. All synths freed.".postln;
};

"Composition engine v8 ready.".postln;
)


// ============================================================================
// END
// ============================================================================

/*
HOW TO USE (v8):
  1. Execute sections 1-7 in order (Cmd+Enter each block)
  2. Press PLAY or run ~playComposition.value
  3. Use voice controls to solo/mute any of the 4 voices
  4. Use zone S/M buttons to isolate specific depth zones in the filter bank

VOICE -> DATA MAPPING:
  Migration   <- depth_norm remapped [0.15,0.95]→[80,800 Hz] (3+ octaves),
                 acoustic_entropy remapped [0.6,0.9]→[0,1] (FM index 0.1-4.0),
                 anomaly_norm (detuned 2nd voice), velocity_norm (pan L/R)
                 Night: clean sine (low entropy=organized layers)
                 Day: complex FM (high entropy=diffuse organisms)
                 Biology: Center-of-mass depth of all biomass — the DVM arc

  Pulse       <- onset_strength_norm (trigger threshold), depth_norm (pitch 70-600 Hz),
                 kurtosis_norm (resonance Q), onset strength (sustain)
                 Biology: Sudden acoustic changes — ~137 events at threshold 0.5

  Layers      <- layer_event_type (1=birth, 0=death), layer_event_depth_norm (pitch 80-800 Hz)
                 Birth (layerBirth): Klank ascending bell. Death (layerDeath): Klank descending decay.
                 Biology: Scattering layer formation and dissolution — ~7300 events, bursty

  Zones       <- histogram_8band_0..7 (per-band normalized p2-p98, nearest-sample),
                 histogram_diff_0..7 (Q modulation, nearest-sample)
                 8 BPF filters 60-3200 Hz on pink noise, soloable per zone
                 Band 0 (surface) scaled 0.25x — resident epipelagic community (always active)
                 Band 7 (broadband) scaled 0.15x — near-constant baseline
                 DVM bands 2-6 at full scale — migration dynamics clearly audible
                 Biology: Where acoustic energy lives in the water column right now

  Regime      <- regime_change_score > 0.7 triggers a 0.5s volume duck (~35 per 24h)
                 Creates structural "breaths" at natural section boundaries

DATA: sonification_sc_v6_20110126.json (v6 extraction with short smoothing + events)

SCIENTIFIC NOTES:
  Surface layer (10-50m, Band 0) is always active because of resident epipelagic
  organisms (copepods, siphonophores, fish larvae) sustained by the chlorophyll-a
  maximum. ~50% of mesopelagic organisms don't participate in DVM — they stay put.
  This is well-documented (Bianchi & Mislan 2016, Klevjer et al. 2016).
  Band 0 is NOT an artifact — it's attenuated sonically to let DVM bands speak.
*/
