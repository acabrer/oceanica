/*
================================================================================
SONIFICACIÓ OCEÀNICA - Version 6: PROFESSIONAL MIX & SIGNAL CHAIN
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya

DESIGN PRINCIPLE: Every scientific feature has a direct, audible, and visible
sonic consequence. The four v4 features map to four fundamental musical dimensions:

  acoustic_entropy        → TONALITY     (0=harmonic partials, 1=filtered noise)
  inter_ping_correlation  → STABILITY    (grain dur, envelope times, detuning)
  layer_separation_norm   → INTERVAL     (FM ratio, stereo spread, 2nd voice)
  onset_strength_norm     → IMPULSE      (one-shot resonant transients)

SONIC PHILOSOPHY:
  Night = ALIVE  — oceanicTexture granular swarms + layerVoice + onset hits
  Dawn  = TENSE  — layerVoice FM interval dominant + descent sweep
  Day   = DEEP   — spectralGrain IFFT buffer granular + deepField + freeze
  Dusk  = OPEN   — oceanicTexture ascending + layerVoice + ascent sweep

SECTION TIMING (calibrated DVM model, South Atlantic 25°S Jan 2011):
  night1  00:00 – 05:45 UTC  (hourNorm < 5.75/24)
  dawn    05:45 – 08:30 UTC
  day     08:30 – 19:30 UTC
  dusk    19:30 – 20:15 UTC
  night2  20:15 – 24:00 UTC

v6 CHANGES (from v5):
  - FIX: delay knob precedence bug (was acting as master volume)
  - FIX: distortion gain boost removed (tanh already normalizes)
  - FIX: LeakDC added to layerVoice FM output (prevents DC offset)
  - Dual FX routing: sub-bass (deepField) bypasses delay/reverb
  - HPF at 35 Hz before limiter (removes sub-harmonic buildup)
  - Lag.kr smoothing on all persistent synths (no clicks at transitions)
  - Frequency range separation: deep 25-65, layer 100-500, texture 200-1800
  - Section transition hysteresis (3-tick hold prevents jitter)
  - Speed-scaled onset cooldown and getValue windows
  - Increased spectralGrain position jitter (±5% → ±15%)
  - Reduced max delay feedback (4.0 → 2.5s decay)
================================================================================
*/

// ============================================================================
// SECTION 1: SERVER BOOT + IFFT BUFFER LOAD
// ============================================================================

(
s.options.numOutputBusChannels = 2;
s.options.numInputBusChannels  = 0;
s.options.memSize     = 2.pow(20);
s.options.sampleRate  = 48000;
s.options.numBuffers  = 8192;

~ifftBuf = nil;
~dataPath = thisProcess.nowExecutingPath.dirname.dirname +/+ "output/data/";

s.waitForBoot({
    "Server ready. Loading IFFT buffer...".postln;
    ~ifftBuf = Buffer.read(s, ~dataPath ++ "echogram_audio_20110126_ifft.wav", action: { |buf|
        ("IFFT buffer loaded: " ++ buf.numFrames ++ " frames, "
            ++ buf.numChannels ++ " ch").postln;
    });
});
)

// ============================================================================
// SECTION 2: LOAD V4 DATA + PRE-COMPUTE DISPLAY TRACES
// ============================================================================

(
var jsonFile, jsonString;

// Load v4 JSON
jsonFile   = File(~dataPath ++ "sonification_sc_v4_20110126.json", "r");
jsonString = jsonFile.readAllString;
jsonFile.close;
~jsonData  = jsonString.parseJSON;

// --- Existing arrays ---
~times      = ~jsonData["38kHz"]["time_seconds"].asArray.collect(_.asFloat);
~depth      = ~jsonData["38kHz"]["depth_norm"].asArray.collect(_.asFloat);
~intensity  = ~jsonData["38kHz"]["intensity_norm"].asArray.collect(_.asFloat);
~spread     = ~jsonData["38kHz"]["spread_norm"].asArray.collect(_.asFloat);
~layers     = ~jsonData["38kHz"]["layers_norm"].asArray.collect(_.asFloat);
~velocity   = ~jsonData["38kHz"]["velocity_norm"].asArray.collect(_.asFloat);
~anomaly    = ~jsonData["38kHz"]["anomaly_norm"].asArray.collect(_.asFloat);
~hour       = ~jsonData["38kHz"]["hour_norm"].asArray.collect(_.asFloat);

// --- New v4 feature arrays ---
~entropy    = ~jsonData["38kHz"]["acoustic_entropy"].asArray.collect(_.asFloat);
~ipc        = ~jsonData["38kHz"]["inter_ping_correlation"].asArray.collect(_.asFloat);
~layerSep   = ~jsonData["38kHz"]["layer_separation_norm"].asArray.collect(_.asFloat);
~onset      = ~jsonData["38kHz"]["onset_strength_norm"].asArray.collect(_.asFloat);
~surfaceN   = ~jsonData["38kHz"]["surface_norm"].asArray.collect(_.asFloat);
~midwaterN  = ~jsonData["38kHz"]["midwater_norm"].asArray.collect(_.asFloat);
~deepN      = ~jsonData["38kHz"]["deep_norm"].asArray.collect(_.asFloat);

// --- Raw for display ---
~depthRaw   = ~jsonData["38kHz"]["depth_m"].asArray.collect(_.asFloat);
~velocityRaw= ~jsonData["38kHz"]["velocity_m_h"].asArray.collect(_.asFloat);

// --- Playback time ---
~speed         = 1.0;
~totalDuration = 240.0;   // 4 minutes at 1x
~compression   = ~times.last / ~totalDuration;

// --- Pre-compute 400-point display traces for GUI UserView (performance) ---
~vizN = 400;
~vizDepthTrace   = Array.fill(~vizN, { |i|
    var idx = ((i / (~vizN - 1)) * (~depthRaw.size - 1)).round.asInteger.clip(0, ~depthRaw.size - 1);
    ~depthRaw[idx] ? 400;
});
~vizEntropyTrace = Array.fill(~vizN, { |i|
    var idx = ((i / (~vizN - 1)) * (~entropy.size - 1)).round.asInteger.clip(0, ~entropy.size - 1);
    ~entropy[idx] ? 0.5;
});

// --- Ring buffer for onset event markers ---
~recentOnsets = Array.new(20);

// --- Shared state for GUI draw ---
~currentTime  = 0.0;
~depthRawVal  = 400.0;

"".postln;
"=== V4 DATA LOADED ===".postln;
("Points: " ++ ~times.size).postln;
("Duration: " ++ ~times.last.round(1) ++ " s").postln;
("Compression: " ++ ~compression.round(1) ++ "x").postln;
"New features: acoustic_entropy, inter_ping_correlation, layer_separation_norm, onset_strength_norm".postln;
)

// ============================================================================
// SECTION 3: CONTROL BUSES
// ============================================================================

(
~ctrl = IdentityDictionary.new;

// Master
~ctrl[\masterVol]      = Bus.control(s, 1).set(0.75);
~ctrl[\chaos]          = Bus.control(s, 1).set(0.5);

// Synthesis voices
~ctrl[\swarmVol]       = Bus.control(s, 1).set(0.6);
~ctrl[\droneVol]       = Bus.control(s, 1).set(0.5);
~ctrl[\layerVol]       = Bus.control(s, 1).set(0.4);
~ctrl[\grainVol]       = Bus.control(s, 1).set(0.5);

// Onset / percussion
~ctrl[\onsetThreshold] = Bus.control(s, 1).set(0.15);
~ctrl[\onsetVol]       = Bus.control(s, 1).set(0.45);

// Feature influence scales (0=off, 1=full)
~ctrl[\entropyMix]     = Bus.control(s, 1).set(1.0);
~ctrl[\ipcMix]         = Bus.control(s, 1).set(1.0);
~ctrl[\layerSepMix]    = Bus.control(s, 1).set(1.0);

// Effects
~ctrl[\reverbMix]      = Bus.control(s, 1).set(0.5);
~ctrl[\delayMix]       = Bus.control(s, 1).set(0.3);
~ctrl[\distortion]     = Bus.control(s, 1).set(0.3);
~ctrl[\feedback]       = Bus.control(s, 1).set(0.3);
~ctrl[\filterMod]      = Bus.control(s, 1).set(0.5);

// Playback speed (informational; actual speed via ~speed)
~ctrl[\speed]          = Bus.control(s, 1).set(1.0);

"Control buses ready.".postln;
)

// ============================================================================
// SECTION 4: SYNTHDEFS (v5)
// ============================================================================

(
// -------------------------------------------------------------------------
// oceanicTexture — entropy crossfades partials ↔ noise; ipc shapes envelope
// -------------------------------------------------------------------------
SynthDef(\oceanicTexture, { |out=0, freq=200, amp=0.3,
    entropy=0.5, ipc=0.8, spread=0.5, gate=1, lagTime=1.5|
    var tonal, noisy, sig, env;

    // Smooth section transitions — prevents clicks at boundary crossings
    freq = Lag.kr(freq, lagTime);
    amp  = Lag.kr(amp,  lagTime);
    entropy = Lag.kr(entropy, lagTime);
    ipc  = Lag.kr(ipc,  lagTime);
    spread = Lag.kr(spread, lagTime);

    // Tonal: 8 additive partials; ipc drives detuning jitter
    tonal = Mix.fill(8, { |i|
        var h = i + 1;
        SinOsc.ar(
            freq * h * LFNoise2.kr((1 - ipc) * 3 + 0.05).range(0.993, 1.007),
            Rand(0, 2 * pi),
            (0.4 / h.sqrt) * (1 - entropy)
        )
    });

    // Noisy: 5 BPF pink noise bands; band width and density scale with entropy
    noisy = Mix.fill(5, { |i|
        BPF.ar(
            PinkNoise.ar(1),
            (freq * (2 ** (i * 0.5))) * LFNoise1.kr(entropy * 2 + 0.1).range(0.88, 1.12),
            entropy.linlin(0, 1, 0.02, 0.65)
        ) * entropy * 0.3
    });

    sig = Pan2.ar(
        tonal + noisy,
        LFNoise2.kr((1 - ipc) * 0.4 + 0.05).range(spread.neg, spread)
    );

    env = EnvGen.kr(
        Env.asr(
            ipc.linlin(0, 1, 0.05, 2.5),
            1,
            ipc.linlin(0, 1, 0.05, 4.0)
        ), gate, doneAction: 2);

    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// layerVoice — FM with ratio from layer_separation; stereo spread follows sep
// -------------------------------------------------------------------------
SynthDef(\layerVoice, { |out=0, freq=150, amp=0.25,
    layerSep=0.0, ipc=0.8, entropy=0.5, gate=1, lagTime=1.5|
    var sig, env, mod, car1, car2, ratio;

    freq = Lag.kr(freq, lagTime);
    amp  = Lag.kr(amp,  lagTime);
    layerSep = Lag.kr(layerSep, lagTime);
    ipc  = Lag.kr(ipc,  lagTime);
    entropy = Lag.kr(entropy, lagTime);

    // FM ratio: unison (sep=0) → 4th harmonic (sep=1)
    ratio = (layerSep + 0.001).linexp(0.001, 1.0, 1.001, 4.0);

    // Modulator: harshness grows with entropy and separation
    mod = SinOsc.ar(freq * ratio, 0,
        freq * layerSep * (2.0 + (entropy * 2.0)));

    // Two carriers: primary always on; secondary fades in with separation
    car1 = SinOsc.ar(freq + mod, 0, 0.5);
    car2 = SinOsc.ar(freq * ratio + (mod * 0.5), 0,
        layerSep.linlin(0, 0.05, 0, 0.4));

    // Stereo: layer distance literally becomes stereo width
    sig = Pan2.ar(car1, layerSep.neg * 0.55) + Pan2.ar(car2, layerSep * 0.55);
    sig = LPF.ar(sig,
        (1 - entropy).clip(0.001, 1).linexp(0.001, 1, 350, 9000));
    sig = LeakDC.ar(sig);  // FM at low ratios near 1.0 can produce DC offset

    env = EnvGen.kr(
        Env.asr(
            ipc.linlin(0, 1, 0.5, 3.0),
            1,
            ipc.linlin(0, 1, 0.3, 2.0)
        ), gate, doneAction: 2);

    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// onsetHit — one-shot percussive transient, spawned from Routine
// -------------------------------------------------------------------------
SynthDef(\onsetHit, { |out=0, freq=200, amp=0.4, sustain=0.15, brightness=0.5|
    var sig;

    sig =
        SinOsc.ar(freq)
            * EnvGen.ar(Env.perc(0.002, sustain * 0.9),  doneAction: 0) * 0.6
      + SinOsc.ar(freq * 2)
            * EnvGen.ar(Env.perc(0.001, sustain * 0.45), doneAction: 0) * 0.3
      + SinOsc.ar(freq * 0.5)
            * EnvGen.ar(Env.perc(0.004, sustain * 1.3),  doneAction: 0) * 0.35
      + (BPF.ar(WhiteNoise.ar(brightness * 0.4), freq * 3, 0.35)
            * EnvGen.ar(Env.perc(0.001, sustain * 0.25), doneAction: 2));

    Out.ar(out, Pan2.ar(sig * amp, Rand(-0.5, 0.5)));
}).add;

// -------------------------------------------------------------------------
// spectralGrain — GrainBuf from IFFT WAV; pos follows composition time
// -------------------------------------------------------------------------
SynthDef(\spectralGrain, { |out=0, bufnum=0, amp=0.3,
    pos=0.5, rate=1.0, ipc=0.8, entropy=0.5, density=12, gate=1, lagTime=1.5|
    var trigger, grains, env;

    amp = Lag.kr(amp, lagTime);
    pos = Lag.kr(pos, lagTime * 0.5);  // shorter lag for position tracking
    ipc = Lag.kr(ipc, lagTime);
    entropy = Lag.kr(entropy, lagTime);
    density = Lag.kr(density, lagTime);

    trigger = Impulse.kr(density * entropy.linlin(0, 1, 0.6, 2.8));

    grains = GrainBuf.ar(
        numChannels: 2,
        trigger:     trigger,
        dur:         ipc.linlin(0, 1, 0.04, 0.9),
        sndbuf:      bufnum,
        rate:        rate * LFNoise2.kr((1 - ipc) * 3 + 0.05).range(0.94, 1.06),
        // Phase 3.1: increased position jitter from ±5% to ±15% for richer texture
        pos:         pos + LFNoise2.kr(entropy * 2 + 0.1).range(
                         entropy.neg * 0.15, entropy * 0.15),
        pan:         LFNoise2.kr((1 - ipc) * 0.5 + 0.05).range(-0.55, 0.55)
    );

    env = EnvGen.kr(Env.asr(4, 1, 5), gate, doneAction: 2);
    Out.ar(out, grains * env * amp);
}).add;

// -------------------------------------------------------------------------
// deepField — sub + harmonics; entropy drives harmonic richness
// (replaces deepDrone)
// -------------------------------------------------------------------------
SynthDef(\deepField, { |out=0, freq=50, amp=0.3,
    darkness=0.5, movement=0.3, entropy=0.5, gate=1, lagTime=1.5|
    var sub, harmonics, sig, filtered, env;

    freq = Lag.kr(freq, lagTime);
    amp  = Lag.kr(amp,  lagTime);
    darkness = Lag.kr(darkness, lagTime);
    movement = Lag.kr(movement, lagTime);
    entropy = Lag.kr(entropy, lagTime);

    sub = SinOsc.ar(freq, 0, 0.6)
        + SinOsc.ar(freq * 1.002, 0, 0.4)
        + SinOsc.ar(freq * 0.5,   0, 0.3);

    harmonics = Mix.fill(6, { |i|
        var h = i + 2;
        SinOsc.ar(
            freq * h * LFNoise2.kr(0.1).range(0.998, 1.002),
            0,
            ((1 - darkness) / h) * entropy.linlin(0, 1, 0.2, 1.0) * 0.15
        )
    });

    sig = sub + harmonics;

    filtered = LPF.ar(
        sig,
        darkness.clip(0.001, 1).linexp(0.001, 1, 2000, 150)
            * LFNoise2.kr(movement + 0.01).range(0.7, 1.3)
    );

    filtered = filtered + (BrownNoise.ar(0.03) * darkness);

    env = EnvGen.kr(Env.asr(4, 1, 5), gate, doneAction: 2);

    Out.ar(out, Pan2.ar(filtered * env * amp,
        LFNoise2.kr(0.1).range(-0.3, 0.3)));
}).add;

// -------------------------------------------------------------------------
// spectralFreeze — additive partials with freeze control (unchanged)
// -------------------------------------------------------------------------
SynthDef(\spectralFreeze, { |out=0, freq=100, amp=0.2,
    freeze=0.5, shimmer=0.3, gate=1, lagTime=1.5|
    var partials, sig, env;

    freq = Lag.kr(freq, lagTime);
    amp  = Lag.kr(amp,  lagTime);
    freeze = Lag.kr(freeze, lagTime);
    shimmer = Lag.kr(shimmer, lagTime);

    partials = Mix.fill(12, { |i|
        var h = i + 1;
        SinOsc.ar(
            freq * h * LFNoise2.kr(shimmer * 0.1 / h).range(0.998, 1.002),
            Rand(0, 2 * pi),
            0.3 / h
        )
    });

    sig = partials * LFNoise2.kr(freeze.linlin(0, 1, 2, 0.1)).range(0.8, 1);
    sig = sig * (1 + (SinOsc.kr(0.05) * 0.1 * (1 - freeze)));

    env = EnvGen.kr(Env.asr(5, 1, 6), gate, doneAction: 2);

    Out.ar(out, Pan2.ar(sig * env * amp, 0));
}).add;

// -------------------------------------------------------------------------
// migrationSweep — dawn/dusk transition sweeps (unchanged)
// -------------------------------------------------------------------------
SynthDef(\migrationSweep, { |out=0, startFreq=300, endFreq=80, dur=20,
    amp=0.2, direction=1, brightness=0.5|
    var sig, filtered, env, freq;

    freq = XLine.kr(startFreq, endFreq, dur);

    sig = Saw.ar(freq, 0.3)
        + SinOsc.ar(freq, 0, 0.4)
        + Pulse.ar(freq * 0.5, 0.3, 0.2)
        + SinOsc.ar(freq * 2, 0, 0.15);

    filtered = RLPF.ar(sig, freq * brightness.linlin(0, 1, 2, 6), 0.4);
    filtered = filtered + (BPF.ar(PinkNoise.ar(0.1), freq * 2, 0.2) * brightness);

    env = EnvGen.kr(Env.linen(dur * 0.1, dur * 0.8, dur * 0.1), doneAction: 2);

    Out.ar(out, Pan2.ar(filtered * env * amp,
        Line.kr(direction * 0.5, direction.neg * 0.5, dur)));
}).add;

// -------------------------------------------------------------------------
// masterFXv4 — distortion → mod LPF → CombL delay → FreeVerb2 → Limiter
// -------------------------------------------------------------------------
SynthDef(\masterFXv4, { |in=0, out=0, delayMix=0.3, delayFb=0.4,
    reverbMix=0.5, reverbRoom=0.7, distortion=0.3, filterMod=0.5,
    masterVol=0.75|
    var sig, distorted, filtered, delayed, reverbed, filterFreq;

    sig = In.ar(in, 2);

    // Soft-clip distortion (no gain boost — tanh already normalizes to [-1,1])
    distorted = sig.tanh;
    distorted = (sig * (1 - distortion)) + (distorted * distortion);

    filterFreq = filterMod.linexp(0.001, 1, 200, 12000)
        * LFNoise2.kr(0.5).range(0.8, 1.2);
    filtered = LPF.ar(distorted, filterFreq);

    // FIX: parentheses required — SC has no operator precedence (left-to-right).
    // Without parens, delayMix was multiplying the entire signal (dry + wet),
    // making the delay knob act as a master volume.
    delayed = filtered + (CombL.ar(
        filtered, 0.5,
        [0.25, 0.33] * LFNoise2.kr(0.1).range(0.9, 1.1),
        delayFb.linlin(0, 1, 0.3, 2.5)
    ) * delayMix);

    reverbed = FreeVerb2.ar(delayed[0], delayed[1],
        mix: reverbMix, room: reverbRoom, damp: 0.4);

    // HPF to remove sub-harmonic buildup from delay/reverb tails
    reverbed = HPF.ar(reverbed, 35);
    reverbed = Limiter.ar(reverbed, 0.95);

    Out.ar(out, reverbed * masterVol);
}).add;

// -------------------------------------------------------------------------
// directOut — clean pass-through for sub-bass (no delay/reverb, just limiter)
// -------------------------------------------------------------------------
SynthDef(\directOut, { |in=0, out=0, masterVol=0.75|
    var sig;
    sig = In.ar(in, 2);
    sig = Limiter.ar(sig, 0.95);
    Out.ar(out, sig * masterVol);
}).add;

"SynthDefs loaded (v5: oceanicTexture, layerVoice, onsetHit, spectralGrain, deepField, spectralFreeze, migrationSweep, masterFXv4, directOut).".postln;
)

// ============================================================================
// SECTION 5: GUI (v6) — 900 × 580
// ============================================================================

(
~createGUI = {
    var win, makeKnob, makeInfluenceKnob;

    if(~guiWindow.notNil) { ~guiWindow.close };

    ~guiWindow = Window("Sonificació Oceànica v6", Rect(80, 60, 900, 580))
        .background_(Color.gray(0.08))
        .front;

    win = ~guiWindow;

    // --- Helper: standard synthesis knob ---
    makeKnob = { |parent, bounds, label, bus, lo=0, hi=1|
        var spec = ControlSpec(lo, hi);
        StaticText(parent, Rect(bounds.left, bounds.top, bounds.width, 14))
            .string_(label)
            .stringColor_(Color.gray(0.7))
            .align_(\center)
            .font_(Font("Helvetica", 9));
        Knob(parent, Rect(bounds.left + 4, bounds.top + 15, bounds.width - 8, bounds.width - 8))
            .value_(spec.unmap(bus.getSynchronous))
            .color_([Color.gray(0.2), Color.cyan, Color.cyan(0.5), Color.white])
            .action_({ |k| bus.set(spec.map(k.value)) });
    };

    // --- Helper: feature influence knob (yellow accent) ---
    makeInfluenceKnob = { |parent, bounds, label, bus, lo=0, hi=1|
        var spec = ControlSpec(lo, hi);
        StaticText(parent, Rect(bounds.left, bounds.top, bounds.width, 14))
            .string_(label)
            .stringColor_(Color.yellow)
            .align_(\center)
            .font_(Font("Helvetica", 9));
        Knob(parent, Rect(bounds.left + 4, bounds.top + 15, bounds.width - 8, bounds.width - 8))
            .value_(spec.unmap(bus.getSynchronous))
            .color_([Color.gray(0.2), Color.yellow, Color.yellow(0.5), Color.white])
            .action_({ |k| bus.set(spec.map(k.value)) });
    };

    // ========= HEADER BAR =========
    StaticText(win, Rect(10, 6, 400, 22))
        .string_("SONIFICACIÓ OCEÀNICA  v6")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 15));

    // Transport buttons (top right)
    Button(win, Rect(650, 5, 75, 26))
        .states_([
            ["▶ PLAY", Color.black,  Color.new255(50, 220, 80)],
            ["◼ STOP", Color.white,  Color.new255(200, 50, 50)]
        ])
        .font_(Font("Helvetica Bold", 11))
        .action_({ |btn|
            if(btn.value == 1) { ~playComposition.value } { ~stopAll.value };
        });

    Button(win, Rect(730, 5, 75, 26))
        .states_([["⏺ REC", Color.white, Color.gray(0.3)]])
        .font_(Font("Helvetica Bold", 11))
        .action_({
            if(s.isRecording)
                { s.stopRecording; "Stopped recording.".postln }
                { s.record;        "Recording...".postln };
        });

    Button(win, Rect(810, 5, 75, 26))
        .states_([["REFRESH", Color.white, Color.gray(0.3)]])
        .font_(Font("Helvetica", 10))
        .action_({ ~createGUI.value });

    // ========= DEPTH TIMELINE — UserView 400 × 220 =========
    StaticText(win, Rect(10, 34, 400, 14))
        .string_("DEPTH TIMELINE  (colour = entropy: blue=tonal, red=noise)")
        .stringColor_(Color.gray(0.55))
        .font_(Font("Helvetica", 9));

    ~vizView = UserView(win, Rect(10, 48, 400, 220))
        .background_(Color.black)
        .animate_(true)
        .frameRate_(20);

    ~vizShowEntropy = true;

    ~vizView.drawFunc = { |uv|
        var w, h, pct, curY, ex;
        w   = uv.bounds.width;
        h   = uv.bounds.height;
        pct = (~currentTime / ~totalDuration).clip(0, 1);

        // Depth trace
        Pen.strokeColor_(Color.gray(0.35));
        Pen.width_(1.0);
        ~vizN.do { |i|
            var x = (i / (~vizN - 1)) * w;
            var y = ~vizDepthTrace[i].linlin(80, 700, h * 0.06, h * 0.94);
            if(i == 0) { Pen.moveTo(x @ y) } { Pen.lineTo(x @ y) };
        };
        Pen.stroke;

        // Entropy colour overlay (every 4th point)
        if(~vizShowEntropy) {
            (~vizN / 4).asInteger.do { |k|
                var i = k * 4;
                var x = (i / (~vizN - 1)) * w;
                var y = ~vizDepthTrace[i].linlin(80, 700, h * 0.06, h * 0.94);
                var e = ~vizEntropyTrace[i] ? 0;
                Pen.fillColor_(Color.hsv(0.63 - (e * 0.63), 0.8, 0.6, 0.45));
                Pen.addOval(Rect(x - 2.5, y - 2.5, 5, 5));
                Pen.fill;
            };
        };

        // Current position line
        Pen.strokeColor_(Color.new255(0, 220, 100));
        Pen.width_(1.5);
        Pen.line((pct * w) @ 0, (pct * w) @ h);
        Pen.stroke;

        // Current depth dot
        curY = ~depthRawVal.linlin(80, 700, h * 0.06, h * 0.94);
        Pen.fillColor_(Color.new255(0, 255, 120));
        Pen.addOval(Rect(pct * w - 5, curY - 5, 10, 10));
        Pen.fill;

        // Recent onset flashes (orange vertical marks)
        ~recentOnsets.do { |evtPct|
            var ex2 = evtPct * w;
            Pen.strokeColor_(Color.new255(255, 140, 0, 180));
            Pen.width_(1.0);
            Pen.line(ex2 @ 0, ex2 @ (h * 0.28));
            Pen.stroke;
        };
    };

    // Entropy overlay toggle
    Button(win, Rect(10, 272, 120, 18))
        .states_([
            ["Entropy ON",  Color.black,  Color.new255(100, 180, 255)],
            ["Entropy OFF", Color.white,  Color.gray(0.3)]
        ])
        .font_(Font("Helvetica", 9))
        .action_({ |btn| ~vizShowEntropy = (btn.value == 0) });

    // ========= FEATURE METERS — MultiSliderView =========
    StaticText(win, Rect(420, 34, 460, 14))
        .string_("FEATURE METERS  [Depth  Inten  Entr  IPC   Sep   Onset  Velo  Sprd]")
        .stringColor_(Color.gray(0.55))
        .font_(Font("Helvetica", 9));

    ~featureMeters = MultiSliderView(win, Rect(420, 48, 460, 80))
        .size_(8)
        .isFilled_(true)
        .drawLines_(false)
        .elasticMode_(1)
        .value_([0.5, 0.5, 0.5, 0.8, 0, 0, 0.5, 0.3])
        .background_(Color.gray(0.12))
        .fillColor_(Color.new255(0, 200, 255))
        .strokeColor_(Color.gray(0.5));

    // Feature labels below meters
    ["Depth","Inten","Entr","IPC","Sep","Onset","Velo","Sprd"].do { |lbl, i|
        StaticText(win, Rect(420 + (i * 57.5), 128, 55, 12))
            .string_(lbl)
            .stringColor_(Color.gray(0.5))
            .align_(\center)
            .font_(Font("Helvetica", 8));
    };

    // ========= INFO PANEL =========
    ~infoDisplay = StaticText(win, Rect(420, 145, 460, 120))
        .string_("UTC: --:--    Section: --------\n" ++
                 "Depth: --- m   Velocity: --- m/h\n" ++
                 "Entropy: ----  IPC: ----\n" ++
                 "LayerSep: ---- Onset: ----")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 11))
        .background_(Color.gray(0.12));

    ~sectionDisplay = StaticText(win, Rect(420, 268, 460, 24))
        .string_("SECTION: ---")
        .stringColor_(Color.cyan)
        .align_(\center)
        .font_(Font("Helvetica Bold", 14))
        .background_(Color.gray(0.18));

    // ========= CONTROLS ROW =========
    // --- Master ---
    StaticText(win, Rect(10, 300, 80, 14))
        .string_("MASTER")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 10));

    makeKnob.(win, Rect(10,  314, 50, 50), "Vol",   ~ctrl[\masterVol]);
    makeKnob.(win, Rect(65,  314, 50, 50), "Chaos", ~ctrl[\chaos]);

    // --- Synthesis voices ---
    StaticText(win, Rect(130, 300, 200, 14))
        .string_("SYNTHESIS VOICES")
        .stringColor_(Color.cyan)
        .font_(Font("Helvetica Bold", 10));

    makeKnob.(win, Rect(130, 314, 50, 50), "Swarm", ~ctrl[\swarmVol]);
    makeKnob.(win, Rect(185, 314, 50, 50), "Deep",  ~ctrl[\droneVol]);
    makeKnob.(win, Rect(240, 314, 50, 50), "Layer", ~ctrl[\layerVol]);
    makeKnob.(win, Rect(295, 314, 50, 50), "Grain", ~ctrl[\grainVol]);
    makeKnob.(win, Rect(350, 314, 50, 50), "OnVol", ~ctrl[\onsetVol]);

    // Onset threshold slider
    StaticText(win, Rect(410, 300, 80, 14))
        .string_("Onset Thr")
        .stringColor_(Color.new255(255, 160, 50))
        .font_(Font("Helvetica", 9));
    Slider(win, Rect(410, 314, 60, 50))
        .value_(ControlSpec(0.03, 0.5).unmap(~ctrl[\onsetThreshold].getSynchronous))
        .background_(Color.gray(0.2))
        .knobColor_(Color.new255(255, 160, 50))
        .action_({ |sl|
            ~ctrl[\onsetThreshold].set(ControlSpec(0.03, 0.5).map(sl.value));
        });

    // --- Effects ---
    StaticText(win, Rect(480, 300, 120, 14))
        .string_("EFFECTS")
        .stringColor_(Color.magenta)
        .font_(Font("Helvetica Bold", 10));

    makeKnob.(win, Rect(480, 314, 50, 50), "Reverb", ~ctrl[\reverbMix]);
    makeKnob.(win, Rect(535, 314, 50, 50), "Delay",  ~ctrl[\delayMix]);
    makeKnob.(win, Rect(590, 314, 50, 50), "Dist",   ~ctrl[\distortion]);
    makeKnob.(win, Rect(645, 314, 50, 50), "Filter", ~ctrl[\filterMod]);

    // --- Feature influence ---
    StaticText(win, Rect(710, 300, 180, 14))
        .string_("FEATURE INFLUENCE")
        .stringColor_(Color.yellow)
        .font_(Font("Helvetica Bold", 10));

    makeInfluenceKnob.(win, Rect(710, 314, 46, 46), "Entr",  ~ctrl[\entropyMix]);
    makeInfluenceKnob.(win, Rect(758, 314, 46, 46), "IPC",   ~ctrl[\ipcMix]);
    makeInfluenceKnob.(win, Rect(806, 314, 46, 46), "Sep",   ~ctrl[\layerSepMix]);

    // ========= BOTTOM ROW: section jumps + speed + presets =========
    StaticText(win, Rect(10, 378, 100, 14))
        .string_("JUMP TO:")
        .stringColor_(Color.gray(0.6))
        .font_(Font("Helvetica", 9));

    [
        ["◀ NIGHT", Color.new255(80, 180, 255),  0.0       ],
        ["↓ DAWN",  Color.new255(255, 160, 50),  5.75/24.0 ],
        ["▬ DAY",   Color.yellow,                 8.5/24.0  ],
        ["↑ DUSK",  Color.new255(255, 100, 180), 19.5/24.0 ]
    ].do { |item, i|
        var lbl = item[0], col = item[1], hourFrac = item[2];
        Button(win, Rect(10 + (i * 78), 392, 72, 26))
            .states_([[lbl, Color.black, col]])
            .font_(Font("Helvetica Bold", 10))
            .action_({
                // Find first time index where hour_norm >= hourFrac
                var targetIdx = ~hour.detectIndex { |h| h >= hourFrac } ? 0;
                var targetReal = ~times[targetIdx] ? 0;
                ~currentTime = targetReal / ~compression;
                ("Jump to " ++ lbl ++ " — composition time: " ++
                    ~currentTime.round(1) ++ " s").postln;
            });
    };

    // Speed buttons
    StaticText(win, Rect(330, 378, 60, 14))
        .string_("SPEED")
        .stringColor_(Color.gray(0.6))
        .font_(Font("Helvetica", 9));

    ~speedBtns = Array.new(4);
    [0.5, 1.0, 2.0, 4.0].do { |spd, i|
        var lbl = if(spd < 1) { "0.5×" } { spd.asInteger.asString ++ "×" };
        ~speedBtns = ~speedBtns.add(
            Button(win, Rect(330 + (i * 52), 392, 46, 26))
                .states_([
                    [lbl, Color.white, Color.gray(0.28)],
                    [lbl, Color.black, Color.yellow]
                ])
                .font_(Font("Helvetica Bold", 10))
                .value_(if(spd == 1.0) { 1 } { 0 })
                .action_({ |btn|
                    if(btn.value == 1) {
                        // Deselect others
                        ~speedBtns.do { |b| if(b !== btn) { b.value_(0) } };
                        ~speed = spd;
                        ~totalDuration = ~times.last / (~times.last / 240) / ~speed;
                        ~compression   = ~times.last / ~totalDuration;
                        ("Speed: " ++ lbl ++ " — compression: " ++ ~compression.round(1) ++ "x").postln;
                    };
                })
        );
    };

    // Presets
    StaticText(win, Rect(555, 378, 60, 14))
        .string_("PRESETS")
        .stringColor_(Color.gray(0.6))
        .font_(Font("Helvetica", 9));

    Button(win, Rect(555, 392, 72, 26))
        .states_([["NIGHT", Color.white, Color.blue(0.5)]])
        .font_(Font("Helvetica Bold", 10))
        .action_({ ~presetNight.value });

    Button(win, Rect(632, 392, 72, 26))
        .states_([["DAY", Color.black, Color.yellow(0.7)]])
        .font_(Font("Helvetica Bold", 10))
        .action_({ ~presetDay.value });

    Button(win, Rect(709, 392, 72, 26))
        .states_([["CHAOS", Color.white, Color.red(0.7)]])
        .font_(Font("Helvetica Bold", 10))
        .action_({ ~presetChaos.value });

    Button(win, Rect(786, 392, 72, 26))
        .states_([["CLEAN", Color.black, Color.new255(200, 255, 200)]])
        .font_(Font("Helvetica Bold", 10))
        .action_({ ~presetClean.value });

    // ========= CREDITS =========
    StaticText(win, Rect(10, 428, 880, 40))
        .string_(
            "SONIFICACIÓ OCEÀNICA v6 — Oceànica (Alex Cabrer & Joan Sala) | CLT019 Generalitat de Catalunya\n" ++
            "MALASPINA 2010 — Atlantic Ocean — January 26, 2011 — 38 kHz · 29,997 pings · DVM calibrated"
        )
        .stringColor_(Color.gray(0.4))
        .font_(Font("Helvetica", 9));

    win.onClose = {
        ~stopAll.value;
        if(~ifftBuf.notNil) { ~ifftBuf.free };
        "Window closed. Buffer freed.".postln;
    };

    "GUI v6 created.".postln;
};

~createGUI.value;
)

// ============================================================================
// SECTION 6: DATA ACCESS FUNCTIONS
// ============================================================================

(
// Windowed mean — returns mean of data array within a time window
~getValue = { |dataArray, timeArray, compTime, compression, windowSec=2|
    var realTime, startIdx, endIdx, values;
    realTime = compTime * compression;
    startIdx = (timeArray.indexOfGreaterThan(realTime - (windowSec * compression * 0.5)) ? 0).max(0);
    endIdx   = (timeArray.indexOfGreaterThan(realTime + (windowSec * compression * 0.5)) ? (timeArray.size - 1)).min(timeArray.size - 1);
    values   = dataArray.copyRange(startIdx, endIdx);
    values   = values.reject({ |v| v.isNil or: { v.isNaN } });
    if(values.size > 0) { values.mean } { 0.5 };
};

// Nearest-sample lookup — for display only
~getRaw = { |dataArray, timeArray, compTime, compression|
    var realTime, idx;
    realTime = compTime * compression;
    idx = (timeArray.indexOfGreaterThan(realTime) ? (timeArray.size - 1)).max(0).min(dataArray.size - 1);
    dataArray[idx] ? 0;
};

"Data functions ready.".postln;
)

// ============================================================================
// SECTION 7: COMPOSITION ENGINE (v6)
// ============================================================================

(
~fxBus     = Bus.audio(s, 2);  // delay + reverb chain
~directBus = Bus.audio(s, 2);  // clean pass-through (sub-bass)

~composition = Routine({
    var updateRate = 0.05;
    var currentTime = 0;

    // Persistent voices
    var textureSynth, layerSynth, grainSynth, deepSynth, freezeSynth;
    var fxSynth, directSynth;
    var bufToUse;

    // Current data values
    var depthVal, intensityVal, spreadVal, layersVal, velocityVal,
        hourVal, anomalyVal, entropyVal, ipcVal, layerSepVal, onsetVal;
    var depthRawVal, velRawVal;

    // Effective (scaled by influence knobs)
    var eff_entropy, eff_ipc, eff_layerSep;

    // Section tracking
    var section = \night1, lastSection = \none;
    var sectionName, bioState;

    // Onset cooldown (avoid re-firing on same event)
    var lastOnsetTime = -1.0;
    var onsetCooldown = 0.15 / ~speed;   // scaled by speed — faster playback = denser onsets

    // Section transition hysteresis — prevent jitter at boundaries
    var sectionHoldCount = 0;
    var candidateSection = \none;

    "".postln;
    "╔══════════════════════════════════════════════════════════════════╗".postln;
    "║   SONIFICACIÓ OCEÀNICA v6 — Professional Mix & Signal Chain     ║".postln;
    "║   Features: entropy · IPC · layerSep · onset                   ║".postln;
    "╚══════════════════════════════════════════════════════════════════╝".postln;

    // Resolve buffer number safely
    bufToUse = if(~ifftBuf.notNil) { ~ifftBuf.bufnum } { 0 };
    ("IFFT buffer: " ++ bufToUse).postln;

    // Start FX chains (tail position — after all source synths)
    fxSynth     = Synth(\masterFXv4, [\in, ~fxBus, \out, 0], addAction: \addToTail);
    directSynth = Synth(\directOut,  [\in, ~directBus, \out, 0], addAction: \addToTail);

    // Start all persistent voices silently
    // Sub-bass (deepField) routes to directBus — clean, no delay/reverb
    // All other voices route to fxBus — delay + reverb processing
    textureSynth = Synth(\oceanicTexture, [\out, ~fxBus, \amp, 0, \gate, 1]);
    layerSynth   = Synth(\layerVoice,    [\out, ~fxBus, \amp, 0, \gate, 1]);
    deepSynth    = Synth(\deepField,     [\out, ~directBus, \amp, 0, \gate, 1]);
    grainSynth   = Synth(\spectralGrain, [\out, ~fxBus, \amp, 0, \gate, 1,
                                          \bufnum, bufToUse]);
    freezeSynth  = Synth(\spectralFreeze,[\out, ~fxBus, \amp, 0, \gate, 1]);

    0.3.wait;

    while({ currentTime < ~totalDuration }, {

        // --- Retrieve all data values (window scaled by speed for consistent coverage) ---
        depthVal     = ~getValue.(~depth,    ~times, currentTime, ~compression, 1.5 * ~speed);
        intensityVal = ~getValue.(~intensity,~times, currentTime, ~compression, 1.5 * ~speed);
        spreadVal    = ~getValue.(~spread,   ~times, currentTime, ~compression, 1.5 * ~speed);
        layersVal    = ~getValue.(~layers,   ~times, currentTime, ~compression, 1.5 * ~speed);
        velocityVal  = ~getValue.(~velocity, ~times, currentTime, ~compression, 1.5 * ~speed);
        hourVal      = ~getValue.(~hour,     ~times, currentTime, ~compression, 1.0 * ~speed);
        anomalyVal   = ~getValue.(~anomaly,  ~times, currentTime, ~compression, 1.5 * ~speed);
        entropyVal   = ~getValue.(~entropy,  ~times, currentTime, ~compression, 1.5 * ~speed);
        ipcVal       = ~getValue.(~ipc,      ~times, currentTime, ~compression, 1.5 * ~speed);
        layerSepVal  = ~getValue.(~layerSep, ~times, currentTime, ~compression, 1.5 * ~speed);
        onsetVal     = ~getValue.(~onset,    ~times, currentTime, ~compression, 0.5 * ~speed);

        depthRawVal  = ~getRaw.(~depthRaw,   ~times, currentTime, ~compression);
        velRawVal    = ~getRaw.(~velocityRaw,~times, currentTime, ~compression);

        // --- Scale by influence controls ---
        eff_entropy  = (entropyVal  * ~ctrl[\entropyMix].getSynchronous).clip(0, 1);
        eff_ipc      = (ipcVal      * ~ctrl[\ipcMix].getSynchronous).clip(0, 1);
        eff_layerSep = (layerSepVal * ~ctrl[\layerSepMix].getSynchronous).clip(0, 1);

        // --- Section logic (calibrated DVM thresholds + hysteresis) ---
        candidateSection = case
            { hourVal < (5.75  / 24) } { \night1 }
            { hourVal < (8.5   / 24) } { \dawn   }
            { hourVal < (19.5  / 24) } { \day    }
            { hourVal < (20.25 / 24) } { \dusk   }
            { true }                   { \night2 };

        // Require 3+ consecutive ticks in new section before switching
        // (prevents jitter at boundaries from triggering repeated sweeps)
        if(candidateSection != section) {
            sectionHoldCount = sectionHoldCount + 1;
            if(sectionHoldCount >= 3) {
                section = candidateSection;
                sectionHoldCount = 0;
            };
        } {
            sectionHoldCount = 0;
        };

        #sectionName, bioState = case
            { section == \night1 } { ["NIGHT (Early)", "Surface feeding"] }
            { section == \dawn   } { ["DAWN",          "Descending to depth"] }
            { section == \day    } { ["DAY",           "Deep refuge"] }
            { section == \dusk   } { ["DUSK",          "Ascending to surface"] }
            { true                } { ["NIGHT (Late)", "Surface activity"] };

        // ===== SECTION SYNTHESIS =====

        case

        // ---- NIGHT (night1, night2): texture dominant ----
        { section == \night1 or: { section == \night2 } } {
            var swarmVol = ~ctrl[\swarmVol].getSynchronous;
            var droneVol = ~ctrl[\droneVol].getSynchronous;
            var layerVol = ~ctrl[\layerVol].getSynchronous;

            textureSynth.set(
                \freq,    depthVal.clip(0.001, 1).linexp(0.001, 1, 200, 1800),
                \amp,     intensityVal.linlin(0, 1, 0.2, 0.65) * swarmVol,
                \entropy, eff_entropy,
                \ipc,     eff_ipc,
                \spread,  spreadVal.linlin(0, 1, 0.2, 0.8)
            );

            layerSynth.set(
                \freq,     depthVal.clip(0.001, 1).linexp(0.001, 1, 100, 500),
                \amp,      layersVal.linlin(0, 1, 0.05, 0.3) * layerVol,
                \layerSep, eff_layerSep,
                \ipc,      eff_ipc,
                \entropy,  eff_entropy
            );

            deepSynth.set(
                \freq,     depthVal.clip(0.001, 1).linexp(0.001, 1, 30, 65),
                \amp,      0.15 * droneVol,
                \darkness, 0.3,
                \movement, (1 - eff_ipc) * 0.5 + 0.01,
                \entropy,  eff_entropy
            );

            grainSynth.set(\amp, 0);
            freezeSynth.set(\amp, 0);
        }

        // ---- DAWN: layerVoice dominant (FM tension) ----
        { section == \dawn } {
            var swarmVol = ~ctrl[\swarmVol].getSynchronous;
            var droneVol = ~ctrl[\droneVol].getSynchronous;
            var layerVol = ~ctrl[\layerVol].getSynchronous;

            if(lastSection != \dawn) {
                Synth(\migrationSweep, [\out, ~fxBus,
                    \startFreq, 400, \endFreq, 80,
                    \dur, 18, \amp, 0.2, \direction, 1, \brightness, 0.6]);
                ">>> DAWN: Descent sweep triggered <<<".postln;
            };

            layerSynth.set(
                \freq,     depthVal.clip(0.001, 1).linexp(0.001, 1, 120, 450),
                \amp,      0.35 * layerVol,
                \layerSep, eff_layerSep,
                \ipc,      eff_ipc,
                \entropy,  (eff_entropy * 1.3).clip(0, 1)
            );

            textureSynth.set(
                \freq,    depthVal.clip(0.001, 1).linexp(0.001, 1, 250, 800),
                \amp,     0.15 * swarmVol,
                \entropy, (eff_entropy * 0.5).clip(0, 1),
                \ipc,     eff_ipc,
                \spread,  spreadVal.linlin(0, 1, 0.15, 0.55)
            );

            deepSynth.set(
                \freq,     depthVal.clip(0.001, 1).linexp(0.001, 1, 28, 60),
                \amp,      0.35 * droneVol,
                \darkness, depthVal.linlin(0, 1, 0.3, 0.85),
                \movement, 0.3,
                \entropy,  eff_entropy
            );

            grainSynth.set(\amp, 0);
            freezeSynth.set(\amp, 0);
        }

        // ---- DAY: spectralGrain dominant + deepField + freeze ----
        { section == \day } {
            var grainVol = ~ctrl[\grainVol].getSynchronous;
            var droneVol = ~ctrl[\droneVol].getSynchronous;
            var swarmVol = ~ctrl[\swarmVol].getSynchronous;

            grainSynth.set(
                \pos,     (currentTime / ~totalDuration).clip(0.001, 0.999),
                \amp,     0.4 * grainVol,
                \ipc,     eff_ipc,
                \entropy, eff_entropy,
                \density, 8 + (eff_entropy * 20)
            );

            freezeSynth.set(
                \freq,    depthVal.clip(0.001, 1).linexp(0.001, 1, 80, 200),
                \amp,     0.3 * droneVol,
                \freeze,  (eff_ipc * 0.9).clip(0, 1),
                \shimmer, eff_entropy * 0.4
            );

            deepSynth.set(
                \freq,     depthVal.clip(0.001, 1).linexp(0.001, 1, 25, 60),
                \amp,      0.45 * droneVol,
                \darkness, 0.85,
                \movement, ((1 - eff_ipc) * 0.15) + 0.01,
                \entropy,  eff_entropy
            );

            textureSynth.set(\amp, 0.05 * swarmVol);
            layerSynth.set(\amp, 0);
        }

        // ---- DUSK: texture ascending + layerVoice + sweep ----
        { section == \dusk } {
            var swarmVol = ~ctrl[\swarmVol].getSynchronous;
            var droneVol = ~ctrl[\droneVol].getSynchronous;
            var layerVol = ~ctrl[\layerVol].getSynchronous;

            if(lastSection != \dusk) {
                Synth(\migrationSweep, [\out, ~fxBus,
                    \startFreq, 80, \endFreq, 400,
                    \dur, 18, \amp, 0.2, \direction, -1, \brightness, 0.7]);
                ">>> DUSK: Ascent sweep triggered <<<".postln;
            };

            textureSynth.set(
                \freq,    depthVal.clip(0.001, 1).linexp(0.001, 1, 250, 1800),
                \amp,     intensityVal.linlin(0, 1, 0.2, 0.55) * swarmVol,
                \entropy, (eff_entropy * 0.7).clip(0, 1),
                \ipc,     eff_ipc,
                \spread,  spreadVal.linlin(0, 1, 0.4, 0.9)
            );

            layerSynth.set(
                \freq,     depthVal.clip(0.001, 1).linexp(0.001, 1, 120, 500),
                \amp,      0.25 * layerVol,
                \layerSep, eff_layerSep,
                \ipc,      eff_ipc,
                \entropy,  eff_entropy
            );

            deepSynth.set(
                \amp,      0.2 * droneVol,
                \darkness, depthVal.linlin(0, 1, 0.6, 0.2),
                \movement, 0.4,
                \entropy,  eff_entropy
            );

            grainSynth.set(\amp, 0);
            freezeSynth.set(\amp, 0);
        };

        lastSection = section;

        // ===== ONSET DETECTION — fires in every section =====
        if((onsetVal > ~ctrl[\onsetThreshold].getSynchronous) and:
           { (currentTime - lastOnsetTime) > onsetCooldown }) {
            lastOnsetTime = currentTime;
            Synth(\onsetHit, [
                \out,        ~fxBus,
                \freq,       depthVal.clip(0.001, 1).linexp(0.001, 1, 70, 600),
                \amp,        onsetVal * 0.55 * ~ctrl[\onsetVol].getSynchronous,
                \sustain,    onsetVal.linlin(0, 1, 0.04, 0.5),
                \brightness, eff_entropy
            ]);
            // Record onset position for viz ring buffer
            ~recentOnsets = (~recentOnsets.add(currentTime / ~totalDuration)).keep(-20);
        };

        // ===== EFFECTS BUS UPDATE =====
        fxSynth.set(
            \delayMix,   ~ctrl[\delayMix].getSynchronous,
            \delayFb,    ~ctrl[\feedback].getSynchronous,
            \reverbMix,  ~ctrl[\reverbMix].getSynchronous,
            \reverbRoom, case
                { section == \day   } { 0.92 }
                { section == \night1 or: { section == \night2 } } { 0.52 }
                { true } { 0.72 },
            \distortion, ~ctrl[\distortion].getSynchronous * case
                { section == \dawn } { 1.5 }
                { section == \day  } { 0.3 }
                { true } { 1.0 },
            \filterMod,  ~ctrl[\filterMod].getSynchronous,
            \masterVol,  ~ctrl[\masterVol].getSynchronous
        );
        // Direct bus (sub-bass) gets same master volume
        directSynth.set(\masterVol, ~ctrl[\masterVol].getSynchronous);

        // ===== GUI UPDATE (deferred to AppClock) =====
        {
            var mins    = (currentTime / 60).floor.asInteger;
            var secs    = (currentTime % 60).floor.asInteger;
            var rHour   = (hourVal * 24).floor.asInteger;
            var rMin    = ((hourVal * 24 * 60) % 60).floor.asInteger;
            var velDir  = if(velRawVal > 0) { " (asc)" } { " (dsc)" };

            // Store shared state for UserView
            ~currentTime = currentTime;
            ~depthRawVal = depthRawVal;

            if(~infoDisplay.notNil and: { ~infoDisplay.isClosed.not }) {
                ~infoDisplay.string_(
                    "UTC: "  ++ rHour.asString.padLeft(2, "0") ++ ":"
                              ++ rMin.asString.padLeft(2,  "0")
                    ++ "    Section: " ++ sectionName ++ "\n"
                    ++ "Depth: "   ++ depthRawVal.round(1)    ++ " m"
                    ++ "   Velocity: " ++ velRawVal.round(1)  ++ " m/h" ++ velDir ++ "\n"
                    ++ "Entropy: " ++ entropyVal.round(0.001) ++ "  IPC: " ++ ipcVal.round(0.001) ++ "\n"
                    ++ "LayerSep: " ++ layerSepVal.round(0.001) ++ "  Onset: " ++ onsetVal.round(0.001)
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

            if(~featureMeters.notNil and: { ~featureMeters.isClosed.not }) {
                ~featureMeters.value_([
                    depthVal, intensityVal, entropyVal, ipcVal,
                    layerSepVal, onsetVal,
                    velocityVal.clip(0, 1), spreadVal
                ]);
            };
        }.defer;

        updateRate.wait;
        currentTime = currentTime + updateRate;
    });

    // === FADE OUT ===
    "Composition ending — fading out...".postln;
    [textureSynth, layerSynth, deepSynth, grainSynth, freezeSynth].do(_.set(\gate, 0));
    5.wait;
    fxSynth.free;
    directSynth.free;

    { if(~sectionDisplay.notNil and: { ~sectionDisplay.isClosed.not }) {
        ~sectionDisplay.string_("COMPLETE  —  24h cycle finished");
        ~sectionDisplay.stringColor_(Color.white);
    } }.defer;

    "Done.".postln;
});

~playComposition = {
    ~composition.reset;
    ~composition.play;
    "Composition started.".postln;
};

~stopAll = {
    ~composition.stop;
    s.freeAll;
    ~recentOnsets = Array.new(20);
    "Stopped. All synths freed.".postln;
};

"Composition engine v6 ready.".postln;
)

// ============================================================================
// SECTION 8: PRESETS
// ============================================================================

(
~presetNight = {
    ~ctrl[\masterVol].set(0.75);
    ~ctrl[\chaos].set(0.6);
    ~ctrl[\swarmVol].set(0.7);
    ~ctrl[\droneVol].set(0.3);
    ~ctrl[\layerVol].set(0.45);
    ~ctrl[\grainVol].set(0.1);
    ~ctrl[\onsetVol].set(0.55);
    ~ctrl[\onsetThreshold].set(0.12);
    ~ctrl[\entropyMix].set(1.0);
    ~ctrl[\ipcMix].set(1.0);
    ~ctrl[\layerSepMix].set(1.0);
    ~ctrl[\distortion].set(0.3);
    ~ctrl[\feedback].set(0.35);
    ~ctrl[\reverbMix].set(0.45);
    ~ctrl[\delayMix].set(0.3);
    ~ctrl[\filterMod].set(0.7);
    "Preset: NIGHT — texture + layer + onset active".postln;
};

~presetDay = {
    ~ctrl[\masterVol].set(0.7);
    ~ctrl[\chaos].set(0.2);
    ~ctrl[\swarmVol].set(0.1);
    ~ctrl[\droneVol].set(0.75);
    ~ctrl[\layerVol].set(0.1);
    ~ctrl[\grainVol].set(0.65);
    ~ctrl[\onsetVol].set(0.2);
    ~ctrl[\onsetThreshold].set(0.3);
    ~ctrl[\entropyMix].set(1.0);
    ~ctrl[\ipcMix].set(1.0);
    ~ctrl[\layerSepMix].set(0.4);
    ~ctrl[\distortion].set(0.1);
    ~ctrl[\feedback].set(0.5);
    ~ctrl[\reverbMix].set(0.75);
    ~ctrl[\delayMix].set(0.4);
    ~ctrl[\filterMod].set(0.3);
    "Preset: DAY — granular IFFT + deep + freeze".postln;
};

~presetChaos = {
    ~ctrl[\masterVol].set(0.8);
    ~ctrl[\chaos].set(0.9);
    ~ctrl[\swarmVol].set(0.8);
    ~ctrl[\droneVol].set(0.5);
    ~ctrl[\layerVol].set(0.7);
    ~ctrl[\grainVol].set(0.5);
    ~ctrl[\onsetVol].set(0.7);
    ~ctrl[\onsetThreshold].set(0.05);
    ~ctrl[\entropyMix].set(1.0);
    ~ctrl[\ipcMix].set(0.2);
    ~ctrl[\layerSepMix].set(1.0);
    ~ctrl[\distortion].set(0.65);
    ~ctrl[\feedback].set(0.6);
    ~ctrl[\reverbMix].set(0.5);
    ~ctrl[\delayMix].set(0.55);
    ~ctrl[\filterMod].set(0.65);
    "Preset: CHAOS — maximum intensity, low IPC, dense onsets".postln;
};

~presetClean = {
    // All feature influence to 0 — pure section-based synthesis, no data modulation
    ~ctrl[\masterVol].set(0.7);
    ~ctrl[\chaos].set(0.5);
    ~ctrl[\swarmVol].set(0.5);
    ~ctrl[\droneVol].set(0.5);
    ~ctrl[\layerVol].set(0.4);
    ~ctrl[\grainVol].set(0.5);
    ~ctrl[\onsetVol].set(0.0);
    ~ctrl[\onsetThreshold].set(0.5);
    ~ctrl[\entropyMix].set(0.0);
    ~ctrl[\ipcMix].set(0.0);
    ~ctrl[\layerSepMix].set(0.0);
    ~ctrl[\distortion].set(0.2);
    ~ctrl[\feedback].set(0.3);
    ~ctrl[\reverbMix].set(0.55);
    ~ctrl[\delayMix].set(0.3);
    ~ctrl[\filterMod].set(0.5);
    "Preset: CLEAN — no feature modulation (A/B comparison baseline)".postln;
};

"Presets ready (NIGHT | DAY | CHAOS | CLEAN).".postln;
)

// ============================================================================
// END
// ============================================================================

/*
HOW TO USE (v6):
  1. Execute sections 1–8 in order (Cmd+Enter each block)
  2. Press PLAY or run ~playComposition.value
  3. Use GUI knobs to shape the sound in real-time

FEATURE MAPPING:
  acoustic_entropy        → oceanicTexture: 0=pure partials, 1=pink noise bands
  inter_ping_correlation  → grain duration, envelope times, detuning width
  layer_separation_norm   → layerVoice FM ratio + stereo spread (0=unison, 1=wide 4th)
  onset_strength_norm     → onsetHit one-shot transients (threshold adjustable)

SIGNAL ROUTING (v6):
  deepField (25–65 Hz)    → directBus → Limiter (clean sub-bass, no FX)
  all other voices         → fxBus    → distortion → LPF → delay → reverb → HPF → Limiter

SECTION TIMING (calibrated DVM, South Atlantic 25°S Jan 2011):
  night1  00:00 – 05:45  surface feeding
  dawn    05:45 – 08:30  deep descent (sweep triggered)
  day     08:30 – 19:30  deep refuge (IFFT buffer granular)
  dusk    19:30 – 20:15  ascent (sweep triggered)
  night2  20:15 – 24:00  surface activity returns

VERIFICATION:
  ~ifftBuf.numFrames.postln     // should be > 0
  ~entropy.mean.postln          // ≈ 0.757
  ~ipc.mean.postln              // ≈ 0.780
  ~onset.maxItem.postln         // ≈ 1.0 (normalised)
*/
