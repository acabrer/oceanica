/*
================================================================================
SONIFICACIÓ OCEÀNICA - Version 7: EVENT-DRIVEN COMPOSITION
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya

v7 DESIGN PRINCIPLES:
  1. DATA-DRIVEN: Parameter ranges calibrated to actual data corridors, not hypothetical 0-1 extremes
  2. TIMBRAL IDENTITY: Each voice has distinct sonic character (SinOsc+Pulse, Saw+filter, etc.)
  3. DEPTH AS SPECTRUM: 32-band filter bank with kurtosis-driven Q from depth histogram
  4. NARRATIVE ARC: Tension system aligned with DVM cycle; FX respond to data entropy
  5. EDUCATIONAL: Solo/mute per voice, vertical structure display, 12 feature meters

VOICE ARCHITECTURE:
  surfaceTexture  (400-2200 Hz) — epipelagic scattering (10-150m); persistent, gradient-driven brightness
  deepDrone       (25-120 Hz)   — mesopelagic refuge (400-800m); acceleration vibrato + autocorr-modulated drift
  migrationVoice  (80-800 Hz)   — DVM movement; persistent, continuous amp from velocity deviation
  histogramFilter (50-5000 Hz)  — 32-band resonant filter, kurtosis-driven Q, midwater excitation rate
  spectralGrain   (IFFT buffer) — granular texture, stable velocity + autocorrelation density, outlier amp boost
  onsetPercussion (70-600 Hz)   — transients with depth-zone timbral variety, outlier-lowered threshold
  infraDrone      (5-15 Hz)     — sub-audible acceleration layer (felt, not heard); bypasses voice controls

SIGNAL ROUTING:
  deepDrone + infraDrone → directBus → Limiter (clean sub-bass, no FX)
  all others             → fxBus → dist → LPF(400-16kHz) → delay(fb≤1.5) → reverb(damp=0.3) → HPF → Limiter
  muteAll gate on both FX + direct outputs (kills reverb/delay tails when all voices muted)

DATA: Reads sonification_sc_v5_{date}.json (v4 fields + depth_histogram, skewness, kurtosis,
      vertical_gradient, peak properties, velocity_stable, outlier, autocorr_3min, autocorr_10min)

SECTION TIMING (calibrated DVM, South Atlantic 25°S Jan 2011):
  night1  00:00 – 05:45 UTC   surface feeding
  dawn    05:45 – 08:30 UTC   deep descent
  day     08:30 – 19:30 UTC   deep refuge
  dusk    19:30 – 20:15 UTC   ascent
  night2  20:15 – 24:00 UTC   surface activity
================================================================================
*/

// ============================================================================
// SECTION 1: SERVER BOOT + BUFFER LOAD
// ============================================================================

(
s.options.numOutputBusChannels = 2;
s.options.numInputBusChannels  = 0;
s.options.memSize     = 2.pow(20);
s.options.sampleRate  = 48000;
s.options.numBuffers  = 8192;

~ifftBuf = nil;
~dataPath = thisProcess.nowExecutingPath.dirname.dirname +/+ "output/data/";
// Parameterized — change date here for different datasets
~dateStr = "20110126";
~ifftWavPath = ~dataPath ++ "echogram_audio_" ++ ~dateStr ++ "_ifft.wav";

s.waitForBoot({
    "Server ready. Loading IFFT buffer...".postln;
    ~ifftBuf = Buffer.read(s, ~ifftWavPath, action: { |buf|
        ("IFFT buffer loaded: " ++ buf.numFrames ++ " frames, "
            ++ buf.numChannels ++ " ch").postln;
    });
});
)

// ============================================================================
// SECTION 2: LOAD V5 DATA + PRE-COMPUTE DISPLAY TRACES
// ============================================================================

(
var jsonFile, jsonString;

// Load v5 JSON
jsonFile   = File(~dataPath ++ "sonification_sc_v5_" ++ ~dateStr ++ ".json", "r");
jsonString = jsonFile.readAllString;
jsonFile.close;
~jsonData  = jsonString.parseJSON;

// --- v4 arrays (unchanged) ---
~times      = ~jsonData["38kHz"]["time_seconds"].asArray.collect(_.asFloat);
~depth      = ~jsonData["38kHz"]["depth_norm"].asArray.collect(_.asFloat);
~intensity  = ~jsonData["38kHz"]["intensity_norm"].asArray.collect(_.asFloat);
~spread     = ~jsonData["38kHz"]["spread_norm"].asArray.collect(_.asFloat);
~layers     = ~jsonData["38kHz"]["layers_norm"].asArray.collect(_.asFloat);
~velocity   = ~jsonData["38kHz"]["velocity_norm"].asArray.collect(_.asFloat);
~anomaly    = ~jsonData["38kHz"]["anomaly_norm"].asArray.collect(_.asFloat);
~hour       = ~jsonData["38kHz"]["hour_norm"].asArray.collect(_.asFloat);
~entropy    = ~jsonData["38kHz"]["acoustic_entropy"].asArray.collect(_.asFloat);
~ipc        = ~jsonData["38kHz"]["inter_ping_correlation"].asArray.collect(_.asFloat);
~layerSep   = ~jsonData["38kHz"]["layer_separation_norm"].asArray.collect(_.asFloat);
~onset      = ~jsonData["38kHz"]["onset_strength_norm"].asArray.collect(_.asFloat);
~surfaceN   = ~jsonData["38kHz"]["surface_norm"].asArray.collect(_.asFloat);
~midwaterN  = ~jsonData["38kHz"]["midwater_norm"].asArray.collect(_.asFloat);
~deepN      = ~jsonData["38kHz"]["deep_norm"].asArray.collect(_.asFloat);
~depthRaw   = ~jsonData["38kHz"]["depth_m"].asArray.collect(_.asFloat);
~velocityRaw= ~jsonData["38kHz"]["velocity_m_h"].asArray.collect(_.asFloat);

// --- v5 new arrays ---
~acceleration  = ~jsonData["38kHz"]["acceleration_norm"].asArray.collect(_.asFloat);
~spreadChange  = ~jsonData["38kHz"]["spread_change_norm"].asArray.collect(_.asFloat);
~skewness      = ~jsonData["38kHz"]["skewness_norm"].asArray.collect(_.asFloat);
~kurtosis      = ~jsonData["38kHz"]["kurtosis_norm"].asArray.collect(_.asFloat);
~vertGrad      = ~jsonData["38kHz"]["vertical_gradient_norm"].asArray.collect(_.asFloat);
~peakProm      = ~jsonData["38kHz"]["peak_prominence_norm"].asArray.collect(_.asFloat);
~peakWidth     = ~jsonData["38kHz"]["peak_width_norm"].asArray.collect(_.asFloat);

// --- Tier 2 new arrays ---
~velocityStable = ~jsonData["38kHz"]["velocity_stable_norm"].asArray.collect(_.asFloat);
~outlier      = ~jsonData["38kHz"]["outlier_norm"].asArray.collect(_.asFloat);
~autocorr3    = ~jsonData["38kHz"]["autocorr_3min_norm"].asArray.collect(_.asFloat);
~autocorr10   = ~jsonData["38kHz"]["autocorr_10min_norm"].asArray.collect(_.asFloat);

// --- v5 depth histogram (2D: n_pings × 32) ---
~depthHist = ~jsonData["38kHz"]["depth_histogram"].asArray.collect({ |row|
    row.asArray.collect(_.asFloat)
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
~vizEntropyTrace = Array.fill(~vizN, { |i|
    var idx = ((i / (~vizN - 1)) * (~entropy.size - 1)).round.asInteger.clip(0, ~entropy.size - 1);
    ~entropy[idx] ? 0.5;
});

// --- Pre-compute echogram visualization (400 columns × 32 rows) ---
~vizEchoW = 400;
~vizEchoH = 32;
~vizEchogram = Array.fill(~vizEchoW, { |col|
    var idx = ((col / (~vizEchoW - 1)) * (~depthHist.size - 1)).round.asInteger.clip(0, ~depthHist.size - 1);
    ~depthHist[idx] ? Array.fill(32, 0);
});

// --- Ring buffer for onset markers ---
~recentOnsets = Array.new(20);

// --- Shared GUI state ---
~currentTime  = 0.0;
~depthRawVal  = 400.0;
~currentHist  = Array.fill(32, 0.0);

"".postln;
"=== V5 DATA LOADED ===".postln;
("Points: " ++ ~times.size).postln;
("Duration: " ++ ~times.last.round(1) ++ " s (" ++ (~times.last / 3600).round(0.1) ++ " h)").postln;
("Compression: " ++ ~compression.round(1) ++ "x → " ++ ~totalDuration ++ " s composition").postln;
("Histogram: " ++ ~depthHist.size ++ " pings × " ++ ~depthHist[0].size ++ " bands").postln;
"v5 features: skewness, kurtosis, vertical_gradient, peak_prominence, peak_width, depth_histogram".postln;
)

// ============================================================================
// SECTION 3: CONTROL BUSES
// ============================================================================

(
~ctrl = IdentityDictionary.new;

// Master
~ctrl[\masterVol] = Bus.control(s, 1).set(0.75);

// Voice names for per-voice controls
~voiceNames = [\reef, \abyss, \migration, \filterBank, \grain, \onset];

// Per-voice volume, mute, solo
~voiceVol  = IdentityDictionary.new;
~voiceMute = IdentityDictionary.new;
~voiceSolo = IdentityDictionary.new;

~voiceNames.do { |name|
    ~voiceVol[name]  = Bus.control(s, 1).set(0.5);
    ~voiceMute[name] = Bus.control(s, 1).set(0);  // 0=active, 1=muted
    ~voiceSolo[name] = Bus.control(s, 1).set(0);  // 0=normal, 1=solo
};

// Default voice levels
~voiceVol[\reef].set(0.55);
~voiceVol[\abyss].set(0.5);
~voiceVol[\migration].set(0.5);
~voiceVol[\filterBank].set(0.45);
~voiceVol[\grain].set(0.5);
~voiceVol[\onset].set(0.45);

// Effects (reduced defaults: less reverb/distortion, higher filter cutoff)
~ctrl[\reverbMix]   = Bus.control(s, 1).set(0.35);
~ctrl[\delayMix]    = Bus.control(s, 1).set(0.3);
~ctrl[\distortion]  = Bus.control(s, 1).set(0.15);
~ctrl[\filterMod]   = Bus.control(s, 1).set(0.65);
~ctrl[\feedback]    = Bus.control(s, 1).set(0.3);

// Onset threshold
~ctrl[\onsetThreshold] = Bus.control(s, 1).set(0.15);

// Tension bias (manual override)
~ctrl[\tensionBias] = Bus.control(s, 1).set(0.5);

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

"Control buses ready (6 voices with solo/mute).".postln;
)

// ============================================================================
// SECTION 4: SYNTHDEFS (v7)
// ============================================================================

(
// -------------------------------------------------------------------------
// reefTexture — epipelagic scattering, 10-150m (400-2200 Hz)
// Persistent voice (surface always active in this dataset)
// Driven by: surface_norm (freq + amp), entropy (tonal↔noise),
//            ipc (stability), vertical_gradient (brightness)
// -------------------------------------------------------------------------
SynthDef(\reefTexture, { |out=0, freq=800, amp=0.3,
    entropy=0.7, ipc=0.8, spread=0.5, gradient=0.5, gate=1, lagTime=0.8|
    var tonal, noisy, sig, env, filterFreq;

    freq = Lag.kr(freq, lagTime);
    amp  = Lag.kr(amp, lagTime);
    entropy = Lag.kr(entropy, lagTime);
    ipc  = Lag.kr(ipc, lagTime);
    spread = Lag.kr(spread, lagTime);
    gradient = Lag.kr(gradient, lagTime);

    // Tonal: SinOsc + Pulse harmonics for spectral richness
    tonal = Mix.fill(5, { |i|
        var h = i + 1;
        var detune = LFNoise2.kr((1 - ipc) * 3 + 0.02).range(0.993, 1.007);
        // Alternate SinOsc and Pulse for richer timbre
        if(i.even,
            { SinOsc.ar(freq * h * detune, Rand(0, 2pi),
                (0.3 / h.sqrt) * (1 - entropy).linlin(0, 0.5, 0.1, 1.0)) },
            { Pulse.ar(freq * h * detune, LFNoise2.kr(0.3).range(0.3, 0.7),
                (0.15 / h) * (1 - entropy).linlin(0, 0.5, 0.1, 1.0)) }
        )
    });

    // Crackle: louder, wider — biologically evocative
    noisy = Mix.fill(3, { |i|
        var bpFreq = freq * (1.5 ** i);
        BPF.ar(
            Dust2.ar(entropy.linlin(0.5, 1.0, 8, 120)) * 0.6,
            bpFreq.clip(200, 16000) * LFNoise1.kr(0.3).range(0.85, 1.15),
            entropy.linlin(0.5, 1.0, 0.08, 0.5)
        )
    });

    sig = tonal + noisy;

    // Gradient-driven brightness filter
    filterFreq = gradient.linexp(0.001, 1, 800, 8000);
    sig = RLPF.ar(sig, filterFreq, 0.5);

    sig = Pan2.ar(sig, LFNoise2.kr(0.15).range(spread.neg, spread));

    env = EnvGen.kr(Env.asr(1.5, 1, 2.5), gate, doneAction: 2);
    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// abyssDrone — mesopelagic deep layer, 400-800m (25-120 Hz)
// Driven by: deep_norm (freq + amp), acceleration_norm (wobble + vibrato),
//            kurtosis_norm (harmonics), entropy (filter cutoff)
// Routes to directBus (clean sub-bass, no delay/reverb)
// -------------------------------------------------------------------------
SynthDef(\abyssDrone, { |out=0, freq=50, amp=0.3,
    accel=0.5, filterFreq=200, kurtosis=0.5,
    driftRate=0.05, driftDepth=0.0, gate=1, lagTime=2.0|
    var sub, harmonics, sig, env, vibrato, drift;

    freq = Lag.kr(freq, lagTime);
    amp  = Lag.kr(amp, lagTime);
    accel = Lag.kr(accel, lagTime);
    filterFreq = Lag.kr(filterFreq, lagTime);
    kurtosis = Lag.kr(kurtosis, lagTime);
    driftRate = Lag.kr(driftRate, 4.0);
    driftDepth = Lag.kr(driftDepth, 4.0);

    // Acceleration-driven pitch vibrato
    vibrato = SinOsc.kr(
        accel.linexp(0, 1, 0.05, 4.0),
        0,
        freq * accel.linlin(0, 1, 0.005, 0.03)
    );

    // Sub with beating + vibrato
    sub = SinOsc.ar(freq + vibrato, 0, 0.6)
        + SinOsc.ar((freq + vibrato) * 1.003, 0, 0.35)
        + SinOsc.ar((freq + vibrato) * 0.5, 0, 0.25);

    // Acceleration-driven AM (wider range: ±40%)
    sub = sub * SinOsc.ar(
        accel.linexp(0, 1, 0.03, 3.0),
        0,
        accel.linlin(0, 1, 0.03, 0.25)
    ).range(0.6, 1.0);

    // Kurtosis modulates harmonic content: peaked dist = clean, flat = rich
    harmonics = Mix.fill(4, { |i|
        var h = i + 2;
        SinOsc.ar((freq + vibrato) * h, 0,
            (1 - kurtosis).linlin(0, 1, 0.02, 0.15) / h)
    });

    sig = sub + harmonics;
    // Data-driven filter with slow drift modulation (subtle "breathing" during stable day)
    drift = LFNoise2.kr(driftRate).range(driftDepth.neg, driftDepth);
    sig = LPF.ar(sig, (filterFreq * (1 + drift)).clip(40, 3000));
    sig = sig + (BrownNoise.ar(0.03) * filterFreq.reciprocal.linlin(0, 0.02, 0.3, 1.0));

    env = EnvGen.kr(Env.asr(4, 1, 5), gate, doneAction: 2);
    Out.ar(out, Pan2.ar(sig * env * amp, LFNoise2.kr(0.05).range(-0.25, 0.25)));
}).add;

// -------------------------------------------------------------------------
// migrationVoice — DVM movement (80-800 Hz)
// Driven by: velocity_norm (pitch direction), anomaly_norm (beating),
//            spread_change_norm (resonance), layer_separation_norm (layers)
// Event-gated: sounds when |velocity_norm - 0.5| > 0.06
// -------------------------------------------------------------------------
SynthDef(\migrationVoice, { |out=0, freq=200, amp=0.3,
    velocity=0.5, anomaly=0.5, spreadChange=0.5, layerSep=0.0, gate=1, lagTime=1.0|
    var sig, sig2, env, velDev, detune, rq;

    freq = Lag.kr(freq, lagTime);
    amp  = Lag.kr(amp, lagTime);
    velocity = Lag.kr(velocity, lagTime);
    anomaly = Lag.kr(anomaly, lagTime);
    spreadChange = Lag.kr(spreadChange, lagTime);
    layerSep = Lag.kr(layerSep, lagTime);

    velDev = (velocity - 0.5).abs;

    // Primary: filtered Saw — richer than pure sine
    sig = Saw.ar(freq, 0.4) + SinOsc.ar(freq, 0, 0.3);

    // Anomaly → second oscillator with beating (depth deviation = dissonance)
    detune = anomaly.linlin(0, 1, 0.998, 1.02);
    sig2 = Saw.ar(freq * detune, 0.25) * anomaly.linlin(0.3, 0.7, 0, 0.6);
    sig = sig + sig2;

    // Spread change drives resonance (Q) — wider scatter = more resonant
    rq = spreadChange.linlin(0, 1, 0.6, 0.1);
    sig = RLPF.ar(sig,
        (freq * LFNoise2.kr(velDev.linexp(0.001, 0.5, 0.2, 3.0)).range(1.5, 5.0)).clip(80, 14000),
        rq);

    // Layer separation adds interval voice
    sig = sig + (SinOsc.ar(
        freq * layerSep.linexp(0.001, 1, 1.01, 2.5))
        * layerSep.linlin(0.05, 0.5, 0, 0.35));

    sig = LeakDC.ar(sig);
    // Pan: descending (vel<0.5)=left, ascending (vel>0.5)=right
    sig = Pan2.ar(sig, velocity.linlin(0, 1, -0.6, 0.6));

    env = EnvGen.kr(Env.asr(0.4, 1, 1.2), gate, doneAction: 2);
    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// histogramFilter — vertical ocean structure (50-5000 Hz)
// 32-band resonant filter bank driven by depth histogram
// Each band = a depth stratum; its amplitude tracks acoustic energy there
// -------------------------------------------------------------------------
SynthDef(\histogramFilter, { |out=0, amp=0.3, rq=0.15, kurtosis=0.5,
    exciteRate=15, gate=1, lagTime=0.5|
    var bands = 32;
    var input, filtered, sig, env, effectiveRq;
    var bandAmps = Array.fill(bands, { |i|
        NamedControl.kr(("h" ++ i).asSymbol, 0.0)
    });

    amp = Lag.kr(amp, lagTime);
    kurtosis = Lag.kr(kurtosis, lagTime);

    // Kurtosis-driven Q: peaked distribution (high kurt) = narrow resonant Q;
    // flat distribution (low kurt) = wide blended bands for smoother spectrum
    effectiveRq = rq * kurtosis.linlin(0, 1, 2.5, 0.6);

    // Excitation: organic texture (pink noise + midwater-driven impulsive rate)
    input = PinkNoise.ar(0.4) + (Dust.ar(Lag.kr(exciteRate, lagTime * 2)) * 0.25);

    // 32 resonant filters, log-spaced 50 Hz (deep) → 5000 Hz (surface)
    filtered = Mix.fill(bands, { |i|
        var centerFreq = 50 * (5000/50).pow(i / (bands - 1));
        var bandAmp = Lag.kr(bandAmps[i], lagTime);
        BPF.ar(input, centerFreq, effectiveRq.clip(0.02, 1.0)) * bandAmp * 4.0
    });

    sig = Pan2.ar(filtered, LFNoise2.kr(0.1).range(-0.3, 0.3));

    env = EnvGen.kr(Env.asr(2, 1, 3), gate, doneAction: 2);
    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// spectralGrain — IFFT buffer granular (unchanged from v6)
// -------------------------------------------------------------------------
SynthDef(\spectralGrain, { |out=0, bufnum=0, amp=0.3,
    pos=0.5, rate=1.0, ipc=0.8, entropy=0.5, intensity=0.5,
    density=12, gate=1, lagTime=1.5|
    var trigger, grains, env, ipcSmooth, grainDur;

    amp = Lag.kr(amp, lagTime);
    pos = Lag.kr(pos, lagTime * 0.5);
    ipc = Lag.kr(ipc, lagTime);
    entropy = Lag.kr(entropy, lagTime);
    intensity = Lag.kr(intensity, lagTime);
    density = Lag.kr(density, lagTime);

    // Power curve on IPC to fix bimodality (0.99 vs 0.0):
    // ipc.pow(0.3) compresses the top end, creating more variety in grain duration
    ipcSmooth = ipc.pow(0.3);
    grainDur = ipcSmooth.linlin(0, 1, 0.06, 0.5);

    trigger = Impulse.kr(density * entropy.linlin(0, 1, 0.6, 2.8));

    grains = GrainBuf.ar(
        numChannels: 2,
        trigger:     trigger,
        dur:         grainDur,
        sndbuf:      bufnum,
        rate:        rate * LFNoise2.kr((1 - ipc) * 3 + 0.05).range(0.94, 1.06),
        pos:         pos + LFNoise2.kr(entropy * 2 + 0.1).range(
                         entropy.neg * 0.15, entropy * 0.15),
        pan:         LFNoise2.kr((1 - ipc) * 0.5 + 0.05).range(-0.55, 0.55)
    );

    env = EnvGen.kr(Env.asr(4, 1, 5), gate, doneAction: 2);
    Out.ar(out, grains * env * amp);
}).add;

// -------------------------------------------------------------------------
// onsetPercussion — transients with prominence and width
// -------------------------------------------------------------------------
SynthDef(\onsetPercussion, { |out=0, freq=200, amp=0.4, sustain=0.15,
    brightness=0.5, prominence=0.5, peakWidth=0.3, depthZone=0.5|
    var body, click, resonance, sig, bodyMix, clickMix, pan;

    // depthZone: 0=deep, 1=surface
    // Deep transients: lower body freq, longer sustain, more sub resonance
    // Surface transients: higher freq, sharper click, shorter body
    bodyMix  = 1 - (depthZone * 0.4);   // deep=1.0, surface=0.6
    clickMix = 0.3 + (depthZone * 0.7);  // deep=0.3, surface=1.0

    // Body: sine at depth frequency, sustain scaled by peak width and zone
    body = SinOsc.ar(freq * depthZone.linlin(0, 1, 0.7, 1.3))
        * EnvGen.ar(Env.perc(0.003,
            sustain * peakWidth.linlin(0, 1, 0.5, 2.0) * depthZone.linlin(0, 1, 1.5, 0.7)),
            doneAction: 0) * 0.5 * bodyMix;

    // Click: prominence drives attack sharpness, surface = sharper
    click = BPF.ar(
        WhiteNoise.ar(prominence.linlin(0, 1, 0.15, 0.5)),
        (freq * depthZone.linlin(0, 1, 2, 5)).clip(100, 16000), 0.2
    ) * EnvGen.ar(Env.perc(0.001, 0.03 * (1 - prominence + 0.1)), doneAction: 0)
      * clickMix;

    // Sub resonance for prominent peaks (stronger in deep zone)
    resonance = SinOsc.ar(freq * 0.5)
        * EnvGen.ar(Env.perc(0.01, sustain * 1.5), doneAction: 2)
        * prominence * 0.35 * bodyMix;

    // Pan based on depthZone: deep=center, surface=wider stereo
    pan = Rand(-1, 1) * depthZone.linlin(0, 1, 0.2, 0.7);

    sig = body + click + resonance;
    Out.ar(out, Pan2.ar(sig * amp, pan));
}).add;

// -------------------------------------------------------------------------
// migrationSweep — dawn/dusk transition sweeps (same as v6)
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
// masterFXv7 — spread_change modulates filter LFO rate
// -------------------------------------------------------------------------
SynthDef(\masterFXv7, { |in=0, out=0, delayMix=0.3, delayFb=0.4,
    reverbMix=0.5, reverbRoom=0.7, distortion=0.3, filterMod=0.5,
    spreadChange=0.5, masterVol=0.75, muteAll=0|
    var sig, distorted, filtered, delayed, reverbed, filterFreq, muteGate;

    // When all voices muted, fade output to silence (kills reverb/delay tails)
    muteGate = 1 - Lag.kr(muteAll, 0.5);

    sig = In.ar(in, 2);

    distorted = sig.tanh;
    distorted = (sig * (1 - distortion)) + (distorted * distortion);

    // Wider LPF range: don't cut histogram's high bands (was 200-12000)
    filterFreq = filterMod.linexp(0.001, 1, 400, 16000)
        * LFNoise2.kr(spreadChange.linlin(0, 1, 0.2, 4.0)).range(0.7, 1.3);
    filtered = LPF.ar(distorted, filterFreq.clip(50, 18000));

    // Capped delay feedback: prevent runaway (was 0.3-2.5)
    delayed = filtered + (CombL.ar(
        filtered, 0.5,
        [0.25, 0.33] * LFNoise2.kr(0.1).range(0.9, 1.1),
        delayFb.linlin(0, 1, 0.2, 1.5)
    ) * delayMix);

    // Slightly brighter reverb: damp 0.3 (was 0.4)
    reverbed = FreeVerb2.ar(delayed[0], delayed[1],
        mix: reverbMix, room: reverbRoom, damp: 0.3);

    reverbed = HPF.ar(reverbed, 35);
    reverbed = Limiter.ar(reverbed, 0.95);

    Out.ar(out, reverbed * masterVol * muteGate);
}).add;

// -------------------------------------------------------------------------
// directOut — clean pass-through for sub-bass
// -------------------------------------------------------------------------
SynthDef(\directOut, { |in=0, out=0, masterVol=0.75, muteAll=0|
    var sig, muteGate;
    muteGate = 1 - Lag.kr(muteAll, 0.5);
    sig = In.ar(in, 2);
    sig = Limiter.ar(sig, 0.95);
    Out.ar(out, sig * masterVol * muteGate);
}).add;

// -------------------------------------------------------------------------
// infraDrone — sub-audible acceleration texture (5-12 Hz)
// Routes to directBus (clean, no FX). Not a GUI voice — felt, not heard.
// Driven by: acceleration_norm (freq + amp)
// -------------------------------------------------------------------------
SynthDef(\infraDrone, { |out=0, freq=8, amp=0.04, gate=1, lagTime=2.0|
    var sig, env;
    freq = Lag.kr(freq, lagTime);
    amp = Lag.kr(amp, lagTime);
    sig = SinOsc.ar(freq) * amp;
    env = EnvGen.kr(Env.asr(3, 1, 5), gate, doneAction: 2);
    Out.ar(out, sig ! 2 * env);
}).add;

"SynthDefs loaded (v7: reefTexture, abyssDrone, migrationVoice, histogramFilter, spectralGrain, onsetPercussion, migrationSweep, masterFXv7, directOut, infraDrone).".postln;
)

// ============================================================================
// SECTION 5: GUI (v7) — 1100 × 700
// ============================================================================

(
~createGUI = {
    var win, makeKnob, makeVoiceCol;

    if(~guiWindow.notNil) { ~guiWindow.close };

    ~guiWindow = Window("Sonificació Oceànica v7 — Event-Driven", Rect(40, 30, 1100, 700))
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
        .string_("SONIFICACIÓ OCEÀNICA  v7  —  Event-Driven Composition")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 14));

    // Transport buttons
    Button(win, Rect(780, 5, 75, 26))
        .states_([
            ["▶ PLAY", Color.black, Color.new255(50, 220, 80)],
            ["◼ STOP", Color.white, Color.new255(200, 50, 50)]
        ])
        .font_(Font("Helvetica Bold", 11))
        .action_({ |btn|
            if(btn.value == 1) { ~playComposition.value } { ~stopAll.value };
        });

    Button(win, Rect(860, 5, 75, 26))
        .states_([["⏺ REC", Color.white, Color.gray(0.3)]])
        .font_(Font("Helvetica Bold", 11))
        .action_({
            if(s.isRecording)
                { s.stopRecording; "Stopped recording.".postln }
                { s.record;        "Recording...".postln };
        });

    Button(win, Rect(940, 5, 75, 26))
        .states_([["REFRESH", Color.white, Color.gray(0.3)]])
        .font_(Font("Helvetica", 10))
        .action_({ ~createGUI.value });

    // ========= VERTICAL STRUCTURE VISUALIZATION (550 × 260) =========
    StaticText(win, Rect(10, 34, 550, 14))
        .string_("VERTICAL STRUCTURE  (32-band energy × time; white = depth of mass; green = playhead)")
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
        var bandH = h / 32;

        // Draw 32-band energy columns
        ~vizEchoW.do { |col|
            var x = (col / (~vizEchoW - 1)) * w;
            var hist = ~vizEchogram[col];
            if(hist.notNil) {
                32.do { |row|
                    var val = hist[row] ? 0;
                    var y = h - ((row + 1) * bandH);  // bottom=deep, top=shallow
                    // Color: black→blue→cyan→yellow→red
                    var hue = 0.63 - (val * 0.63);
                    Pen.fillColor_(Color.hsv(hue.clip(0, 1), 0.85, val.linlin(0, 0.3, 0.05, 0.9).clip(0.05, 0.9), 0.85));
                    Pen.fillRect(Rect(x, y, (w / ~vizEchoW).ceil + 1, bandH.ceil + 1));
                };
            };
        };

        // Depth-of-mass trace (white line — the DVM migration curve)
        Pen.strokeColor_(Color.new255(255, 255, 255, 200));
        Pen.width_(1.5);
        Pen.moveTo(0 @ (((~vizDepthTrace[0] ? 400) / 1000) * h));
        ~vizN.do { |i|
            var x = (i / (~vizN - 1)) * w;
            var depthM = (~vizDepthTrace[i] ? 400).clip(0, 1000);
            var y = (depthM / 1000) * h;  // 0m=top, 1000m=bottom
            Pen.lineTo(x @ y);
        };
        Pen.stroke;

        // Depth zone boundaries (surface/midwater at 150m, midwater/deep at 400m)
        [150, 400].do { |zoneDepth|
            var y = (zoneDepth / 1000) * h;
            Pen.strokeColor_(Color.gray(0.35));
            Pen.width_(0.5);
            Pen.line(0 @ y, w @ y);
            Pen.stroke;
            Pen.stringAtPoint(
                zoneDepth.asString ++ "m",
                (w - 35) @ (y - 10),
                Font("Helvetica", 7), Color.gray(0.45));
        };

        // Onset flash marks (orange at top)
        ~recentOnsets.do { |evtPct|
            var ox = evtPct * w;
            Pen.strokeColor_(Color.new255(255, 140, 0, 180));
            Pen.width_(1.0);
            Pen.line(ox @ 0, ox @ (h * 0.15));
            Pen.stroke;
        };

        // Green playhead
        Pen.strokeColor_(Color.new255(0, 220, 100));
        Pen.width_(2.0);
        Pen.line((pct * w) @ 0, (pct * w) @ h);
        Pen.stroke;

        // Depth labels
        ["0m", "250m", "500m", "750m", "1000m"].do { |lbl, i|
            var y = i / 4 * h;
            Pen.fillColor_(Color.gray(0.4));
            Pen.stringAtPoint(lbl, 3 @ (h - y - 10), Font("Helvetica", 8), Color.gray(0.4));
        };
    };

    // ========= LIVE HISTOGRAM (180 × 260, smoothed) =========
    StaticText(win, Rect(570, 34, 180, 14))
        .string_("DEPTH PROFILE (smoothed)")
        .stringColor_(Color.gray(0.55))
        .font_(Font("Helvetica", 9));

    ~histView = UserView(win, Rect(570, 48, 180, 260))
        .background_(Color.gray(0.1))
        .animate_(true)
        .frameRate_(15);

    ~histView.drawFunc = { |uv|
        var w = uv.bounds.width;
        var h = uv.bounds.height;
        var bandH = h / 32;
        var barMaxW = w - 30;

        32.do { |i|
            var val = ~currentHist[i] ? 0;
            var y = h - ((i + 1) * bandH);
            var barW = val * barMaxW;
            var hue = 0.63 - (val * 0.63);

            Pen.fillColor_(Color.hsv(hue.clip(0, 1), 0.8, 0.7, 0.9));
            Pen.fillRect(Rect(25, y + 1, barW, bandH - 2));
        };

        // Depth scale
        [0, 8, 16, 24, 31].do { |idx|
            var y = h - ((idx + 1) * bandH) + (bandH * 0.3);
            var depthM = (10 * (1000/10).pow(idx / 31)).round(1);
            Pen.stringAtPoint(depthM.asString, 0 @ y, Font("Helvetica", 7), Color.gray(0.5));
        };
    };

    // ========= INFO PANEL =========
    ~infoDisplay = StaticText(win, Rect(760, 48, 330, 110))
        .string_("UTC: --:--    Section: --------\n" ++
                 "Depth: --- m   Velocity: --- m/h\n" ++
                 "Entropy: ----  IPC: ----\n" ++
                 "Accel: ----    SprdChg: ----\n" ++
                 "Skewness: ---- Kurtosis: ----")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 10))
        .background_(Color.gray(0.12));

    ~sectionDisplay = StaticText(win, Rect(760, 162, 330, 24))
        .string_("SECTION: ---")
        .stringColor_(Color.cyan)
        .align_(\center)
        .font_(Font("Helvetica Bold", 13))
        .background_(Color.gray(0.18));

    // ========= FEATURE METERS (12 bars) =========
    StaticText(win, Rect(760, 192, 330, 14))
        .string_("FEATURES [Dpt Itn Ent IPC Sep Ons Vel Acc Spr SpC Skw Krt]")
        .stringColor_(Color.gray(0.55))
        .font_(Font("Helvetica", 8));

    ~featureMeters = MultiSliderView(win, Rect(760, 206, 330, 60))
        .size_(12)
        .isFilled_(true)
        .drawLines_(false)
        .elasticMode_(1)
        .value_(Array.fill(12, 0.5))
        .background_(Color.gray(0.12))
        .fillColor_(Color.new255(0, 200, 255))
        .strokeColor_(Color.gray(0.5));

    ["Dpt","Itn","Ent","IPC","Sep","Ons","Vel","Acc","Spr","SpC","Skw","Krt"].do { |lbl, i|
        StaticText(win, Rect(760 + (i * 27.5), 266, 27, 10))
            .string_(lbl)
            .stringColor_(Color.gray(0.45))
            .align_(\center)
            .font_(Font("Helvetica", 7));
    };

    // ========= VOICE CONTROLS (with Solo/Mute) =========
    StaticText(win, Rect(10, 318, 200, 14))
        .string_("VOICES")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 11));

    [\reef, \abyss, \migration, \filterBank, \grain, \onset].do { |name, i|
        var x = 10 + (i * 100);
        var labels = ["SURFACE", "DEEP", "MIGR", "FILTER", "GRAIN", "ONSET"];
        var colors = [
            Color.new255(50, 220, 80),   // surface = green (epipelagic 10-150m)
            Color.new255(80, 100, 200),   // deep = deep blue (mesopelagic 400-800m)
            Color.new255(255, 160, 50),   // migration = orange (DVM movement)
            Color.new255(200, 100, 255),  // filter = purple (32-band depth structure)
            Color.yellow,                  // grain = yellow (IFFT granular)
            Color.new255(255, 80, 80)     // onset = red (acoustic transients)
        ];

        // Label
        StaticText(win, Rect(x, 334, 90, 14))
            .string_(labels[i])
            .stringColor_(colors[i])
            .align_(\center)
            .font_(Font("Helvetica Bold", 9));

        // Volume knob
        makeKnob.(win, Rect(x + 20, 348, 50, 50), "Vol", ~voiceVol[name], 0, 1, colors[i]);

        // Solo button
        Button(win, Rect(x + 5, 406, 38, 18))
            .states_([
                ["S", Color.white, Color.gray(0.25)],
                ["S", Color.black, Color.yellow]
            ])
            .font_(Font("Helvetica Bold", 9))
            .action_({ |btn|
                ~voiceSolo[name].set(btn.value);
            });

        // Mute button
        Button(win, Rect(x + 48, 406, 38, 18))
            .states_([
                ["M", Color.white, Color.gray(0.25)],
                ["M", Color.white, Color.red(0.6)]
            ])
            .font_(Font("Helvetica Bold", 9))
            .action_({ |btn|
                ~voiceMute[name].set(btn.value);
            });
    };

    // ========= EFFECTS + MASTER =========
    StaticText(win, Rect(620, 318, 100, 14))
        .string_("MASTER + EFFECTS")
        .stringColor_(Color.magenta)
        .font_(Font("Helvetica Bold", 10));

    makeKnob.(win, Rect(620, 338, 50, 50), "Master", ~ctrl[\masterVol], 0, 1, Color.white);
    makeKnob.(win, Rect(675, 338, 50, 50), "Reverb", ~ctrl[\reverbMix], 0, 1, Color.magenta);
    makeKnob.(win, Rect(730, 338, 50, 50), "Delay",  ~ctrl[\delayMix], 0, 1, Color.magenta);
    makeKnob.(win, Rect(785, 338, 50, 50), "Dist",   ~ctrl[\distortion], 0, 1, Color.magenta);
    makeKnob.(win, Rect(840, 338, 50, 50), "Filter", ~ctrl[\filterMod], 0, 1, Color.magenta);
    makeKnob.(win, Rect(895, 338, 50, 50), "Tension", ~ctrl[\tensionBias], 0, 1, Color.new255(255, 100, 100));

    // Onset threshold slider — controls percussion event sensitivity
    StaticText(win, Rect(955, 318, 90, 14))
        .string_("Onset Thr")
        .stringColor_(Color.new255(255, 160, 50))
        .font_(Font("Helvetica", 9));
    Slider(win, Rect(965, 338, 40, 50))
        .value_(ControlSpec(0.03, 0.5).unmap(~ctrl[\onsetThreshold].getSynchronous))
        .background_(Color.gray(0.2))
        .knobColor_(Color.new255(255, 160, 50))
        .action_({ |sl|
            ~ctrl[\onsetThreshold].set(ControlSpec(0.03, 0.5).map(sl.value));
        });
    StaticText(win, Rect(948, 392, 100, 22))
        .string_("transient\nsensitivity")
        .stringColor_(Color.gray(0.45))
        .align_(\center)
        .font_(Font("Helvetica", 7));

    // ========= NAVIGATION ROW =========
    StaticText(win, Rect(10, 440, 100, 14))
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
        Button(win, Rect(10 + (i * 78), 454, 72, 24))
            .states_([[lbl, Color.black, col]])
            .font_(Font("Helvetica Bold", 10))
            .action_({
                var targetIdx = ~hour.detectIndex { |h| h >= hourFrac } ? 0;
                var targetReal = ~times[targetIdx] ? 0;
                ~currentTime = targetReal / ~compression;
                ("Jump to " ++ lbl ++ " — comp time: " ++ ~currentTime.round(1) ++ " s").postln;
            });
    };

    // Speed buttons
    StaticText(win, Rect(340, 440, 60, 14))
        .string_("SPEED")
        .stringColor_(Color.gray(0.6))
        .font_(Font("Helvetica", 9));

    ~speedBtns = Array.new(4);
    [0.5, 1.0, 2.0, 4.0].do { |spd, i|
        var lbl = if(spd < 1) { "0.5×" } { spd.asInteger.asString ++ "×" };
        ~speedBtns = ~speedBtns.add(
            Button(win, Rect(340 + (i * 52), 454, 46, 24))
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
    StaticText(win, Rect(10, 490, 1080, 30))
        .string_(
            "SONIFICACIÓ OCEÀNICA v7 — Oceànica (Alex Cabrer & Joan Sala) | CLT019 Generalitat de Catalunya\n" ++
            "MALASPINA 2010 — Atlantic Ocean — January 26, 2011 — 38 kHz · 29,997 pings · DVM calibrated · 32-band depth histogram"
        )
        .stringColor_(Color.gray(0.4))
        .font_(Font("Helvetica", 9));

    win.onClose = {
        ~stopAll.value;
        if(~ifftBuf.notNil) { ~ifftBuf.free };
        "Window closed. Buffer freed.".postln;
    };

    "GUI v7 created (1100×700, vertical structure, histogram, solo/mute).".postln;
};

~createGUI.value;
)

// ============================================================================
// SECTION 6: DATA ACCESS FUNCTIONS
// ============================================================================

(
// Windowed mean (same as v6)
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

// Histogram lookup (nearest sample for the 32-band array)
~getHistogram = { |timeArray, histArray, compTime, compression|
    var realTime, idx;
    realTime = compTime * compression;
    idx = (timeArray.indexOfGreaterThan(realTime) ? (histArray.size - 1)).max(0).min(histArray.size - 1);
    histArray[idx] ? Array.fill(32, 0);
};

"Data functions ready.".postln;
)

// ============================================================================
// SECTION 7: COMPOSITION ENGINE (v7 — Event-Driven)
// ============================================================================

(
~fxBus     = Bus.audio(s, 2);
~directBus = Bus.audio(s, 2);

~composition = Routine({
    var updateRate = 0.05;
    var currentTime = 0;

    // Synth references
    var synths = IdentityDictionary.new;
    var fxSynth, directSynth;
    var bufToUse;

    // Current data values
    var depthVal, intensityVal, spreadVal, layersVal, velocityVal,
        hourVal, anomalyVal, entropyVal, ipcVal, layerSepVal, onsetVal;
    var accelVal, spreadChangeVal, skewnessVal, kurtosisVal;
    var vertGradVal, peakPromVal, peakWidthVal;
    var surfaceVal, midwaterVal, deepVal;
    var depthRawVal, velRawVal;
    var histVals;
    // Tier 2 data values
    var velocityStableVal, outlierVal, autocorr3Val, autocorr10Val;

    // Section tracking
    var section = \night1, lastSection = \none;
    var sectionName, bioState;
    var sectionHoldCount = 0, candidateSection = \none;

    // Mute-all detection + sweep tracking
    var allMuted = false;
    var activeSweeps = List.new;

    // Onset cooldown
    var lastOnsetTime = -1.0;
    var onsetCooldown = (0.15 / ~speed) / (~compression.sqrt);

    // Tension
    var tension;

    "".postln;
    "╔══════════════════════════════════════════════════════════════════╗".postln;
    "║   SONIFICACIÓ OCEÀNICA v7 — Event-Driven Composition           ║".postln;
    "║   6 voices · 32-band filter · solo/mute · 8-min cycle          ║".postln;
    "╚══════════════════════════════════════════════════════════════════╝".postln;

    bufToUse = if(~ifftBuf.notNil) { ~ifftBuf.bufnum } { 0 };
    ("IFFT buffer: " ++ bufToUse).postln;

    // Start FX chains
    fxSynth     = Synth(\masterFXv7, [\in, ~fxBus, \out, 0], addAction: \addToTail);
    directSynth = Synth(\directOut, [\in, ~directBus, \out, 0], addAction: \addToTail);

    // Persistent voices (always run, amplitude modulated by data)
    synths[\abyss] = Synth(\abyssDrone, [\out, ~directBus, \amp, 0, \gate, 1]);
    synths[\reef] = Synth(\reefTexture, [\out, ~fxBus, \amp, 0, \gate, 1]);
    synths[\histogram] = Synth(\histogramFilter, [\out, ~fxBus, \amp, 0, \gate, 1]);
    synths[\grain] = Synth(\spectralGrain, [\out, ~fxBus, \amp, 0, \gate, 1,
        \bufnum, bufToUse]);

    // Migration is now persistent (continuous amplitude driven by velocity deviation)
    synths[\migration] = Synth(\migrationVoice, [\out, ~fxBus, \amp, 0, \gate, 1]);

    // Infrasound (sub-audible acceleration texture, no GUI control)
    synths[\infra] = Synth(\infraDrone, [\out, ~directBus, \amp, 0, \gate, 1]);

    0.3.wait;

    while({ currentTime < ~totalDuration }, {

        // ===== READ ALL DATA VALUES =====
        depthVal     = ~getValue.(~depth,       ~times, currentTime, ~compression, 1.5 * ~speed);
        intensityVal = ~getValue.(~intensity,   ~times, currentTime, ~compression, 1.5 * ~speed);
        spreadVal    = ~getValue.(~spread,      ~times, currentTime, ~compression, 1.5 * ~speed);
        layersVal    = ~getValue.(~layers,      ~times, currentTime, ~compression, 1.5 * ~speed);
        velocityVal  = ~getValue.(~velocity,    ~times, currentTime, ~compression, 1.5 * ~speed);
        hourVal      = ~getValue.(~hour,        ~times, currentTime, ~compression, 1.0 * ~speed);
        anomalyVal   = ~getValue.(~anomaly,     ~times, currentTime, ~compression, 1.5 * ~speed);
        entropyVal   = ~getValue.(~entropy,     ~times, currentTime, ~compression, 1.5 * ~speed);
        ipcVal       = ~getValue.(~ipc,         ~times, currentTime, ~compression, 1.5 * ~speed);
        layerSepVal  = ~getValue.(~layerSep,    ~times, currentTime, ~compression, 1.5 * ~speed);
        onsetVal     = ~getValue.(~onset,       ~times, currentTime, ~compression, 0.5 * ~speed);
        surfaceVal   = ~getValue.(~surfaceN,    ~times, currentTime, ~compression, 1.5 * ~speed);
        midwaterVal  = ~getValue.(~midwaterN,   ~times, currentTime, ~compression, 1.5 * ~speed);
        deepVal      = ~getValue.(~deepN,       ~times, currentTime, ~compression, 1.5 * ~speed);

        // v5 features
        accelVal        = ~getValue.(~acceleration, ~times, currentTime, ~compression, 1.5 * ~speed);
        spreadChangeVal = ~getValue.(~spreadChange, ~times, currentTime, ~compression, 1.5 * ~speed);
        skewnessVal     = ~getValue.(~skewness,     ~times, currentTime, ~compression, 1.5 * ~speed);
        kurtosisVal     = ~getValue.(~kurtosis,     ~times, currentTime, ~compression, 1.5 * ~speed);
        vertGradVal     = ~getValue.(~vertGrad,     ~times, currentTime, ~compression, 1.5 * ~speed);
        peakPromVal     = ~getValue.(~peakProm,     ~times, currentTime, ~compression, 1.5 * ~speed);
        peakWidthVal    = ~getValue.(~peakWidth,    ~times, currentTime, ~compression, 1.5 * ~speed);

        // Tier 2 features
        velocityStableVal = ~getValue.(~velocityStable, ~times, currentTime, ~compression, 0.5 * ~speed);
        outlierVal      = ~getValue.(~outlier,      ~times, currentTime, ~compression, 1.0 * ~speed);
        autocorr3Val    = ~getValue.(~autocorr3,    ~times, currentTime, ~compression, 1.5 * ~speed);
        autocorr10Val   = ~getValue.(~autocorr10,   ~times, currentTime, ~compression, 1.5 * ~speed);

        // Raw display values
        depthRawVal  = ~getRaw.(~depthRaw, ~times, currentTime, ~compression);
        velRawVal    = ~getRaw.(~velocityRaw, ~times, currentTime, ~compression);

        // Histogram (32-element array)
        histVals = ~getHistogram.(~times, ~depthHist, currentTime, ~compression);

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

        // ===== TENSION SYSTEM =====
        tension = (accelVal * onsetVal.max(0.1) * (1 - ipcVal)).clip(0, 1);
        tension = tension * ~ctrl[\tensionBias].getSynchronous.linlin(0, 1, 0.3, 1.5);
        tension = tension.clip(0, 1);

        // ===== UPDATE PERSISTENT VOICES =====

        // Deep drone (always on, with midwater-driven filter drift for day-section breathing)
        synths[\abyss].set(
            \freq,       deepVal.clip(0.001, 1).linexp(0.001, 1, 25, 120),
            \amp,        ~getVoiceAmp.(\abyss, deepVal.linlin(0, 1, 0.08, 0.45)),
            \accel,      accelVal,
            \filterFreq, entropyVal.clip(0.001, 1).linexp(0.3, 1.0, 60, 1500),
            \kurtosis,   kurtosisVal,
            \driftRate,  midwaterVal.linlin(0.0, 1.0, 0.03, 0.12)
                * autocorr10Val.linlin(0, 1, 1.5, 0.5),
            \driftDepth, (1 - accelVal).linlin(0.0, 1.0, 0.05, 0.35)
        );

        // Histogram filter (always on, midwater-driven excitation rate for day-section texture)
        32.do { |i|
            synths[\histogram].set(("h" ++ i).asSymbol, histVals[i] ? 0);
        };
        synths[\histogram].set(
            \amp, ~getVoiceAmp.(\filterBank,
                intensityVal.linlin(0.3, 0.9, 0.15, 0.45)),
            \kurtosis, kurtosisVal,
            \rq, 0.15,
            \exciteRate, midwaterVal.linlin(0.0, 1.0, 8, 35)
        );

        // Spectral grain (stable velocity + autocorrelation modulate density; outlier boosts amp)
        synths[\grain].set(
            \pos,       (currentTime / ~totalDuration).clip(0.001, 0.999),
            \amp,       ~getVoiceAmp.(\grain,
                intensityVal.linlin(0.3, 0.9, 0.05, 0.4)
                * (1 + (outlierVal * 0.4))),
            \ipc,       ipcVal,
            \entropy,   entropyVal,
            \intensity, intensityVal,
            \density,   (6 + (entropyVal * 30) + ((velocityStableVal - 0.5).abs * 20))
                * autocorr3Val.linlin(0, 1, 1.3, 0.8)
        );

        // ===== UPDATE PERSISTENT SURFACE VOICE =====
        synths[\reef].set(
            \freq,     surfaceVal.clip(0.001, 1).linexp(0.5, 0.95, 400, 2200),
            \amp,      ~getVoiceAmp.(\reef, surfaceVal.linlin(0.5, 0.95, 0.05, 0.5)),
            \entropy,  entropyVal,
            \ipc,      ipcVal,
            \spread,   spreadVal.linlin(0, 1, 0.2, 0.8),
            \gradient, vertGradVal
        );

        // Migration (persistent, amplitude driven by velocity deviation — 0 at stationary)
        synths[\migration].set(
            \freq,         velocityVal.clip(0.001, 1).linexp(0.001, 1, 80, 800),
            \amp,          ~getVoiceAmp.(\migration,
                (velocityVal - 0.5).abs.linlin(0.0, 0.5, 0.0, 0.45)),
            \velocity,     velocityVal,
            \anomaly,      anomalyVal,
            \spreadChange, spreadChangeVal,
            \layerSep,     layerSepVal
        );

        // Infrasound (5-12 Hz, acceleration-driven, felt not heard — bypasses voice controls)
        synths[\infra].set(
            \freq, accelVal.linexp(0.001, 1, 5, 15),
            \amp,  accelVal.linlin(0, 1, 0.01, 0.06)
        );

        // ===== DAWN/DUSK SWEEPS (data-driven, respects migration voice controls) =====
        if(section == \dawn and: { lastSection != \dawn }) {
            var sweepAmp = ~getVoiceAmp.(\migration,
                (velocityVal - 0.5).abs.linlin(0, 0.5, 0.08, 0.22));
            if(sweepAmp > 0) {
                var sw = Synth(\migrationSweep, [\out, ~fxBus,
                    \startFreq, depthRawVal.clip(100, 800).linexp(100, 800, 500, 100),
                    \endFreq,   80,
                    \dur,       accelVal.linlin(0, 1, 28, 14),
                    \amp,       sweepAmp,
                    \direction, 1,
                    \brightness, entropyVal]);
                activeSweeps.add(sw);
            };
            ">>> DAWN sweep (data-driven) <<<".postln;
        };
        if(section == \dusk and: { lastSection != \dusk }) {
            var sweepAmp = ~getVoiceAmp.(\migration,
                (velocityVal - 0.5).abs.linlin(0, 0.5, 0.08, 0.22));
            if(sweepAmp > 0) {
                var sw = Synth(\migrationSweep, [\out, ~fxBus,
                    \startFreq, 80,
                    \endFreq,   depthRawVal.clip(100, 800).linexp(100, 800, 500, 100),
                    \dur,       accelVal.linlin(0, 1, 24, 12),
                    \amp,       sweepAmp,
                    \direction, -1,
                    \brightness, entropyVal]);
                activeSweeps.add(sw);
            };
            ">>> DUSK sweep (data-driven) <<<".postln;
        };

        lastSection = section;

        // ===== ONSET DETECTION (outlier lowers threshold → more hits during anomalies) =====
        if((onsetVal > (~ctrl[\onsetThreshold].getSynchronous * (1 - (outlierVal * 0.6)))) and:
           { (currentTime - lastOnsetTime) > onsetCooldown }) {
            var depthZone;
            lastOnsetTime = currentTime;
            // Ratio of surface vs deep energy: 0=deep dominant, 1=surface dominant
            depthZone = (surfaceVal / (surfaceVal + deepVal + 0.001)).clip(0, 1);
            Synth(\onsetPercussion, [
                \out,        ~fxBus,
                \freq,       depthVal.clip(0.001, 1).linexp(0.001, 1, 70, 600),
                \amp,        ~getVoiceAmp.(\onset, onsetVal * 0.5),
                \sustain,    onsetVal.linlin(0, 1, 0.04, 0.4),
                \brightness, entropyVal,
                \prominence, peakPromVal,
                \peakWidth,  peakWidthVal,
                \depthZone,  depthZone
            ]);
            ~recentOnsets = (~recentOnsets.add(currentTime / ~totalDuration)).keep(-20);
        };

        // ===== EFFECTS UPDATE (data-driven reverb, no hardcoded section values) =====
        fxSynth.set(
            \delayMix,    ~ctrl[\delayMix].getSynchronous,
            \delayFb,     ~ctrl[\feedback].getSynchronous,
            \reverbMix,   ~ctrl[\reverbMix].getSynchronous,
            \reverbRoom,  entropyVal.linlin(0.4, 0.9, 0.4, 0.8),
            \distortion,  (~ctrl[\distortion].getSynchronous + (tension * 0.3)).clip(0, 1),
            \filterMod,   ~ctrl[\filterMod].getSynchronous,
            \spreadChange, spreadChangeVal,
            \masterVol,   ~ctrl[\masterVol].getSynchronous
        );
        directSynth.set(\masterVol, ~ctrl[\masterVol].getSynchronous);

        // ===== MUTE-ALL DETECTION (kills FX tails + active sweeps) =====
        allMuted = ~voiceNames.every { |n| ~getVoiceAmp.(n, 1.0) == 0 };
        fxSynth.set(\muteAll, allMuted.asInteger);
        directSynth.set(\muteAll, allMuted.asInteger);
        if(allMuted) {
            activeSweeps.do { |sw| try { sw.free } };
            activeSweeps.clear;
        };

        // ===== GUI UPDATE =====
        {
            var mins    = (currentTime / 60).floor.asInteger;
            var secs    = (currentTime % 60).floor.asInteger;
            var rHour   = (hourVal * 24).floor.asInteger;
            var rMin    = ((hourVal * 24 * 60) % 60).floor.asInteger;
            var velDir  = if(velRawVal > 0) { " (asc)" } { " (dsc)" };

            ~currentTime = currentTime;
            ~depthRawVal = depthRawVal;
            // Exponential moving average for smooth histogram display
            ~currentHist = ~currentHist.collect({ |old, i|
                (old * 0.75) + ((histVals[i] ? 0) * 0.25)
            });

            if(~infoDisplay.notNil and: { ~infoDisplay.isClosed.not }) {
                ~infoDisplay.string_(
                    "UTC: "  ++ rHour.asString.padLeft(2, "0") ++ ":"
                              ++ rMin.asString.padLeft(2,  "0")
                    ++ "    Section: " ++ sectionName ++ "\n"
                    ++ "Depth: "   ++ depthRawVal.round(1) ++ " m"
                    ++ "   Velocity: " ++ velRawVal.round(1) ++ " m/h" ++ velDir ++ "\n"
                    ++ "Entropy: " ++ entropyVal.round(0.001) ++ "  IPC: " ++ ipcVal.round(0.001) ++ "\n"
                    ++ "Accel: " ++ accelVal.round(0.001) ++ "    SprdChg: " ++ spreadChangeVal.round(0.001) ++ "\n"
                    ++ "Skewness: " ++ skewnessVal.round(0.001) ++ "  Kurtosis: " ++ kurtosisVal.round(0.001)
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
                    layerSepVal, onsetVal, velocityVal, accelVal,
                    spreadVal, spreadChangeVal,
                    skewnessVal, kurtosisVal
                ]);
            };
        }.defer;

        updateRate.wait;
        currentTime = currentTime + updateRate;
    });

    // === FADE OUT ===
    "Composition ending — fading out...".postln;
    [synths[\abyss], synths[\reef], synths[\migration], synths[\histogram], synths[\grain], synths[\infra]].do { |s|
        if(s.notNil) { s.set(\gate, 0) };
    };
    activeSweeps.do { |sw| try { sw.free } };

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
    "Composition started (v7, 8 min).".postln;
};

~stopAll = {
    ~composition.stop;
    s.freeAll;
    ~recentOnsets = Array.new(20);
    "Stopped. All synths freed.".postln;
};

"Composition engine v7 ready.".postln;
)


// ============================================================================
// END
// ============================================================================

/*
HOW TO USE (v7):
  1. Execute sections 1–7 in order (Cmd+Enter each block)
  2. Press PLAY or run ~playComposition.value
  3. Use GUI to shape sound, solo/mute voices, explore sections
  4. All voice volume knobs, solo, and mute controls work correctly

KEY DIFFERENCES FROM v6:
  - Data-driven: ranges calibrated to actual data corridors (not hypothetical 0-1 extremes)
  - Surface voice persistent (no event gate); gradient-driven brightness via RLPF
  - Deep drone: wider freq range (25-120Hz), acceleration vibrato, autocorrelation-modulated filter drift
  - Migration: persistent with continuous amp from velocity deviation (no binary pop)
  - Histogram filter: kurtosis-driven Q bandwidth, midwater-driven excitation rate (day texture)
  - Granular: stable velocity + autocorrelation density modulation, outlier amp boost
  - Onset: depth-zone timbral variety, outlier-lowered threshold, compression-scaled cooldown
  - FX: muteAll gate on both FX + direct chains (kills reverb/delay tails on full mute)
  - Infrasound layer (5-15 Hz): acceleration-driven, felt not heard
  - Presets removed (simplification)
  - Uses v5 JSON with Tier 2 features: velocity_stable, outlier, autocorrelation

VOICE → DATA MAPPING:
  surfaceTexture  ← surface_norm (freq 400-2200Hz + amp), entropy (tonal↔noise),
                     vertical_gradient (RLPF brightness 800-8000Hz), ipc (stability)
                     Represents: epipelagic scattering layer (10-150m)
  deepDrone       ← deep_norm (freq 25-120Hz + amp), acceleration (vibrato + AM wobble),
                     entropy (filter 60-1500Hz), kurtosis (harmonics),
                     midwater (drift rate), autocorr_10min (drift modulation)
                     Represents: mesopelagic refuge layer (400-800m)
  migrationVoice  ← velocity_norm (freq 80-800Hz), continuous amp from |velocity-0.5|,
                     anomaly_norm (detuned beating), spread_change (filter resonance)
                     Represents: DVM vertical movement (persistent, silent at stationary)
  migrationSweep  ← depthRawVal (freq), accelVal (duration), entropyVal (brightness)
                     Represents: dawn/dusk transition events (controlled by MIGR voice)
  histogramFilter ← depth_histogram[32] (band drives), kurtosis (Q bandwidth),
                     intensity_norm (amp 0.15-0.45), midwater (excitation rate 8-35)
                     Represents: vertical ocean structure (energy at each depth stratum)
  spectralGrain   ← composition time (pos), ipc^0.3 (grain duration),
                     density = entropy + velocity_stable + autocorr_3min modulation,
                     amp boosted by outlier_score
                     Represents: overall echogram texture (from IFFT audio buffer)
  onsetPercussion ← onset_strength (trigger, threshold lowered by outlier_score),
                     peak_prominence (attack), peak_width (sustain),
                     depthZone (surface/deep → timbral variety + stereo width),
                     cooldown scaled by sqrt(compression)
                     Represents: sudden acoustic transients (biomass changes)
  infraDrone      ← acceleration_norm (freq 5-15Hz, amp 0.01-0.06)
                     Represents: physical sensation of rapid depth changes (sub-audible)

SIGNAL ROUTING:
  deepDrone + infraDrone → directBus → Limiter (clean sub/infra, no FX)
  all others             → fxBus    → dist → LPF → delay → reverb → HPF → Limiter
  muteAll gate on both chains → 0.5s fade kills FX tails when all voices muted

ONSET THRESHOLD:
  Controls minimum onset_strength_norm to trigger percussion hits.
  Effective threshold = knob value * (1 - outlier * 0.6) — anomalous moments get more hits.
  Cooldown = (0.15 / speed) / sqrt(compression) — ~30-80 events over 8 min.

DATA: sonification_sc_v5_20110126.json (v4 + histogram + shape + velocity_stable + outlier + autocorrelation)
BUFFER: echogram_audio_20110126_ifft.wav
*/
