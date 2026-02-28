/*
================================================================================
SONIFICACIÓ OCEÀNICA - Version 4: CREATIVE/AGGRESSIVE
================================================================================
Project: Sonificació Oceànica
Team: Oceànica (Alex Cabrer & Joan Sala)
Grant: CLT019 - Generalitat de Catalunya

THIS VERSION IS DIFFERENT:
- Uses ONLY 38 kHz data (120 kHz was broken/unusable)
- Fixed velocity with proper smoothing
- More aggressive, interesting synthesis
- Distinct sonic SCENES for each biological phase
- Spectral freeze, distortion, feedback
- Rhythmic elements from data
- Real contrast between sections

SONIC PHILOSOPHY:
Night = ALIVE, rhythmic, granular swarms, high activity
Dawn = TENSION, descending sweep, FM harshness, building
Day = SUSPENDED, spectral freeze, deep drone, minimal
Dusk = RELEASE, ascending sweep, opening filters, anticipation
Night Return = CELEBRATION, return of rhythm and life

================================================================================
*/

// ============================================================================
// SECTION 1: SERVER BOOT
// ============================================================================

(
s.options.numOutputBusChannels = 2;
s.options.numInputBusChannels = 0;
s.options.memSize = 2.pow(20);
s.options.sampleRate = 48000;
s.options.numBuffers = 8192;

s.waitForBoot({
    "Server ready.".postln;
});
)

// ============================================================================
// SECTION 2: LOAD FIXED V3 DATA
// ============================================================================

(
// Data path - relative to this .scd file location
// Uses SuperCollider's path resolution to find project root
~dataPath = thisProcess.nowExecutingPath.dirname.dirname +/+ "output/data/";

~jsonFile = File(~dataPath ++ "sonification_sc_v3_20110126.json", "r");
~jsonString = ~jsonFile.readAllString;
~jsonFile.close;
~jsonData = ~jsonString.parseJSON;

// All 38 kHz data (120 kHz removed - was broken)
~times = ~jsonData["38kHz"]["time_seconds"].asArray.collect(_.asFloat);
~depth = ~jsonData["38kHz"]["depth_norm"].asArray.collect(_.asFloat);
~intensity = ~jsonData["38kHz"]["intensity_norm"].asArray.collect(_.asFloat);
~spread = ~jsonData["38kHz"]["spread_norm"].asArray.collect(_.asFloat);
~layers = ~jsonData["38kHz"]["layers_norm"].asArray.collect(_.asFloat);
~velocity = ~jsonData["38kHz"]["velocity_norm"].asArray.collect(_.asFloat);
~acceleration = ~jsonData["38kHz"]["acceleration_norm"].asArray.collect(_.asFloat);
~anomaly = ~jsonData["38kHz"]["anomaly_norm"].asArray.collect(_.asFloat);
~hour = ~jsonData["38kHz"]["hour_norm"].asArray.collect(_.asFloat);

// Raw for display
~depthRaw = ~jsonData["38kHz"]["depth_m"].asArray.collect(_.asFloat);
~velocityRaw = ~jsonData["38kHz"]["velocity_m_h"].asArray.collect(_.asFloat);

~totalDuration = 240;  // 4 minutes
~compression = ~times.last / ~totalDuration;

"".postln;
"=== V3 FIXED DATA LOADED ===".postln;
"Points: ".post; ~times.size.postln;
"Using 38 kHz only (120 kHz removed - was broken)".postln;
"Compression: ".post; ~compression.round(1).post; "x".postln;
)

// ============================================================================
// SECTION 3: CONTROL BUSES
// ============================================================================

(
~ctrl = IdentityDictionary.new;

// Master
~ctrl[\masterVol] = Bus.control(s, 1).set(0.75);
~ctrl[\chaos] = Bus.control(s, 1).set(0.5);  // Global chaos/intensity

// Scene blend (0=night, 0.5=day, affects overall character)
~ctrl[\sceneBlend] = Bus.control(s, 1).set(0.5);

// Granular swarm
~ctrl[\swarmVol] = Bus.control(s, 1).set(0.6);
~ctrl[\swarmDensity] = Bus.control(s, 1).set(0.5);
~ctrl[\swarmPitch] = Bus.control(s, 1).set(0.5);

// Deep drone
~ctrl[\droneVol] = Bus.control(s, 1).set(0.5);
~ctrl[\droneDark] = Bus.control(s, 1).set(0.5);

// Rhythmic pulse
~ctrl[\pulseVol] = Bus.control(s, 1).set(0.4);
~ctrl[\pulseRate] = Bus.control(s, 1).set(0.5);

// Distortion/Aggression
~ctrl[\distortion] = Bus.control(s, 1).set(0.3);
~ctrl[\feedback] = Bus.control(s, 1).set(0.3);

// Effects
~ctrl[\reverbMix] = Bus.control(s, 1).set(0.5);
~ctrl[\delayMix] = Bus.control(s, 1).set(0.3);
~ctrl[\filterMod] = Bus.control(s, 1).set(0.5);

"Control buses ready.".postln;
)

// ============================================================================
// SECTION 4: AGGRESSIVE SYNTHDEFS
// ============================================================================

(
// -------------------------------------------------------------------------
// GRANULAR SWARM - Living, breathing texture
// Active at night, sparse during day
// -------------------------------------------------------------------------
SynthDef(\granularSwarm, { |out=0, freq=200, amp=0.4, density=20, spread=0.5,
    chaos=0.5, brightness=0.5, gate=1|
    var sig, env, grains, trigger, pan;

    // Chaotic trigger - more irregular with higher chaos
    trigger = Dust.kr(density * LFNoise1.kr(chaos * 2).range(0.5, 1.5));

    // Granular cloud with pitch variation
    grains = GrainSin.ar(
        numChannels: 2,
        trigger: trigger,
        dur: LFNoise1.kr(1).range(0.03, 0.15),
        freq: freq * LFNoise1.kr(chaos * 5 + 1).range(0.85, 1.15),
        pan: LFNoise2.kr(2).range(spread.neg, spread),
        mul: 0.5
    );

    // Add some FM grit
    sig = grains.sum + SinOsc.ar(
        freq * SinOsc.kr(chaos * 3).range(0.98, 1.02),
        SinOsc.ar(freq * 2, 0, chaos * 4),
        0.2 * brightness
    );

    // Brightness filter
    sig = LPF.ar(sig, brightness.linexp(0, 1, 400, 8000));

    env = EnvGen.kr(Env.asr(1, 1, 2), gate, doneAction: 2);

    Out.ar(out, sig * env * amp);
}).add;

// -------------------------------------------------------------------------
// DEEP DRONE - Subterranean presence
// Dominant during day, subtle at night
// -------------------------------------------------------------------------
SynthDef(\deepDrone, { |out=0, freq=50, amp=0.3, darkness=0.5, movement=0.3, gate=1|
    var sig, env, sub, harmonics, filtered;

    // Sub oscillator with slow beating
    sub = SinOsc.ar(freq, 0, 0.6)
        + SinOsc.ar(freq * 1.002, 0, 0.4)
        + SinOsc.ar(freq * 0.5, 0, 0.3);

    // Harmonic content - less with more darkness
    harmonics = Mix.fill(6, { |i|
        var h = i + 2;
        var amt = (1 - darkness) / h;
        SinOsc.ar(freq * h * LFNoise2.kr(0.1).range(0.998, 1.002), 0, amt * 0.15)
    });

    sig = sub + harmonics;

    // Slow filter movement
    filtered = LPF.ar(sig, darkness.linexp(0, 1, 2000, 150)
        * LFNoise2.kr(movement).range(0.7, 1.3));

    // Add subtle noise texture
    filtered = filtered + (BrownNoise.ar(0.03) * darkness);

    env = EnvGen.kr(Env.asr(4, 1, 5), gate, doneAction: 2);

    Out.ar(out, Pan2.ar(filtered * env * amp, LFNoise2.kr(0.1).range(-0.3, 0.3)));
}).add;

// -------------------------------------------------------------------------
// RHYTHMIC PULSE - Heartbeat of the migration
// Based on layer count and activity
// -------------------------------------------------------------------------
SynthDef(\rhythmicPulse, { |out=0, freq=80, amp=0.25, rate=2, decay=0.3,
    brightness=0.5, gate=1|
    var sig, env, impulse, resonant;

    impulse = Impulse.kr(rate);

    // Resonant body
    sig = SinOsc.ar(freq, 0, EnvGen.ar(Env.perc(0.005, decay), impulse))
        + SinOsc.ar(freq * 2, 0, EnvGen.ar(Env.perc(0.003, decay * 0.7), impulse) * 0.4)
        + SinOsc.ar(freq * 0.5, 0, EnvGen.ar(Env.perc(0.01, decay * 1.5), impulse) * 0.5);

    // Add click transient
    sig = sig + (HPF.ar(WhiteNoise.ar(0.3), 2000)
        * EnvGen.ar(Env.perc(0.001, 0.02), impulse) * brightness);

    // Resonant filter
    resonant = RLPF.ar(sig, freq * brightness.linlin(0, 1, 2, 8), 0.3);

    env = EnvGen.kr(Env.asr(0.5, 1, 1), gate, doneAction: 2);

    Out.ar(out, Pan2.ar(resonant * env * amp, 0));
}).add;

// -------------------------------------------------------------------------
// FM TENSION - Harsh, metallic, for transitions
// -------------------------------------------------------------------------
SynthDef(\fmTension, { |out=0, freq=150, amp=0.2, modRatio=2, modDepth=3,
    brightness=0.5, gate=1|
    var sig, env, mod, car;

    // Modulator with noise
    mod = SinOsc.ar(freq * modRatio, 0, modDepth)
        + LFNoise1.ar(freq * 0.1, modDepth * 0.3);

    // Carrier
    car = SinOsc.ar(freq + mod, 0, 0.5);

    // Add harmonics
    sig = car + SinOsc.ar(freq * 2 + (mod * 0.5), 0, 0.2)
        + SinOsc.ar(freq * 3 + (mod * 0.3), 0, 0.1);

    // Brightness control
    sig = LPF.ar(sig, brightness.linexp(0, 1, 500, 6000));

    env = EnvGen.kr(Env.asr(2, 1, 3), gate, doneAction: 2);

    Out.ar(out, Pan2.ar(sig * env * amp, 0));
}).add;

// -------------------------------------------------------------------------
// SPECTRAL FREEZE - Suspended, timeless (for day section)
// -------------------------------------------------------------------------
SynthDef(\spectralFreeze, { |out=0, freq=100, amp=0.2, freeze=0.5, shimmer=0.3, gate=1|
    var sig, env, partials, frozen;

    // Build frozen spectrum from harmonics
    partials = Mix.fill(12, { |i|
        var h = i + 1;
        var pFreq = freq * h;
        var pAmp = 0.3 / h;
        // Each partial has slow random movement
        SinOsc.ar(
            pFreq * LFNoise2.kr(shimmer * 0.1 / h).range(0.998, 1.002),
            Rand(0, 2pi),
            pAmp
        )
    });

    // Freeze = less movement
    sig = partials * LFNoise2.kr(freeze.linlin(0, 1, 2, 0.1)).range(0.8, 1);

    // Subtle amplitude modulation
    sig = sig * (1 + (SinOsc.kr(0.05) * 0.1 * (1 - freeze)));

    env = EnvGen.kr(Env.asr(5, 1, 6), gate, doneAction: 2);

    Out.ar(out, Pan2.ar(sig * env * amp, 0));
}).add;

// -------------------------------------------------------------------------
// SWEEP - For dawn/dusk migrations
// -------------------------------------------------------------------------
SynthDef(\migrationSweep, { |out=0, startFreq=300, endFreq=80, dur=20,
    amp=0.2, direction=1, brightness=0.5|
    var sig, env, freq, filtered;

    freq = XLine.kr(startFreq, endFreq, dur);

    // Rich sweep sound
    sig = Saw.ar(freq, 0.3)
        + SinOsc.ar(freq, 0, 0.4)
        + Pulse.ar(freq * 0.5, 0.3, 0.2)
        + SinOsc.ar(freq * 2, 0, 0.15);

    // Moving filter
    filtered = RLPF.ar(sig, freq * brightness.linlin(0, 1, 2, 6), 0.4);

    // Add noise sweep
    filtered = filtered + (BPF.ar(PinkNoise.ar(0.1), freq * 2, 0.2) * brightness);

    env = EnvGen.kr(Env.linen(dur * 0.1, dur * 0.8, dur * 0.1), doneAction: 2);

    Out.ar(out, Pan2.ar(filtered * env * amp, Line.kr(direction * 0.5, direction.neg * 0.5, dur)));
}).add;

// -------------------------------------------------------------------------
// MASTER FX - With distortion and feedback options
// -------------------------------------------------------------------------
SynthDef(\masterFXv4, { |in=0, out=0, delayMix=0.3, delayFb=0.4,
    reverbMix=0.5, reverbRoom=0.7, distortion=0.3, filterMod=0.5,
    masterVol=0.75|
    var sig, delayed, distorted, filtered, reverbed, filterFreq;

    sig = In.ar(in, 2);

    // Soft distortion
    distorted = sig.tanh * (1 + distortion);
    distorted = (sig * (1 - distortion)) + (distorted * distortion);

    // Modulated filter
    filterFreq = filterMod.linexp(0, 1, 200, 12000)
        * LFNoise2.kr(0.5).range(0.8, 1.2);
    filtered = LPF.ar(distorted, filterFreq);

    // Delay with feedback
    delayed = filtered + CombL.ar(filtered, 0.5,
        [0.25, 0.33] * LFNoise2.kr(0.1).range(0.9, 1.1),
        delayFb.linlin(0, 1, 0.5, 4)
    ) * delayMix;

    // Reverb
    reverbed = FreeVerb2.ar(
        delayed[0], delayed[1],
        mix: reverbMix,
        room: reverbRoom,
        damp: 0.4
    );

    // Soft limit
    reverbed = Limiter.ar(reverbed, 0.95);

    Out.ar(out, reverbed * masterVol);
}).add;

"SynthDefs loaded.".postln;
)

// ============================================================================
// SECTION 5: SIMPLE GUI
// ============================================================================

(
~createGUI = {
    var win, makeKnob;

    if(~guiWindow.notNil) { ~guiWindow.close };

    ~guiWindow = Window("Sonificació Oceànica v4 - Creative", Rect(100, 100, 750, 500))
        .background_(Color.gray(0.1))
        .front;

    makeKnob = { |parent, bounds, label, bus, spec|
        var knob, labelView;

        labelView = StaticText(parent, Rect(bounds.left, bounds.top, bounds.width, 16))
            .string_(label)
            .stringColor_(Color.white)
            .align_(\center)
            .font_(Font("Helvetica", 10));

        knob = Knob(parent, Rect(bounds.left + 5, bounds.top + 18, bounds.width - 10, bounds.width - 10))
            .value_(spec.unmap(bus.getSynchronous))
            .color_([Color.gray(0.25), Color.cyan, Color.cyan(0.5), Color.white])
            .action_({ |k| bus.set(spec.map(k.value)) });
    };

    // Master
    StaticText(~guiWindow, Rect(15, 10, 100, 20))
        .string_("MASTER").stringColor_(Color.yellow).font_(Font("Helvetica Bold", 12));

    makeKnob.(~guiWindow, Rect(15, 32, 55, 55), "Volume", ~ctrl[\masterVol], ControlSpec(0, 1));
    makeKnob.(~guiWindow, Rect(75, 32, 55, 55), "Chaos", ~ctrl[\chaos], ControlSpec(0, 1));

    // Swarm
    StaticText(~guiWindow, Rect(15, 105, 100, 18))
        .string_("SWARM").stringColor_(Color.cyan).font_(Font("Helvetica Bold", 11));

    makeKnob.(~guiWindow, Rect(15, 125, 55, 55), "Vol", ~ctrl[\swarmVol], ControlSpec(0, 1));
    makeKnob.(~guiWindow, Rect(75, 125, 55, 55), "Density", ~ctrl[\swarmDensity], ControlSpec(0, 1));
    makeKnob.(~guiWindow, Rect(135, 125, 55, 55), "Pitch", ~ctrl[\swarmPitch], ControlSpec(0, 1));

    // Drone
    StaticText(~guiWindow, Rect(15, 200, 100, 18))
        .string_("DRONE").stringColor_(Color.blue).font_(Font("Helvetica Bold", 11));

    makeKnob.(~guiWindow, Rect(15, 220, 55, 55), "Vol", ~ctrl[\droneVol], ControlSpec(0, 1));
    makeKnob.(~guiWindow, Rect(75, 220, 55, 55), "Dark", ~ctrl[\droneDark], ControlSpec(0, 1));

    // Pulse
    StaticText(~guiWindow, Rect(15, 295, 100, 18))
        .string_("PULSE").stringColor_(Color.green).font_(Font("Helvetica Bold", 11));

    makeKnob.(~guiWindow, Rect(15, 315, 55, 55), "Vol", ~ctrl[\pulseVol], ControlSpec(0, 1));
    makeKnob.(~guiWindow, Rect(75, 315, 55, 55), "Rate", ~ctrl[\pulseRate], ControlSpec(0, 1));

    // Aggression
    StaticText(~guiWindow, Rect(220, 10, 120, 20))
        .string_("AGGRESSION").stringColor_(Color.red).font_(Font("Helvetica Bold", 12));

    makeKnob.(~guiWindow, Rect(220, 32, 55, 55), "Distort", ~ctrl[\distortion], ControlSpec(0, 1));
    makeKnob.(~guiWindow, Rect(280, 32, 55, 55), "Feedback", ~ctrl[\feedback], ControlSpec(0, 1));

    // Effects
    StaticText(~guiWindow, Rect(220, 105, 100, 18))
        .string_("EFFECTS").stringColor_(Color.magenta).font_(Font("Helvetica Bold", 11));

    makeKnob.(~guiWindow, Rect(220, 125, 55, 55), "Reverb", ~ctrl[\reverbMix], ControlSpec(0, 1));
    makeKnob.(~guiWindow, Rect(280, 125, 55, 55), "Delay", ~ctrl[\delayMix], ControlSpec(0, 1));
    makeKnob.(~guiWindow, Rect(340, 125, 55, 55), "Filter", ~ctrl[\filterMod], ControlSpec(0, 1));

    // Data display
    ~dataDisplay = StaticText(~guiWindow, Rect(420, 10, 310, 100))
        .string_("Time: 0:00 / 4:00\nSection: ---\nDepth: --- m\nVelocity: --- m/h")
        .stringColor_(Color.white)
        .font_(Font("Monaco", 12))
        .background_(Color.gray(0.15));

    // Section indicator
    ~sectionDisplay = StaticText(~guiWindow, Rect(420, 120, 310, 50))
        .string_("SECTION: ---")
        .stringColor_(Color.cyan)
        .font_(Font("Helvetica Bold", 18))
        .align_(\center)
        .background_(Color.gray(0.2));

    // Transport
    Button(~guiWindow, Rect(420, 185, 150, 50))
        .states_([
            ["▶ PLAY", Color.black, Color.green],
            ["◼ STOP", Color.white, Color.red]
        ])
        .font_(Font("Helvetica Bold", 18))
        .action_({ |btn|
            if(btn.value == 1) { ~playComposition.value } { ~stopAll.value };
        });

    Button(~guiWindow, Rect(580, 185, 150, 50))
        .states_([["⏺ RECORD", Color.white, Color.gray(0.3)]])
        .font_(Font("Helvetica Bold", 14))
        .action_({
            if(s.isRecording) { s.stopRecording; "Stopped.".postln }
            { s.record; "Recording...".postln };
        });

    // Presets
    StaticText(~guiWindow, Rect(420, 250, 100, 18))
        .string_("PRESETS").stringColor_(Color.white).font_(Font("Helvetica Bold", 11));

    Button(~guiWindow, Rect(420, 275, 100, 35))
        .states_([["NIGHT", Color.white, Color.blue(0.6)]])
        .action_({ ~presetNight.value });

    Button(~guiWindow, Rect(525, 275, 100, 35))
        .states_([["DAY", Color.black, Color.yellow(0.7)]])
        .action_({ ~presetDay.value });

    Button(~guiWindow, Rect(630, 275, 100, 35))
        .states_([["CHAOS", Color.white, Color.red(0.7)]])
        .action_({ ~presetChaos.value });

    // Credits
    StaticText(~guiWindow, Rect(15, 400, 720, 90))
        .string_(
            "SONIFICACIÓ OCEÀNICA v4 - Creative/Aggressive\n" ++
            "Oceànica (Alex Cabrer & Joan Sala) | CLT019 Generalitat de Catalunya\n" ++
            "MALASPINA 2010 - Atlantic Ocean - January 26, 2011\n" ++
            "38 kHz only (120 kHz removed - unusable at depth)"
        )
        .stringColor_(Color.gray(0.5))
        .font_(Font("Helvetica", 10));

    "GUI created.".postln;
};

~createGUI.value;
)

// ============================================================================
// SECTION 6: DATA ACCESS
// ============================================================================

(
~getValue = { |dataArray, timeArray, compTime, compression, windowSec=2|
    var realTime, startIdx, endIdx, values;

    realTime = compTime * compression;
    startIdx = (timeArray.indexOfGreaterThan(realTime - (windowSec * compression * 0.5)) ? 0).max(0);
    endIdx = (timeArray.indexOfGreaterThan(realTime + (windowSec * compression * 0.5)) ? (timeArray.size - 1)).min(timeArray.size - 1);

    values = dataArray.copyRange(startIdx, endIdx);
    values = values.reject({ |v| v.isNil or: { v.isNaN } });

    if(values.size > 0) { values.mean } { 0.5 };
};

~getRaw = { |dataArray, timeArray, compTime, compression|
    var realTime, idx;
    realTime = compTime * compression;
    idx = timeArray.indexOfGreaterThan(realTime) ? (timeArray.size - 1);
    idx = idx.max(0).min(dataArray.size - 1);
    dataArray[idx] ? 0;
};

"Data functions ready.".postln;
)

// ============================================================================
// SECTION 7: COMPOSITION ENGINE
// ============================================================================

(
~fxBus = Bus.audio(s, 2);

~composition = Routine({
    var updateRate = 0.05;
    var currentTime = 0;
    var swarmSynth, droneSynth, pulseSynth, fmSynth, freezeSynth, fxSynth;
    var depthVal, intensityVal, spreadVal, layersVal, velocityVal, hourVal, anomalyVal;
    var depthRawVal, velRawVal;
    var section, lastSection = \none;
    var sectionName, bioState;

    "".postln;
    "╔═══════════════════════════════════════════════════════════════╗".postln;
    "║     SONIFICACIÓ OCEÀNICA v4 - Creative Composition           ║".postln;
    "║     Distinct sonic scenes for each biological phase          ║".postln;
    "╚═══════════════════════════════════════════════════════════════╝".postln;

    // Start effects
    fxSynth = Synth(\masterFXv4, [\in, ~fxBus, \out, 0], addAction: \addToTail);

    // Start voices (all to effects bus)
    swarmSynth = Synth(\granularSwarm, [\out, ~fxBus, \amp, 0]);
    droneSynth = Synth(\deepDrone, [\out, ~fxBus, \amp, 0]);
    pulseSynth = Synth(\rhythmicPulse, [\out, ~fxBus, \amp, 0]);
    fmSynth = Synth(\fmTension, [\out, ~fxBus, \amp, 0]);
    freezeSynth = Synth(\spectralFreeze, [\out, ~fxBus, \amp, 0]);

    0.3.wait;

    while({ currentTime < ~totalDuration }, {
        // Get data values
        depthVal = ~getValue.(~depth, ~times, currentTime, ~compression, 1.5);
        intensityVal = ~getValue.(~intensity, ~times, currentTime, ~compression, 1.5);
        spreadVal = ~getValue.(~spread, ~times, currentTime, ~compression, 1.5);
        layersVal = ~getValue.(~layers, ~times, currentTime, ~compression, 1.5);
        velocityVal = ~getValue.(~velocity, ~times, currentTime, ~compression, 1.5);
        hourVal = ~getValue.(~hour, ~times, currentTime, ~compression, 1);
        anomalyVal = ~getValue.(~anomaly, ~times, currentTime, ~compression, 1.5);

        // Raw for display
        depthRawVal = ~getRaw.(~depthRaw, ~times, currentTime, ~compression);
        velRawVal = ~getRaw.(~velocityRaw, ~times, currentTime, ~compression);

        // Determine section based on HOUR OF DAY (not composition time)
        // This ensures biological accuracy
        section = case
            { hourVal < 0.21 }  { \night1 }     // 00:00 - 05:00
            { hourVal < 0.33 }  { \dawn }       // 05:00 - 08:00
            { hourVal < 0.71 }  { \day }        // 08:00 - 17:00
            { hourVal < 0.88 }  { \dusk }       // 17:00 - 21:00
            { true }            { \night2 };    // 21:00 - 24:00

        // Section info
        #sectionName, bioState = case
            { section == \night1 } { ["NIGHT (Early)", "Surface feeding"] }
            { section == \dawn }   { ["DAWN", "Descending to depth"] }
            { section == \day }    { ["DAY", "Deep refuge"] }
            { section == \dusk }   { ["DUSK", "Ascending to surface"] }
            { true }               { ["NIGHT (Late)", "Surface activity"] };

        // ===== SECTION-BASED SOUND DESIGN =====

        case
        // ----- NIGHT: Active, granular, rhythmic -----
        { section == \night1 or: { section == \night2 } } {
            var swarmAmp = intensityVal.linlin(0, 1, 0.25, 0.6) * ~ctrl[\swarmVol].getSynchronous;
            var droneAmp = 0.15 * ~ctrl[\droneVol].getSynchronous;
            var pulseAmp = layersVal.linlin(0, 1, 0.15, 0.4) * ~ctrl[\pulseVol].getSynchronous;
            var chaos = ~ctrl[\chaos].getSynchronous;

            swarmSynth.set(
                \freq, depthVal.linexp(0, 1, 150, 800),
                \amp, swarmAmp,
                \density, (~ctrl[\swarmDensity].getSynchronous * 40 + 10) * intensityVal.linlin(0, 1, 0.5, 1.5),
                \spread, spreadVal.linlin(0, 1, 0.3, 0.9),
                \chaos, chaos,
                \brightness, depthVal.linlin(0, 1, 0.3, 0.9)
            );

            droneSynth.set(
                \freq, depthVal.linexp(0, 1, 40, 100),
                \amp, droneAmp,
                \darkness, 0.3,
                \movement, spreadVal * 0.5
            );

            pulseSynth.set(
                \freq, depthVal.linexp(0, 1, 50, 150),
                \amp, pulseAmp,
                \rate, ~ctrl[\pulseRate].getSynchronous.linlin(0, 1, 1, 6) * layersVal.linlin(0, 1, 0.5, 1.5),
                \decay, spreadVal.linlin(0, 1, 0.1, 0.4),
                \brightness, 0.7
            );

            fmSynth.set(\amp, 0);
            freezeSynth.set(\amp, 0);
        }

        // ----- DAWN: Tension building, FM harsh, descending -----
        { section == \dawn } {
            var fmAmp = 0.3 * ~ctrl[\swarmVol].getSynchronous;
            var droneAmp = 0.35 * ~ctrl[\droneVol].getSynchronous;
            var swarmAmp = intensityVal.linlin(0, 1, 0.1, 0.3) * ~ctrl[\swarmVol].getSynchronous;

            // Trigger sweep on section change
            if(lastSection != \dawn) {
                Synth(\migrationSweep, [\out, ~fxBus, \startFreq, 400, \endFreq, 80,
                    \dur, 18, \amp, 0.2, \direction, 1, \brightness, 0.6]);
                ">>> DAWN: Descent sweep triggered <<<".postln;
            };

            swarmSynth.set(
                \freq, depthVal.linexp(0, 1, 100, 400),
                \amp, swarmAmp,
                \density, 15,
                \chaos, 0.7,
                \brightness, 0.4
            );

            droneSynth.set(
                \freq, depthVal.linexp(0, 1, 35, 80),
                \amp, droneAmp,
                \darkness, depthVal.linlin(0, 1, 0.3, 0.8),
                \movement, 0.4
            );

            fmSynth.set(
                \freq, depthVal.linexp(0, 1, 80, 200),
                \amp, fmAmp * velocityVal.linlin(0, 1, 1.5, 0.3),  // More when descending fast
                \modRatio, 2.5,
                \modDepth, velocityVal.linlin(0, 1, 5, 1),
                \brightness, 0.5
            );

            pulseSynth.set(\amp, 0.1);
            freezeSynth.set(\amp, 0);
        }

        // ----- DAY: Suspended, frozen, minimal -----
        { section == \day } {
            var freezeAmp = 0.35 * ~ctrl[\droneVol].getSynchronous;
            var droneAmp = 0.4 * ~ctrl[\droneVol].getSynchronous;
            var swarmAmp = 0.08 * ~ctrl[\swarmVol].getSynchronous;

            freezeSynth.set(
                \freq, depthVal.linexp(0, 1, 60, 150),
                \amp, freezeAmp,
                \freeze, 0.8,
                \shimmer, spreadVal * 0.3
            );

            droneSynth.set(
                \freq, depthVal.linexp(0, 1, 30, 70),
                \amp, droneAmp,
                \darkness, 0.8,
                \movement, 0.1
            );

            swarmSynth.set(
                \freq, depthVal.linexp(0, 1, 200, 400),
                \amp, swarmAmp,
                \density, 5,
                \chaos, 0.2,
                \brightness, 0.2
            );

            fmSynth.set(\amp, 0);
            pulseSynth.set(\amp, 0.05);
        }

        // ----- DUSK: Release, ascending, anticipation -----
        { section == \dusk } {
            var swarmAmp = intensityVal.linlin(0, 1, 0.15, 0.4) * ~ctrl[\swarmVol].getSynchronous;
            var fmAmp = 0.2 * ~ctrl[\swarmVol].getSynchronous;
            var droneAmp = 0.25 * ~ctrl[\droneVol].getSynchronous;

            // Trigger ascending sweep
            if(lastSection != \dusk) {
                Synth(\migrationSweep, [\out, ~fxBus, \startFreq, 80, \endFreq, 350,
                    \dur, 18, \amp, 0.2, \direction, -1, \brightness, 0.7]);
                ">>> DUSK: Ascent sweep triggered <<<".postln;
            };

            swarmSynth.set(
                \freq, depthVal.linexp(0, 1, 150, 600),
                \amp, swarmAmp,
                \density, 20 * velocityVal.linlin(0, 1, 0.5, 1.5),
                \chaos, 0.5,
                \brightness, depthVal.linlin(0, 1, 0.4, 0.8)
            );

            droneSynth.set(
                \freq, depthVal.linexp(0, 1, 40, 100),
                \amp, droneAmp,
                \darkness, depthVal.linlin(0, 1, 0.6, 0.3),
                \movement, 0.5
            );

            fmSynth.set(
                \freq, depthVal.linexp(0, 1, 100, 250),
                \amp, fmAmp * velocityVal.linlin(0, 1, 0.3, 1.2),
                \modRatio, 1.5,
                \modDepth, 2,
                \brightness, 0.6
            );

            freezeSynth.set(\amp, 0);
            pulseSynth.set(\amp, layersVal * 0.15);
        };

        lastSection = section;

        // ===== EFFECTS =====
        fxSynth.set(
            \delayMix, ~ctrl[\delayMix].getSynchronous,
            \delayFb, ~ctrl[\feedback].getSynchronous,
            \reverbMix, ~ctrl[\reverbMix].getSynchronous,
            \reverbRoom, case
                { section == \day } { 0.9 }
                { section == \night1 or: { section == \night2 } } { 0.5 }
                { true } { 0.7 },
            \distortion, ~ctrl[\distortion].getSynchronous * case
                { section == \dawn } { 1.5 }
                { section == \day } { 0.3 }
                { true } { 1 },
            \filterMod, ~ctrl[\filterMod].getSynchronous,
            \masterVol, ~ctrl[\masterVol].getSynchronous
        );

        // ===== GUI UPDATE =====
        {
            var mins = (currentTime / 60).floor.asInteger;
            var secs = (currentTime % 60).floor.asInteger;
            var realHour = (hourVal * 24).floor.asInteger;
            var realMin = ((hourVal * 24 * 60) % 60).floor.asInteger;

            if(~dataDisplay.notNil and: { ~dataDisplay.isClosed.not }) {
                ~dataDisplay.string_(
                    "Time: " ++ mins ++ ":" ++ if(secs < 10) { "0" } { "" } ++ secs ++ " / 4:00\n" ++
                    "Real: " ++ if(realHour < 10) { "0" } { "" } ++ realHour ++ ":" ++
                        if(realMin < 10) { "0" } { "" } ++ realMin ++ " UTC\n" ++
                    "Depth: " ++ depthRawVal.round(1) ++ " m\n" ++
                    "Velocity: " ++ velRawVal.round(1) ++ " m/h"
                );
            };

            if(~sectionDisplay.notNil and: { ~sectionDisplay.isClosed.not }) {
                ~sectionDisplay.string_(sectionName ++ "\n" ++ bioState);
                ~sectionDisplay.stringColor_(case
                    { section == \night1 or: { section == \night2 } } { Color.cyan }
                    { section == \dawn } { Color.new255(255, 150, 50) }
                    { section == \day } { Color.yellow }
                    { section == \dusk } { Color.new255(255, 100, 150) }
                    { true } { Color.white }
                );
            };
        }.defer;

        updateRate.wait;
        currentTime = currentTime + updateRate;
    });

    // Fade out
    "Ending...".postln;
    [swarmSynth, droneSynth, pulseSynth, fmSynth, freezeSynth].do(_.set(\gate, 0));
    5.wait;
    fxSynth.free;

    { ~sectionDisplay.string_("COMPLETE\n24h cycle finished") }.defer;
    "Done.".postln;
});

~playComposition = {
    ~composition.reset;
    ~composition.play;
};

~stopAll = {
    ~composition.stop;
    s.freeAll;
    "Stopped.".postln;
};

"Composition engine ready.".postln;
)

// ============================================================================
// SECTION 8: PRESETS
// ============================================================================

(
~presetNight = {
    ~ctrl[\masterVol].set(0.75);
    ~ctrl[\chaos].set(0.6);
    ~ctrl[\swarmVol].set(0.7);
    ~ctrl[\swarmDensity].set(0.7);
    ~ctrl[\droneVol].set(0.3);
    ~ctrl[\droneDark].set(0.3);
    ~ctrl[\pulseVol].set(0.5);
    ~ctrl[\pulseRate].set(0.6);
    ~ctrl[\distortion].set(0.3);
    ~ctrl[\feedback].set(0.4);
    ~ctrl[\reverbMix].set(0.4);
    ~ctrl[\delayMix].set(0.3);
    ~ctrl[\filterMod].set(0.7);
    "Preset: NIGHT - Active, granular, rhythmic".postln;
};

~presetDay = {
    ~ctrl[\masterVol].set(0.7);
    ~ctrl[\chaos].set(0.2);
    ~ctrl[\swarmVol].set(0.2);
    ~ctrl[\swarmDensity].set(0.2);
    ~ctrl[\droneVol].set(0.8);
    ~ctrl[\droneDark].set(0.8);
    ~ctrl[\pulseVol].set(0.1);
    ~ctrl[\pulseRate].set(0.2);
    ~ctrl[\distortion].set(0.1);
    ~ctrl[\feedback].set(0.5);
    ~ctrl[\reverbMix].set(0.7);
    ~ctrl[\delayMix].set(0.4);
    ~ctrl[\filterMod].set(0.3);
    "Preset: DAY - Suspended, frozen, deep".postln;
};

~presetChaos = {
    ~ctrl[\masterVol].set(0.8);
    ~ctrl[\chaos].set(0.9);
    ~ctrl[\swarmVol].set(0.8);
    ~ctrl[\swarmDensity].set(0.9);
    ~ctrl[\droneVol].set(0.5);
    ~ctrl[\droneDark].set(0.4);
    ~ctrl[\pulseVol].set(0.6);
    ~ctrl[\pulseRate].set(0.8);
    ~ctrl[\distortion].set(0.6);
    ~ctrl[\feedback].set(0.6);
    ~ctrl[\reverbMix].set(0.5);
    ~ctrl[\delayMix].set(0.5);
    ~ctrl[\filterMod].set(0.6);
    "Preset: CHAOS - Maximum intensity".postln;
};

"Presets ready.".postln;
)

// ============================================================================
// END
// ============================================================================

/*
HOW TO USE:
1. Execute sections 1-8 in order
2. Press PLAY or run ~playComposition.value
3. Use knobs to shape sound in real-time
4. Try presets: NIGHT, DAY, CHAOS

SONIC DESIGN:
- Night: Granular swarms, rhythmic pulses, active
- Dawn: FM tension, descending sweep, building pressure
- Day: Spectral freeze, deep drone, suspended time
- Dusk: Rising sweep, anticipation, releasing tension
- Each section has distinct character while flowing smoothly

SCIENTIFIC ACCURACY:
- Only uses 38 kHz data (120 kHz was broken)
- Velocity properly smoothed (30-min window)
- Section timing based on actual hour of day from data
*/
