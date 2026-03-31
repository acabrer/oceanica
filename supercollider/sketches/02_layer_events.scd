/*
================================================================================
SONIFICACIO OCEANICA — Sketch 02: Layer Events
================================================================================

When scattering layers in the water column are born (a new peak in acoustic
backscatter appears) or die (a peak vanishes), a structural boundary has shifted.
These are discrete ecological events — a layer of organisms coalescing, or
dispersing — and they happen thousands of times across a 24-hour cycle.

This sketch makes those structural transitions audible as two complementary
sub-voices:

  EVENT BELLS   — percussive Klank impulses on layer births and deaths.
                  Births ring high and metallic (2-8 kHz).
                  Deaths thud low and hollow (60-200 Hz).

  LAYER DRONES  — four quiet sustained resonances tracking persistent layer
                  identities (800-2500 Hz). Young layers are barely audible;
                  long-lived layers gain subtle presence.

Both are modulated by the regime change score, which drives reverb expansion
during major structural transitions (dawn/dusk).

MAPPING:
  layer_event_type   -> event bell trigger (birth=high bell, death=low thud)
  layer_event_depth  -> event bell pitch (shallow=higher, deep=lower)
  regime_change_score -> reverb size + event decay time (transitions = spacious)
  regime_id_norm     -> drone detuning spread (each regime has different timbre)
  tracked_layer_depth -> drone pitch (4 independent frequencies, 800-2500 Hz)
  tracked_layer_age  -> drone amplitude (young=silent, old=subtle presence)

SPECTRAL SEPARATION FROM SKETCH 01:
  Sketch 01 occupies 90-560 Hz (FM voice).
  Birth bells:  2000-8000 Hz (above)
  Death thuds:  60-200 Hz (below, very short decay — no masking)
  Layer drones: 800-2500 Hz (gap between Sketch 01 ceiling and birth bells)

DATA: sonification_sc_v8_20110126.json (same file as Sketch 01)
      Uses fields that Sketch 01 does NOT consume.

HOW TO USE:
  1. Execute sections 1-5 in order (Cmd+Enter each block)
  2. Press PLAY or spacebar
  3. Listen. Sparse metallic pings against a subtle harmonic backdrop.

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
~times = ~jsonData["38kHz"]["time_seconds"].asArray.collect(_.asFloat);
~hour  = ~jsonData["38kHz"]["hour_norm"].asArray.collect(_.asFloat);

// --- Layer events (discrete: 0=death, 0.5=none, 1=birth) ---
~layerEvent      = ~jsonData["38kHz"]["layer_event_type"].asArray.collect(_.asFloat);
~layerEventDepth = ~jsonData["38kHz"]["layer_event_depth_norm"].asArray.collect(_.asFloat);

// --- Regime changes (continuous [0,1]) ---
~regimeScore = ~jsonData["38kHz"]["regime_change_score"].asArray.collect(_.asFloat);
~regimeId    = ~jsonData["38kHz"]["regime_id_norm"].asArray.collect(_.asFloat);

// --- Tracked layer trajectories (4 persistent slots) ---
// depth_norm: 0=deep, 1=shallow (matches dvm_depth_norm convention)
// age_norm: 0=just born/empty, 1=max age (~67 min)
~trackedDepth = Array.fill(4, { |slot|
    ~jsonData["38kHz"]["tracked_layer_" ++ slot.asString ++ "_depth_norm"].asArray.collect(_.asFloat);
});
~trackedAge = Array.fill(4, { |slot|
    ~jsonData["38kHz"]["tracked_layer_" ++ slot.asString ++ "_age_norm"].asArray.collect(_.asFloat);
});

// --- Auto-detected timing from extractor (hours UTC) ---
~sunriseH   = ~jsonData["info"]["sunrise_h"].asFloat   ? 5.75;
~sunsetH    = ~jsonData["info"]["sunset_h"].asFloat    ? 20.25;
~dawnEndH   = ~jsonData["info"]["dawn_end_h"].asFloat  ? 8.5;
~duskStartH = ~jsonData["info"]["dusk_start_h"].asFloat ? 19.5;
~sunriseFrac   = ~sunriseH / 24;
~sunsetFrac    = ~sunsetH / 24;
~dawnEndFrac   = ~dawnEndH / 24;
~duskStartFrac = ~duskStartH / 24;

// --- Metadata for display ---
~nRegimes       = ~jsonData["info"]["n_regimes"].asFloat ? 35;
~maxTrackedAge  = ~jsonData["info"]["max_tracked_layer_age"].asFloat ? 1344;

// --- Playback timing (same as Sketch 01) ---
~speed          = 1.0;
~totalDuration  = 480.0;   // 8 minutes at 1x
~compression    = ~times.last / ~totalDuration;
~masterVol      = 0.75;

// --- Visualization arrays (500 columns) ---
~vizN = 500;

// Pre-compute hour fraction per viz column (for section coloring)
~vizHourTrace = Array.fill(~vizN, { |i|
    var idx = ((i / ~vizN) * (~hour.size - 1)).round.asInteger.clip(0, ~hour.size - 1);
    ~hour[idx];
});

// Event scatter: accumulates during playback [vizIdx] -> [Array of [type, depthNorm]]
~vizEvents = Array.fill(~vizN, { nil });

// Tracked layer preview: pre-compute for dim background trace
~vizTrackedPreview = Array.fill(4, { |slot|
    Array.fill(~vizN, { |i|
        var idx = ((i / ~vizN) * (~trackedDepth[slot].size - 1)).round.asInteger
            .clip(0, ~trackedDepth[slot].size - 1);
        ~trackedDepth[slot][idx] ? 0;
    });
});

// --- Shared GUI state ---
~currentTime    = 0.0;
~playing        = false;
~lastEventType  = nil;
~lastEventDepth = 0.0;
~regimeVal      = 0.0;
~activeLayers   = 0;
~eventCount     = 0;

"".postln;
"=== LAYER EVENTS DATA LOADED (v8) ===".postln;
("Points: " ++ ~times.size).postln;
("Duration: " ++ ~times.last.round(1) ++ "s (" ++ (~times.last / 3600).round(0.1) ++ "h)").postln;
("Compression: " ++ ~compression.round(1) ++ "x -> " ++ ~totalDuration ++ "s").postln;
("Regimes: " ++ ~nRegimes.round.asInteger).postln;
("Sections: sunrise=" ++ ~sunriseH.round(0.1) ++ "h, dawn_end=" ++ ~dawnEndH.round(0.1)
    ++ "h, dusk_start=" ++ ~duskStartH.round(0.1) ++ "h, sunset=" ++ ~sunsetH.round(0.1) ++ "h").postln;
{
    var eventCount = ~layerEvent.count({ |v| v != 0.5 });
    var birthCount = ~layerEvent.count({ |v| v == 1.0 });
    var deathCount = ~layerEvent.count({ |v| v == 0.0 });
    ("Events: " ++ eventCount ++ " total (" ++ birthCount ++ " births, " ++ deathCount ++ " deaths)").postln;
}.value;
"Features loaded: layer_event, regime_score, tracked_layers[4]".postln;
)


// ============================================================================
// SECTION 3: SYNTHDEFS
// ============================================================================

(
// -------------------------------------------------------------------------
// eventBell — Klank resonant filter bank for layer births and deaths
//
// Each event spawns a one-shot Synth (doneAction: 2, self-freeing).
// Birth: bright, metallic, inharmonic bell ratios (2-8 kHz).
// Death: dark, hollow, more harmonic ratios (60-200 Hz).
//
// The biological meaning:
//   Birth bell = a new scattering layer has coalesced — organisms are
//   organizing into a coherent structure. Something has arrived.
//   Death thud = a layer has dissolved — organisms have dispersed or
//   descended out of detection range. Something has departed.
// -------------------------------------------------------------------------
SynthDef(\eventBell, { |out=0, freq=4000, amp=0.15, type=1, pan=0, decay=0.5|
    var exciter, sig, env;

    // Birth: bright PinkNoise click through inharmonic Klank
    // Death: soft BrownNoise thud through harmonic Klank
    exciter = Select.ar(type, [
        // type=0: DEATH — soft, dark exciter
        EnvGen.ar(Env.perc(0.002, 0.02), doneAction: 0) * BrownNoise.ar(0.5),
        // type=1: BIRTH — sharp, bright exciter
        EnvGen.ar(Env.perc(0.001, 0.005), doneAction: 0) * PinkNoise.ar(0.5)
    ]);

    sig = Select.ar(type, [
        // type=0: DEATH — harmonic, dark ratios (4 partials)
        DynKlank.ar(
            `[
                [1, 1.5, 2.0, 2.8] * freq,              // more harmonic
                [1, 0.5, 0.2, 0.1],                       // amplitudes
                [0.2, 0.15, 0.1, 0.08] * decay             // short decay
            ],
            exciter
        ),
        // type=1: BIRTH — inharmonic, bell-like ratios (5 partials)
        DynKlank.ar(
            `[
                [1, 2.756, 4.091, 5.385, 6.7] * freq,    // inharmonic bell
                [1, 0.6, 0.3, 0.2, 0.1],                  // amplitudes
                [0.8, 0.5, 0.3, 0.2, 0.15] * decay         // longer ring
            ],
            exciter
        )
    ]);

    // Spectral containment
    sig = Select.ar(type, [
        LPF.ar(sig, 400),    // death: darken, keep below 400 Hz
        HPF.ar(sig, 1500)    // birth: brighten, keep above 1500 Hz
    ]);

    // Self-freeing envelope
    env = EnvGen.ar(Env.perc(0.001, decay * 1.5), doneAction: 2);
    sig = sig * env * amp;

    Out.ar(out, Pan2.ar(sig, pan));
}).add;


// -------------------------------------------------------------------------
// layerDrone — Tracked layer resonance (4 persistent instances)
//
// Each tracked layer slot gets one of these synths. They run continuously,
// amplitude-gated by the layer's age. Young layers are silent; established
// layers emit a quiet, focused tone in the 800-2500 Hz range.
//
// The biological meaning:
//   A persistent tone = a stable scattering layer exists at that depth.
//   The tone's pitch tells you the depth. Its loudness tells you how long
//   the layer has persisted. When it fades, the layer has dissolved.
// -------------------------------------------------------------------------
SynthDef(\layerDrone, { |out=0, freq=1200, amp=0.0, regimeCol=0.0,
    gate=1, lagTime=2.0|
    var sig, env, spread, breath;

    // Smooth all inputs — biological processes are continuous
    freq      = Lag.kr(freq, lagTime);
    amp       = Lag.kr(amp, lagTime);
    regimeCol = Lag.kr(regimeCol, lagTime * 2);

    // Regime coloring: each regime subtly changes the detuning spread
    // This means different "eras" of the ocean sound slightly different
    spread = regimeCol * 3.0;  // 0-3 Hz detuning spread

    // Three detuned sines — tight cluster creates focused resonance
    sig = SinOsc.ar(freq - spread) + SinOsc.ar(freq) + SinOsc.ar(freq + spread);
    sig = sig / 3;

    // Organic breath: slow random amplitude modulation
    breath = LFNoise1.kr(0.5).range(0.7, 1.0);
    sig = sig * breath;

    // Bandpass filter for focused resonance
    sig = BPF.ar(sig, freq.clip(200, 8000), 0.1);

    // Overall envelope (gate-driven, for cleanup)
    env = EnvGen.kr(Env.asr(2, 1, 3), gate, doneAction: 2);
    sig = sig * env * amp;

    Out.ar(out, Pan2.ar(sig, 0));  // center-panned (events provide spatial scatter)
}).add;


// -------------------------------------------------------------------------
// layerEventReverb — Regime-driven spatial processor
//
// Longer tails than Sketch 01's reverb. The regime change score modulates
// the room size: during structural transitions (dawn/dusk), the space
// expands dramatically — events ring out into a vast, shifting acoustic
// environment.
// -------------------------------------------------------------------------
SynthDef(\layerEventReverb, { |in, out=0, mix=0.25, room=0.7, damp=0.2,
    masterVol=0.75|
    var sig;
    mix  = Lag.kr(mix, 0.5);
    room = Lag.kr(room, 1.0);
    damp = Lag.kr(damp, 1.0);
    sig = In.ar(in, 2);
    sig = FreeVerb2.ar(sig[0], sig[1], mix, room, damp);
    sig = HPF.ar(sig, 50);     // allow some death thud sub through
    sig = Limiter.ar(sig, 0.85);
    Out.ar(out, sig * masterVol);
}).add;

"SynthDefs loaded: eventBell, layerDrone (x4), layerEventReverb.".postln;
)


// ============================================================================
// SECTION 4: GUI
// ============================================================================

(
~createGUI = {
    var win;

    if(~guiWindow.notNil) { ~guiWindow.close };

    ~guiWindow = Window("Sketch 02: Layer Events", Rect(80, 500, 740, 380))
        .background_(Color.gray(0.06))
        .front;
    win = ~guiWindow;

    // --- Header ---
    StaticText(win, Rect(15, 8, 550, 20))
        .string_("LAYER EVENTS  —  Structural Ecology, 24 Hours")
        .stringColor_(Color.white)
        .font_(Font("Helvetica Bold", 13));

    // --- Event scatter plot (710 x 200) ---
    ~scatterView = UserView(win, Rect(15, 32, 710, 200))
        .background_(Color.black)
        .animate_(true)
        .frameRate_(15);

    ~scatterView.drawFunc = { |uv|
        var w = uv.bounds.width;
        var h = uv.bounds.height;
        var pct = (~currentTime / ~totalDuration).clip(0, 1);
        var depthMax = 1000;

        // --- Section background bands ---
        ~vizN.do { |i|
            var x = (i / ~vizN) * w;
            var colW = (w / ~vizN).ceil + 1;
            var hourFrac = ~vizHourTrace[i] ? 0;
            var bgColor = case
                { hourFrac < ~sunriseFrac }  { Color.new255(10, 18, 45, 50) }
                { hourFrac < ~dawnEndFrac }  { Color.new255(90, 55, 15, 45) }
                { hourFrac < ~duskStartFrac }{ Color.new255(65, 60, 20, 30) }
                { hourFrac < ~sunsetFrac }   { Color.new255(90, 35, 55, 50) }
                { true }                     { Color.new255(10, 18, 45, 50) };

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

        // --- Tracked layer preview (dim traces for all 4 slots) ---
        4.do { |slot|
            var preview = ~vizTrackedPreview[slot];
            var color = [
                Color.new255(100, 200, 255, 20),   // slot 0: cyan
                Color.new255(255, 180, 100, 20),   // slot 1: amber
                Color.new255(150, 255, 150, 15),   // slot 2: green
                Color.new255(255, 150, 255, 15)    // slot 3: pink
            ][slot];

            Pen.strokeColor_(color);
            Pen.width_(1);
            Pen.moveTo(0 @ ((1.0 - (preview[0] ? 0)) * h).clip(0, h));
            ~vizN.do { |i|
                var x = (i / (~vizN - 1)) * w;
                var depthNorm = preview[i] ? 0;
                var y;
                if(depthNorm > 0.01) {
                    y = ((1.0 - depthNorm) * depthMax / depthMax) * h;
                    Pen.lineTo(x @ y);
                } {
                    Pen.moveTo(x @ (h * 0.5));
                };
            };
            Pen.stroke;
        };

        // --- Accumulated events (scatter dots) ---
        ~vizN.do { |i|
            var events = ~vizEvents[i];
            if(events.notNil) {
                events.do { |ev|
                    var evType = ev[0];
                    var evDepth = ev[1];
                    var x = (i / (~vizN - 1)) * w;
                    var y = (evDepth * depthMax / depthMax) * h;
                    // layer_event_depth_norm: 0=surface, 1=deep
                    // So y = evDepth * h (deeper events lower on screen)

                    if(evType == 1.0) {
                        // Birth: cyan upward triangle
                        Pen.fillColor_(Color.new255(0, 220, 220, 180));
                        Pen.moveTo((x - 3) @ (y + 2));
                        Pen.lineTo(x @ (y - 4));
                        Pen.lineTo((x + 3) @ (y + 2));
                        Pen.fill;
                    } {
                        // Death: orange-red downward triangle
                        Pen.fillColor_(Color.new255(255, 100, 50, 160));
                        Pen.moveTo((x - 3) @ (y - 2));
                        Pen.lineTo(x @ (y + 4));
                        Pen.lineTo((x + 3) @ (y - 2));
                        Pen.fill;
                    };
                };
            };
        };

        // --- Green playhead ---
        Pen.strokeColor_(Color.new255(0, 220, 100, 180));
        Pen.width_(1.5);
        Pen.line((pct * w) @ 0, (pct * w) @ h);
        Pen.stroke;
    };

    // --- Regime score bar (710 x 16) ---
    ~regimeView = UserView(win, Rect(15, 236, 710, 16))
        .background_(Color.gray(0.04))
        .animate_(true)
        .frameRate_(15);

    ~regimeView.drawFunc = { |uv|
        var w = uv.bounds.width;
        var h = uv.bounds.height;
        var pct = (~currentTime / ~totalDuration).clip(0, 1);

        // Pre-computed regime score preview
        ~vizN.do { |i|
            var x = (i / ~vizN) * w;
            var colW = (w / ~vizN).ceil + 1;
            var idx = ((i / ~vizN) * (~regimeScore.size - 1)).round.asInteger
                .clip(0, ~regimeScore.size - 1);
            var score = ~regimeScore[idx] ? 0;
            var alpha = (score * 200).clip(10, 200).asInteger;
            // Color: blue(stable) -> yellow(transitioning) -> red(major change)
            var r = (score * 255).clip(0, 255).asInteger;
            var g = ((1.0 - (score - 0.5).abs * 2) * 200).clip(0, 200).asInteger;
            var b = ((1.0 - score) * 180).clip(0, 180).asInteger;

            Pen.fillColor_(Color.new255(r, g, b, alpha));
            Pen.fillRect(Rect(x, 0, colW, h));
        };

        // Playhead
        Pen.strokeColor_(Color.new255(0, 220, 100, 150));
        Pen.width_(1);
        Pen.line((pct * w) @ 0, (pct * w) @ h);
        Pen.stroke;

        // Label
        Pen.stringAtPoint("regime", 3 @ 1,
            Font("Helvetica", 7), Color.gray(0.35));
    };

    // --- Info display ---
    ~infoDisplay = StaticText(win, Rect(15, 258, 430, 50))
        .string_("UTC: --:--   Last event: ---   Depth: ---m\nRegime: ----   Active layers: -   Events: 0")
        .stringColor_(Color.new255(170, 195, 215))
        .font_(Font("Monaco", 10))
        .background_(Color.gray(0.09));

    ~sectionDisplay = StaticText(win, Rect(455, 258, 270, 50))
        .string_("---\nAwaiting playback")
        .stringColor_(Color.gray(0.4))
        .align_(\center)
        .font_(Font("Helvetica Bold", 12))
        .background_(Color.gray(0.09));

    // --- Transport ---
    ~playBtn = Button(win, Rect(15, 320, 90, 32))
        .states_([
            ["PLAY", Color.black, Color.new255(50, 220, 80)],
            ["STOP", Color.white, Color.new255(200, 50, 50)]
        ])
        .font_(Font("Helvetica Bold", 12))
        .action_({ |btn|
            if(btn.value == 1) { ~playComposition.value } { ~stopAll.value };
        });

    // --- Volume ---
    StaticText(win, Rect(125, 325, 30, 20))
        .string_("Vol")
        .stringColor_(Color.gray(0.5))
        .font_(Font("Helvetica", 10));
    Slider(win, Rect(155, 325, 110, 20))
        .value_(~masterVol)
        .background_(Color.gray(0.18))
        .knobColor_(Color.white)
        .action_({ |sl| ~masterVol = sl.value });

    // --- Speed ---
    StaticText(win, Rect(290, 325, 45, 20))
        .string_("Speed")
        .stringColor_(Color.gray(0.5))
        .font_(Font("Helvetica", 10));

    ~speedBtns = Array.new(4);
    [0.5, 1.0, 2.0, 4.0].do { |spd, i|
        var lbl = if(spd < 1) { "0.5x" } { spd.asInteger.asString ++ "x" };
        ~speedBtns = ~speedBtns.add(
            Button(win, Rect(340 + (i * 52), 320, 46, 28))
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
    Button(win, Rect(560, 320, 60, 28))
        .states_([["REC", Color.white, Color.gray(0.25)]])
        .font_(Font("Helvetica Bold", 10))
        .action_({
            if(s.isRecording)
                { s.stopRecording; "Stopped recording.".postln }
                { s.record;        "Recording...".postln };
        });

    // --- Credits ---
    StaticText(win, Rect(15, 355, 710, 25))
        .string_(
            "Sonificacio Oceanica — Sketch 02: Layer Events  |  " ++
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

// Event scanner: returns all events in a time window
// Returns Array of [event_type, depth_norm, regime_score, ping_index]
~getEventsInWindow = { |compTime, compression|
    var realTime, halfWin, startIdx, endIdx, events;
    realTime = compTime * compression;
    halfWin = compression * 0.025;  // half of one 50ms tick in real time
    startIdx = (~times.indexOfGreaterThan(realTime - halfWin) ? 0).max(0);
    endIdx = (~times.indexOfGreaterThan(realTime + halfWin) ? (~times.size - 1)).min(~times.size - 1);
    events = List.new;
    (startIdx..endIdx).do { |i|
        if(~layerEvent[i] != 0.5) {
            events.add([~layerEvent[i], ~layerEventDepth[i], ~regimeScore[i], i]);
        };
    };
    events;
};

// --- FX bus ---
~fxBus02 = Bus.audio(s, 2);
~playing = false;

// --- The composition: structural events, four persistent voices ---
~composition = Routine({
    var updateRate = 0.05;  // 50ms = 20 Hz
    var currentTime = 0;
    var fxSynth;
    var droneSynths;

    // Event triggering state
    var cooldown = 0;        // frames until next event allowed
    var cooldownFrames = 3;  // 150ms minimum gap between events
    var eventCount = 0;

    // Section tracking
    var section = \night1, lastSection = \none;
    var sectionName = "---", bioState = "";
    var sectionHoldCount = 0, candidateSection = \none;
    var hourVal;

    // Data values
    var regimeVal, regimeIdVal;

    "".postln;
    "=======================================================".postln;
    "   SKETCH 02: LAYER EVENTS".postln;
    "   Structural ecology of the water column.".postln;
    "=======================================================".postln;
    "".postln;

    // Start reverb (tail of signal chain)
    fxSynth = Synth(\layerEventReverb, [
        \in, ~fxBus02, \out, 0, \masterVol, ~masterVol
    ], addAction: \addToTail);

    // Start 4 layer drone synths (silent initially, will be modulated by data)
    droneSynths = Array.fill(4, { |slot|
        Synth(\layerDrone, [
            \out, ~fxBus02, \freq, 1200, \amp, 0.0, \regimeCol, 0.0, \gate, 1
        ]);
    });

    0.5.wait;
    ~playing = true;

    while({ currentTime < ~totalDuration and: { ~playing } }, {

        if(~playing.not) {
            updateRate.wait;
            currentTime = currentTime + updateRate;
        } {

        // ===== READ DATA =====
        hourVal   = ~getValue.(~hour, ~times, currentTime, ~compression, 1.5 * ~speed);
        regimeVal = ~getValue.(~regimeScore, ~times, currentTime, ~compression, 2.0 * ~speed);
        regimeIdVal = ~getValue.(~regimeId, ~times, currentTime, ~compression, 2.0 * ~speed);

        // ===== EVENT BELL TRIGGERING =====
        if(cooldown > 0) {
            cooldown = cooldown - 1;
        } {
            var events = ~getEventsInWindow.(currentTime, ~compression);
            if(events.size > 0) {
                // Select the event with highest regime_change_score (structural priority)
                var bestEvent = events.sort({ |a, b| a[2] > b[2] }).first;
                var evType  = bestEvent[0];   // 0.0=death, 1.0=birth
                var evDepth = bestEvent[1];   // 0=surface, 1=deep
                var evRegime = bestEvent[2];  // regime score at this ping
                var bellFreq, bellDecay, bellAmp, bellType, bellPan;

                bellPan = 1.4.rand - 0.7;  // random scatter -0.7 to 0.7

                if(evType == 1.0) {
                    // BIRTH: high metallic bell
                    // layer_event_depth_norm: 0=surface, 1=deep
                    // shallow births ring higher, deep births ring lower
                    bellFreq = evDepth.linexp(0, 1, 8000, 2000);
                    bellDecay = evRegime.linlin(0, 1, 0.3, 0.8);
                    bellAmp = 0.12 * ~masterVol;
                    bellType = 1;
                } {
                    // DEATH: low hollow thud
                    bellFreq = evDepth.linexp(0, 1, 200, 60);
                    bellDecay = 0.05 + (0.15 * evRegime);
                    bellAmp = 0.10 * ~masterVol;
                    bellType = 0;
                };

                Synth(\eventBell, [
                    \out, ~fxBus02,
                    \freq, bellFreq,
                    \amp, bellAmp,
                    \type, bellType,
                    \pan, bellPan,
                    \decay, bellDecay
                ]);

                cooldown = cooldownFrames;
                eventCount = eventCount + 1;

                // Update GUI state
                ~lastEventType = if(bellType == 1) { "BIRTH" } { "DEATH" };
                ~lastEventDepth = if(bellType == 1) {
                    evDepth.linlin(0, 1, 0, 1000)  // depth in meters
                } {
                    evDepth.linlin(0, 1, 0, 1000)
                };

                // Accumulate event in visualization
                {
                    var vizIdx = ((currentTime / ~totalDuration) * ~vizN).floor.asInteger.clip(0, ~vizN - 1);
                    if(~vizEvents[vizIdx].isNil) { ~vizEvents[vizIdx] = List.new };
                    ~vizEvents[vizIdx].add([evType, evDepth]);
                }.value;
            };
        };

        // ===== TRACKED LAYER DRONES (continuous update) =====
        4.do { |slot|
            var depthVal = ~getValue.(~trackedDepth[slot], ~times, currentTime, ~compression, 2.0 * ~speed);
            var ageVal   = ~getValue.(~trackedAge[slot], ~times, currentTime, ~compression, 2.0 * ~speed);
            var droneFreq, droneAmp;

            // depth_norm: 0=deep, 1=shallow -> freq: 800-2500 Hz
            droneFreq = depthVal.linexp(0.0, 1.0, 800, 2500);
            // age_norm: 0=empty/young, 1=max age -> amp: very quiet
            droneAmp = ageVal * 0.04 * ~masterVol;

            droneSynths[slot].set(
                \freq, droneFreq,
                \amp, droneAmp,
                \regimeCol, regimeIdVal
            );
        };

        // ===== UPDATE REVERB =====
        // Regime change score drives reverb: high = spacious (structural transition)
        fxSynth.set(
            \mix,       regimeVal.linlin(0, 1, 0.15, 0.55),
            \room,      regimeVal.linlin(0, 1, 0.5, 0.92),
            \damp,      regimeVal.linlin(0, 1, 0.35, 0.08),
            \masterVol, ~masterVol
        );

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
                ["NIGHT",  "Active layer formation — organisms organizing"]
            }
            { section == \dawn } {
                ["DAWN",   "Layers dissolving — surface structure collapsing"]
            }
            { section == \day } {
                ["DAY",    "Deep stability — fewer events, persistent layers"]
            }
            { section == \dusk } {
                ["DUSK",   "Rapid restructuring — layers reforming upward"]
            }
            { true } {
                ["NIGHT",  "Return to surface layers — the cycle completes"]
            };

        if(section != lastSection) {
            "".postln;
            (">>> " ++ sectionName).postln;
            ("    " ++ bioState).postln;
            lastSection = section;
        };

        // ===== GUI UPDATE =====
        ~currentTime = currentTime;
        ~regimeVal = regimeVal;
        ~eventCount = eventCount;

        // Count active layers (age > 0.01)
        ~activeLayers = 4.collect({ |slot|
            var ageVal = ~getValue.(~trackedAge[slot], ~times, currentTime, ~compression, 2.0 * ~speed);
            if(ageVal > 0.01) { 1 } { 0 };
        }).sum;

        {
            var rHour = (hourVal * 24).floor.asInteger;
            var rMin  = ((hourVal * 24 * 60) % 60).floor.asInteger;
            var lastEvStr = if(~lastEventType.notNil) {
                ~lastEventType ++ " at " ++ ~lastEventDepth.round(1) ++ "m"
            } { "---" };

            if(~infoDisplay.notNil and: { ~infoDisplay.isClosed.not }) {
                ~infoDisplay.string_(
                    "UTC: " ++ rHour.asString.padLeft(2, "0") ++ ":" ++
                    rMin.asString.padLeft(2, "0") ++
                    "   Last: " ++ lastEvStr ++
                    "\nRegime: " ++ regimeVal.round(0.01) ++
                    "   Active layers: " ++ ~activeLayers ++
                    "   Events: " ++ eventCount
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
        ("Layer events complete. " ++ eventCount ++ " events triggered.").postln;
        droneSynths.do { |synth| synth.set(\gate, 0) };
    };
    ~playing = false;

    3.wait;  // let drone envelopes release and reverb tail decay
    fxSynth.free;

    { if(~sectionDisplay.notNil and: { ~sectionDisplay.isClosed.not }) {
        ~sectionDisplay.string_("COMPLETE\n" ++ eventCount ++ " structural events");
        ~sectionDisplay.stringColor_(Color.white);
    } }.defer;

    "Done.".postln;
});

// --- Transport controls ---
~playComposition = {
    ~playing = false;
    // Reset visualization
    ~vizN.do { |i| ~vizEvents[i] = nil };
    ~lastEventType = nil;
    ~eventCount = 0;
    ~composition.reset;
    ~composition.play;
    "Playing layer events...".postln;
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
  Dense, active event bells — the surface is alive with layer formation.
  Birth bells ring at high frequencies (organisms coalescing into feeding
  layers). Death thuds mark layers dispersing. The 4 drone slots track
  persistent layers near the surface (~200m). Reverb is moderate.

Dawn descent (1:55-2:50):
  Event density increases: surface layers die (low thuds) as organisms
  descend. Deep births appear (lower-pitched bells) as new layers form
  at depth. The regime change score peaks — reverb expands dramatically.
  The drones slide downward as tracked layers sink.

Day (2:50-6:30):
  Fewer events — the deep refuge is stable. Occasional births/deaths
  mark slow reorganization. The drones are quiet, steady, tracking
  persistent deep layers. Reverb is moderate-to-large (regime score
  settles). Long stretches of near-silence punctuated by isolated bells.

Dusk ascent (6:30-7:00):
  Rapid restructuring: deep layers die (thuds), shallow layers are born
  (bright bells). Event density surges. Regime score peaks again — the
  reverb swells. Drones shift upward as tracked layers ascend.

Night return (7:00-8:00):
  The surface layer structure re-establishes. Event rate returns to the
  active nighttime pattern. The cycle is complete.

SPECTRAL SEPARATION FROM SKETCH 01:
  This voice lives ABOVE and BELOW Sketch 01's FM range (90-560 Hz).
  Birth bells: 2000-8000 Hz — crystalline, metallic, above everything.
  Death thuds: 60-200 Hz — felt more than heard, below the FM voice.
  Layer drones: 800-2500 Hz — subtle backdrop in the spectral gap.

  When played simultaneously with Sketch 01, these are complementary
  instruments: the arc tells you WHERE, the events tell you WHEN.

GUI NOTES:
  Upward cyan triangles = birth events (new layer appeared)
  Downward orange triangles = death events (layer dissolved)
  The regime bar shows structural transition intensity (blue=stable, red=major change)
  Tracked layer traces show the 4 persistent slots as dim colored lines.

NEXT SKETCHES:
  03 — Zone energy (8-band BPF filter bank from depth histogram)
  04 — Onset pulses (high-percussion on acoustic transients)
  05 — Composed (all voices combined with intentional arrangement)
*/
