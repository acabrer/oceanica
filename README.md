# Sonificació Oceànica

**Transformar dades acústiques de l'oceà en música.**

Projecte creatiu de sonificació que utilitza dades hidroacústiques per crear composicions electroacústiques que representen la migració vertical diària de la vida marina.

**Equip:** Oceànica (Alex Cabrer & Joan Sala)
**Subvenció:** CLT019 - Generalitat de Catalunya, Departament de Cultura

---

## Instal·lació

### Prerequisits
- Python 3.11+
- SuperCollider 3.14+

### Configuració
```bash
git clone <repo-url>
cd oceanica_dev

python3.11 -m venv venv_echosound_py311
source venv_echosound_py311/bin/activate
pip install -r requirements.txt
```

### Dades en brut (opcional, per reprocessar)
Descarregar les dades d'ecosonda MALASPINA des de PANGAEA:
- **LEG2**: https://doi.pangaea.de/10.1594/PANGAEA.938719
- Citació: Irigoien et al. (2021) Sci Data 8:259

Extreure a `data/raw/MALASPINA_LEG2_1/`. Les dades de sonificació preprocessades s'inclouen a `output/data/` — només cal descarregar les dades en brut si es vol reprocessar des de zero.

---

## Inici ràpid

### Opció A: Reproduir la composició (no calen dades en brut)
1. Obrir `supercollider/ocean_sonification.scd` a SuperCollider
2. Executar les seccions 1-8 en ordre (Cmd+Enter)
3. Prémer PLAY a la interfície

### Opció B: Reprocessar des de dades en brut
```bash
source venv_echosound_py311/bin/activate

# Pas 1: Netejar tots els fitxers d'un dia → NetCDF net
python src/processing/run_24h_processing.py 20110126

# Pas 2: Generar ecograma a partir de les dades netes
python src/visualization/create_echogram_24h_validation.py 20110126

# Pas 3: Extreure característiques de sonificació → JSON per SuperCollider
python src/extraction/sonification_extractor.py 20110126

# Pas 4: Generar àudio directe (ecograma → WAV, 4 minuts)
python src/extraction/echogram_to_audio.py 20110126

# Pas 5: Crear vídeo amb playhead (ecograma PNG + WAV → MP4)
python src/visualization/create_echogram_video.py 20110126
```

---

## Estructura del projecte

```
oceanica_dev/
├── src/
│   ├── config.py                   # Centralized configuration (paths, params, presets)
│   ├── processing/
│   │   ├── echopype_main.py        # MalaspinaProcessor (calibration, noise removal)
│   │   └── run_24h_processing.py   # 24h cleaning orchestrator → cleaned_Sv_24h.nc
│   ├── visualization/
│   │   ├── colormaps.py                        # EK500-standard colormaps
│   │   ├── create_echogram_24h_validation.py   # 24h echogram from cleaned .nc
│   │   └── create_echogram_video.py            # Montage: PNG + WAV → MP4
│   └── extraction/
│       ├── sonification_extractor.py  # Feature extraction → JSON
│       └── echogram_to_audio.py       # Direct echogram → WAV mapping
├── supercollider/
│   └── ocean_sonification.scd      # Sound synthesis composition
├── output/
│   ├── data/                       # Cleaned NetCDF + JSON + WAV
│   └── visualizations/             # Echogram images + video
├── data/raw/                       # Raw MALASPINA data (not in git)
├── requirements.txt
└── .gitignore
```

### Pipeline de dades

```
raw .raw files (297 per day)
        │
        ▼
run_24h_processing.py
  (uses MalaspinaProcessor from echopype_main.py for each file:
   load → calibrate Sv → remove noise → mask impulse noise → extract 38 kHz)
        │
        ▼
cleaned_Sv_24h_{date}.nc  (single NetCDF with Sv + Sv_corrected)
        │
        ├──► create_echogram_24h_validation.py → echogram PNG ─┐
        │                                                      │
        ├──► sonification_extractor.py → JSON → SuperCollider  │
        │                                                      │
        ├──► echogram_to_audio.py → WAV ───────────────────────┤
        │                                                      │
        └──► create_echogram_video.py ◄── PNG + WAV → MP4 ────┘
```

---

## La ciència

### Migració Vertical Diària (DVM)

La migració animal més gran de la Terra, que es repeteix cada dia:

| Hora (UTC) | Esdeveniment | Profunditat |
|------------|-------------|-------------|
| 00:00-05:00 | Nit: alimentació en superfície | 0-200m |
| 05:00-07:00 | Alba: descens massiu | 200→500m |
| 07:00-17:00 | Dia: refugi profund | 400-600m |
| 17:00-20:00 | Capvespre: ascens massiu | 500→200m |
| 20:00-24:00 | Nit: retorn a la superfície | 0-200m |
holi
holi2

### Font de dades
- **Expedició de Circumnavegació MALASPINA 2010**
- Instrument: ecosonda split-beam Simrad EK60
- Freqüència: 38 kHz (peixos, calamars, organismes grans)
- Localització: Oceà Atlàntic, gener de 2011

### Per què només 38 kHz?
La freqüència de 120 kHz s'atenua ràpidament a l'aigua de mar i no pot penetrar fins a la capa de dispersió profunda (500-800m). Només els 38 kHz proporcionen senyal biològic útil a aquestes profunditats.

### Referències de processament
- **Eliminació de soroll**: De Robertis & Higginbottom (2007) ICES J. Mar. Sci. 64:1282-1291
- **Soroll impulsiu**: Ryan et al. (2015) — emmascarament de pings per llindar
- **Calibratge**: Demer et al. (2015) ICES CRR 326
- **Anàlisi DVM**: Klevjer et al. (2016) Sci. Rep. 6:19873

---

## Sonificació

### Mapeig de paràmetres

| Característica biològica | Paràmetre musical |
|---|---|
| Profunditat del centre de massa | To (poc profund = agut, profund = greu) |
| Intensitat total (dB) | Amplitud (indicador de biomassa) |
| Velocitat de migració | Modulació de filtre |
| Dispersió vertical | Reverberació / amplada espacial |
| Nombre de capes | Densitat rítmica / polifonia |

### Durada
24 hores de dades oceàniques comprimides en 4 minuts (360x).

### Arc narratiu de 24 hores

| Secció | Caràcter |
|--------|----------|
| **Nit** (00:00-05:00) | Eixams granulars, rítmic, viu |
| **Alba** (05:00-08:00) | Tensió FM, escombrat descendent |
| **Dia** (08:00-17:00) | Congelació espectral, dron profund |
| **Capvespre** (17:00-21:00) | Escombrat ascendent, anticipació |
| **Nit** (21:00-24:00) | Retorn del ritme i la vida |

### Format de dades (SuperCollider JSON)

El fitxer `sonification_sc_v3_{date}.json` conté arrays prenormalitzats (0.0-1.0) per cada punt temporal (~3 seg de resolució):

| Camp | Descripció |
|------|-----------|
| `time_seconds` | Segons des de l'inici del dia (0-86400) |
| `depth_norm` | Centre de massa: 0 = profund (1000m), 1 = superficial (0m) |
| `intensity_norm` | Retroescampament acústic: 0 = silenciós, 1 = fort |
| `velocity_norm` | Velocitat de migració: 0.5 = estacionari |
| `spread_norm` | Dispersió vertical: 0 = estret, 1 = ample |
| `layers_norm` | Nombre d'agregacions: 0 = cap, 1 = moltes |

### Exemple SuperCollider

```supercollider
// Load data
~data = JSONFileReader.read("sonification_sc_v3_20110126.json".resolveRelative);
~time = ~data["38kHz"]["time_seconds"];
~pitch = ~data["38kHz"]["depth_norm"];
~amp = ~data["38kHz"]["intensity_norm"];

// Simple SynthDef
SynthDef(\oceanVoice, { |freq=440, amp=0.5, spread=0.5|
    var sig = SinOsc.ar(freq, 0, amp);
    var reverb = FreeVerb.ar(sig, spread, 0.8, 0.5);
    Out.ar(0, reverb ! 2);
}).add;

// Play through data (compressed time)
~synth = Synth(\oceanVoice);
~compression = 1000;
r = Routine({
    ~time.size.do { |i|
        ~synth.set(
            \freq, ~pitch[i].linexp(0, 1, 100, 800),
            \amp, ~amp[i] * 0.5
        );
        (~time[i+1] - ~time[i] / ~compression).wait;
    };
}).play;
```

---

## Referències científiques

1. **Klevjer, T.A. et al. (2016)** — Large scale patterns in vertical distribution and behaviour of mesopelagic scattering layers. *Scientific Reports*, 6, 19873.
2. **Lee, W-J. et al. (2024)** — Interoperable and scalable echosounder data processing with Echopype. *ICES Journal of Marine Science*, 81(10), 1941-1952.
3. **De Robertis, A. & Higginbottom, I. (2007)** — A post-processing technique to estimate the signal-to-noise ratio and remove echosounder background noise. *ICES Journal of Marine Science*, 64, 1282-1291.
4. **Irigoien, X. et al. (2021)** — MALASPINA 2010 expedition echosounder data. *PANGAEA*, doi:10.1594/PANGAEA.938719.

---

## Llicència

Aquest projecte forma part d'una subvenció cultural de la Generalitat de Catalunya, Departament de Cultura.

---

*Sonificació Oceànica — Fer audible la migració invisible.*
