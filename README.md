# Sonificació Oceànica

**Transformar dades acústiques de l'oceà en experiències sonores.**

Oceànica és una recerca transdisciplinària que combina la recerca artística i científica per explorar noves relacions i comprensions amb l'entorn oceànic. El projecte parteix de la idea que la distància amb l'oceà és també una distància sensorial i afectiva, i explora la sonificació de dades d'ecosondes com a canal per escurçar-la.

La temàtica central és la transformació de dades hidroacústiques oceàniques en experiències sonores immersives. Aquestes dades, obtingudes a través d'ecosondes oceanogràfiques, permeten estudiar la presència, densitat i moviment d'organismes marins dins la columna d'aigua. La sonificació no s'utilitza com un recurs il·lustratiu, sinó com una pràctica de recerca-creació que situa art i ciència en un pla de relació no jeràrquic.

**Equip:** Oceànica (Alex Cabrer & Joan Sala)
**Subvenció:** CLT019 - Generalitat de Catalunya, Departament de Cultura

---

## Instal·lació

### Prerequisits
- Python 3.11+
- SuperCollider 3.14+
- ffmpeg (`brew install ffmpeg`) — per a la generació de vídeo

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

Extreure a `data/raw/MALASPINA_LEG2_1/`. Les dades preprocessades s'inclouen a `output/data/` — només cal descarregar les dades en brut si es vol reprocessar des de zero.

---

## Comandes inicialització

```bash
source venv_echosound_py311/bin/activate

# 1. Netejar tots els fitxers d'un dia → NetCDF net
python src/processing/run_24h_processing.py 20110126

# 2. Generar ecograma a partir de les dades netes
python src/visualization/create_echogram_24h_validation.py 20110126

# 3. Extreure característiques de sonificació → JSON per SuperCollider
python src/extraction/sonification_extractor.py 20110126

# 4. Generar àudio directe (ecograma → WAV, 4 minuts)
python src/extraction/echogram_to_audio.py 20110126 --method ifft --ifft-iter 32

# 5. Crear vídeo amb playhead (ecograma PNG + WAV → MP4)
python src/visualization/create_echogram_video.py 20110126

# 6. (Opcional) Generar espectrograma de validació
python src/visualization/audio_to_spectrogram.py 20110126 --method ifft --ifft-iter 32
```

---

## Estructura del projecte

```
oceanica_dev/
├── src/
│   ├── config.py                                  # Configuració centralitzada (rutes, paràmetres)
│   ├── processing/
│   │   ├── echopype_main.py                       # MalaspinaProcessor (calibratge, neteja de soroll)
│   │   └── run_24h_processing.py                  # Orquestrador 24h → cleaned_Sv_24h.nc
│   ├── visualization/
│   │   ├── colormaps.py                           # Colormaps estàndard EK500
│   │   ├── create_echogram_24h_validation.py      # Ecograma 24h des de .nc net
│   │   ├── create_echogram_video.py               # Muntatge: PNG + WAV → MP4
│   │   └── audio_to_spectrogram.py                # Espectrograma de validació
│   └── extraction/
│       ├── sonification_extractor.py              # Extracció de característiques → JSON v8
│       ├── echogram_to_audio.py                   # Mapeig directe ecograma → WAV
│       └── features/                              # Mòduls d'extracció de característiques
│           ├── config.py                          # Paràmetres d'extracció
│           ├── normalization.py                   # Normalització percentil
│           ├── ping_features.py                   # Per ping: CoM, entropia, pics, histograma
│           ├── derived_features.py                # Velocitat, acceleració, anomalia, onset
│           ├── dvm.py                             # Seguiment DVM (corredor CoM 50-600m)
│           ├── histogram.py                       # 8 bandes oceanogràfiques
│           ├── events.py                          # Esdeveniments de capes, autocorrelació
│           └── formatter_v8.py                    # Constructor JSON v8 per SuperCollider
├── supercollider/
│   ├── ocean_sonification_v9.scd                  # DEPRECAT
│   └── voices/
│       ├── 01_migration_arc.scd                   # Veu 1: arc melòdic DVM (FM)
│       └── 02_layer_events.scd                    # Veu 2: esdeveniments de capes
├── output/
│   ├── data/                                      # NetCDF nets + JSON + WAV
│   └── visualizations/                            # Ecogrames + vídeo
├── data/raw/                                      # Dades crues MALASPINA (no a git)
├── requirements.txt
└── .gitignore
```

---

## Pipeline de dades

```
fitxers .raw (297 per dia)
        │
        ▼
run_24h_processing.py
  (utilitza MalaspinaProcessor d'echopype_main.py per cada fitxer:
   càrrega → calibratge Sv → eliminació soroll → emmascarament impulsos → extracció 38 kHz)
        │
        ▼
cleaned_Sv_24h_{data}.nc  (NetCDF únic amb Sv + Sv_corrected)
        │
        ├──► create_echogram_24h_validation.py → ecograma PNG ────┐
        │                                                         │
        ├──► sonification_extractor.py → JSON → SuperCollider     │
        │                                                         │
        ├──► echogram_to_audio.py → WAV ──────────────────────────┤
        │                                                         │
        └──► create_echogram_video.py ◄── PNG + WAV → MP4 ───────┘
```

---

### Migració Vertical Diària (DVM)

La migració animal més gran de la Terra, que es repeteix cada dia. Milers de milions d'organismes mesopelàgics — principalment peixos de la família Myctophidae — ascendeixen cada vespre des de profunditats de 400-600 metres fins a la superfície per alimentar-se, i descendeixen a l'alba per refugiar-se de la depredació visual.

### Font de dades

- **Expedició de Circumnavegació MALASPINA 2010** (buc oceanogràfic Hespérides)
- Instrument: ecosonda split-beam Simrad EK60
- Freqüència: 38 kHz (peixos mesopelàgics amb bufeta natatòria)
- Localització: Oceà Atlàntic Sud, ~25°S, gener de 2011
- Dades públiques: [PANGAEA](https://doi.pangaea.de/10.1594/PANGAEA.921760)

### Per què 38 kHz?

A 38 kHz, el gas de les bufetes natatòries dels peixos mesopelàgics proporciona un contrast acústic molt elevat respecte l'aigua de mar. La freqüència de 120 kHz, tot i oferir major resolució espacial, s'atenua ràpidament i no pot penetrar de forma útil per sota dels 200-250 metres de profunditat, impossibilitant l'observació de la capa de dispersió profunda (DSL) a 500-800m.

---

## Sonificació

### Transposició sonora directa

L'ecograma és, per naturalesa, un espectrograma acústic: l'eix vertical representa profunditat, l'horitzontal el temps, i la intensitat de color codifica la força de rebot acústic. Aquesta estructura és isomorfa a un sonograma temps-freqüència-amplitud.

El mapeig estableix una correspondència 1:1 entre les dimensions de l'ecograma i del sonograma:

| Dimensió ecograma | Dimensió sonora | Mapeig |
|---|---|---|
| Profunditat (50-1000m) | Freqüència (8000-50 Hz) | Escala logarítmica (poc profund = agut) |
| Intensitat Sv (-90 a -50 dB) | Amplitud | Lineal amb conformació perceptiva |
| Temps (24h) | Temps (4 min) | Compressió 360× |

S'han explorat tres mètodes de síntesi:

1. **Síntesi additiva**: suma de 256 sinusoides amb envolupants d'amplitud variables. Introdueix intermodulació (contingut freqüencial no present a les dades originals).
2. **ISTFT amb fase aleatòria**: tracta l'ecograma com a espectrograma de magnitud i n'extreu el senyal via transformada de Fourier inversa. Elimina la intermodulació però manca de coherència temporal.
3. **ISTFT + Griffin-Lim**: afegeix 32 iteracions de refinament de fase (Griffin & Lim, 1984), millorant la continuïtat temporal de les estructures biològiques al sonograma.

```bash
# Síntesi additiva
python src/extraction/echogram_to_audio.py 20110126 --method additive

# ISTFT amb fase aleatòria
python src/extraction/echogram_to_audio.py 20110126 --method ifft

# ISTFT + Griffin-Lim (32 iteracions)
python src/extraction/echogram_to_audio.py 20110126 --method ifft --ifft-iter 32
```

### Extracció de característiques per composició

L'objectiu és extreure característiques concretes de les dades perquè serveixin de motor per manipular paràmetres de síntesi sonora a SuperCollider, produint veus controlades tant estèticament com narrativament.

#### Característiques per ping

| Característica | Descripció |
|---|---|
| Centre de massa energètic | Profunditat mitjana ponderada per intensitat lineal (m) |
| Intensitat total | Suma d'energia retornada, indicador de biomassa (dB) |
| Dispersió vertical | Desviació de profunditat al voltant del CoM (m) |
| Entropia acústica | Grau d'uniformitat de la distribució vertical (Shannon) |
| Detecció de capes | Concentracions diferenciades d'organismes (scipy find_peaks) |
| Correlació inter-ping | Similitud cosinus entre perfils Sv consecutius |

#### Característiques derivades temporals

| Característica | Descripció |
|---|---|
| Velocitat vertical | Canvi del CoM en m/h (finestra 3-5 min) |
| Acceleració | Derivada de la velocitat, m/h² |
| Anomalia de profunditat | Desviació entre profunditat observada i esperada |
| Força d'onset | Transicions sobtades entre pings consecutius |

#### Característiques especialitzades

- **DVM (corredor CoM 50-600m)**: seguiment del centre de massa restringit, excloent zona superficial i DSL permanent. Detecta automàticament profunditats nocturnes/diürnes i transicions alba/capvespre.
- **Histograma de 8 bandes**: energia en zones amb significat ecològic (10-50m superfície epipelàgica, 50-150m capa de mescla, 150-300m mesopelàgic, 300-500m mesopelàgic inferior, 500-700m mesopelàgic profund, 700-850m mesopelàgic basal, 850-1000m límit meso-batipelàgic, 280-580m corredor DVM).
- **Esdeveniments de capes**: naixement i dissolució de concentracions d'organismes, amb seguiment de fins a 4 capes simultànies.

### Veus compositives (SuperCollider)

#### Veu 1 — Migration Arc

Sonifica la DVM com una línia melòdica FM que evoluciona al llarg de les 24 hores. La profunditat del CoM controla el to (ascens = agut, descens = greu), l'entropia governa el timbre (concentrada = tonal, dispersa = complex), i la velocitat controla el vibrato i la brillantor. Una comporta (gate) basada en l'activitat de la columna d'aigua gestiona els silencis, que reflecteixen moments d'estabilitat.

#### Veu 2 — Layer Events

Sonifica esdeveniments estructurals de la columna d'aigua. El naixement d'una concentració genera un impuls metàl·lic agut (2-8 kHz); la dissolució genera un impuls greu (60-200 Hz). Quatre drons segueixen les capes persistents, amb to vinculat a la profunditat i intensitat dependent de l'edat de la capa.

### Format de dades (SuperCollider JSON v8)

El fitxer `sonification_sc_v8_{data}.json` conté arrays prenormalitzats (0.0-1.0) per cada punt temporal (~3 seg de resolució):

| Camp | Descripció |
|------|-----------|
| `time_seconds` | Segons des de l'inici del dia (0-86400) |
| `depth_norm` | Centre de massa: 0 = profund, 1 = superficial |
| `intensity_norm` | Retroescampament acústic: 0 = silenciós, 1 = fort |
| `velocity_norm` | Velocitat de migració: 0.5 = estacionari |
| `dvm_depth_norm` | Profunditat DVM (corredor 50-600m) |
| `dvm_velocity_norm` | Velocitat DVM |
| `entropy_norm` | Entropia acústica: 0 = tonal, 1 = soroll |
| `spread_norm` | Dispersió vertical: 0 = estret, 1 = ample |
| `layers_norm` | Nombre d'agregacions: 0 = cap, 1 = moltes |
| `onset_strength_norm` | Força de transicions sobtades |
| `histogram_8band_norm` | Energia per banda oceanogràfica (8 arrays) |

---

## Referències científiques

1. **Klevjer, T.A. et al. (2016)** — Large scale patterns in vertical distribution and behaviour of mesopelagic scattering layers. *Scientific Reports*, 6, 19873.
2. **Irigoien, X. et al. (2014)** — Large mesopelagic fishes biomass and trophic efficiency. *Nature Communications*, 5, 3271.
3. **De Robertis, A. & Higginbottom, I. (2007)** — A post-processing technique to estimate the signal-to-noise ratio and remove echosounder background noise. *ICES Journal of Marine Science*, 64, 1282-1291.
4. **Griffin, D.W. & Lim, J.S. (1984)** — Signal estimation from modified short-time Fourier transform. *IEEE Trans. Acoustics, Speech, and Signal Processing*, 32(2), 236-243.
5. **Lee, W-J. et al. (2024)** — Interoperable and scalable echosounder data processing with Echopype. *ICES Journal of Marine Science*, 81(10), 1941-1952.
6. **Shannon, C.E. (1948)** — A mathematical theory of communication. *Bell System Technical Journal*, 27(3), 379-423.

---

## Llicència



Aquesta obra ha rebut una beca de creació artística, recerca i innovació de la Generalitat de Catalunya, als elements informatius i de difusió i publicitat del projecte de recerca o creació objecte de la subvenció.

<p align="center">
  <img src="assets/idbh.png" alt="Generalitat de Catalunya" width="300"/>
</p>

---