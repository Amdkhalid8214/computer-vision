# HANDVISION STUDIO — REALITYFRAME: CINEMATIC UNIVERSE PORTAL
### Real-Time Computer Vision, 3D Cosmic Portals, Doctor Strange Mystic Magic, Air Writing Studio & Air Keyboard

An advanced, high-performance standalone Python desktop application powered by **OpenCV**, **MediaPipe Hands**, and **NumPy**. It tracks hand landmarks in real time with single-pass inference to project dynamic filter boxes, transparent optical illusion portals (Ghost Mode), a mid-air virtual finger-typing keyboard, a freehand **Air Writing Studio**, procedural **Doctor Strange-inspired Mystic Magic Mandalas**, and a breathtaking **Cinematic 3D Universe Portal** drawn directly in the air with hand gestures.

---

## 1. Seven Unified Operating Modes

HandVision Studio operates in seven distinct modes with complete gesture isolation, zero cross-mode interference, and single-camera resource sharing:

| Mode | Key | Name | Description |
|:---:|:---:|:---|:---|
| **Mode 1** | `1` | **Standard Filter Box** | Classic two-hand dynamic ROI filter box with smooth corner tracking, EMA jitter reduction, and automatic boundary clamping. |
| **Mode 2** | `2` | **RealityFrame ROI Portal** | Cybernetic filter portal bounded by both index fingertips with neon cyan/green glowing borders, corner accents, and technical grid lines. |
| **Mode 3** | `3` | **RealityFrame Ghost Portal** | Optical illusion portal replacing live video with a static median background plate (`output[ROI] = background_plate[ROI]`), rendering a transparent window / invisibility cloak with neon gold borders. |
| **Mode 4** | `4` or `K` | **Virtual Finger Typing Air Keyboard** | Mid-air QWERTY virtual keyboard with live text input, cursor smoothing, rising-edge pinch typing, dwell timer typing, clipboard copy, and audio feedback. |
| **Mode 5** | `5` or `W` | **Air Writing Studio** | Freehand air drawing and writing system with pinch-to-draw, pen/highlighter/eraser, undo/redo, color palettes, neon glow ink, and PNG export. |
| **Mode 6** | `6` or `G` | **Mystic Magic Engine** | Real-time Doctor Strange-inspired circular mystic shields, rotating sacred geometry mandalas, sparks, energy beams, blasts, and two-hand scaling portals. |
| **Mode 7** | `7` or `U` | **Cinematic Universe Portal** | Air-drawn circular portal opening a magical window into a live 3D Solar System with stars, nebulae, planets, Saturn's rings, and hand-gesture space exploration. |

---

## 2. Core Features & Capabilities

### 🌌 1. RealityFrame: Cinematic Universe Portal (Mode 7)
- **Air-Drawn Circle Recognition (`src/circle_detector.py`)**:
  - The user draws a circle in the air with their index finger (pinching thumb + index to draw).
  - Uses **Kåsa algebraic circle fitting** (`(x - cx)^2 + (y - cy)^2 = r^2`) solved via least squares.
  - **PCA / Covariance analysis** to strictly reject straight lines (`eigvals[0] / eigvals[1] < 0.08`).
  - **Circularity consistency test**: Standard deviation of radial distances normalized by mean radius (`std_r / mean_r <= 0.32`).
  - **Closure gap verification**: Distance between stroke start and endpoint normalized by radius (`gap / r <= 0.85`).
  - **Angular coverage span check**: Unwrapped angular arc must cover at least 300° (`5.2 rad`).
  - Rejects incomplete arcs, accidental scribbles, and undersized/oversized gestures.
- **Cinematic Portal Activation Lifecycle**:
  - `IDLE`: Prompts user with glowing fingertip tracker.
  - `DRAWING`: Visualizes golden energy stroke trajectory with sparks.
  - `CHARGING`: Circle validated! Ignites concentric golden-orange rune rings and tangential spark particles.
  - `DISTORTING`: Space-time localized cosmic distortion ripples inside the ring.
  - `OPENING`: Cosmic iris expands outward revealing stars, nebulae, and the Solar System.
  - `ACTIVE`: Real-time interactive 3D universe exploration inside the portal.
  - `CLOSING`: Portal dissolves with swirling embers and soft fade.
- **Masked Compositing Guarantee**:
  - Uses an anti-aliased, feathered circular alpha mask (`(cx, cy, r)`).
  - The 3D universe is rendered **strictly inside** the circular portal boundary.
  - The live webcam video feed outside the circle remains **100% visible and untouched**.
- **Real-Time 3D Universe Engine (`src/universe_engine.py`)**:
  - **Genuine 3D Perspective Projection**: Transform matrix with 3D camera position, forward, right, and up basis vectors (`(x_cam / z_cam) * focal_length`).
  - **3D Parallax Starfield**: Hundreds of stars with spectral temperatures and realistic twinkling.
  - **Layered Procedural Nebulae**: Multi-frequency cosmic dust clouds in cyan, violet, and deep blue.
  - **Emissive Sun**: Glowing plasma core with corona flares and lighting origin.
  - **8 Realistically Textured Planets**:
    - *Mercury*: Cratered iron-gray surface.
    - *Venus*: Dense yellowish sulfuric acid atmosphere.
    - *Earth*: Blue oceans, green continents, and atmospheric cloud swirls.
    - *Mars*: Rust-red terrain and polar ice caps.
    - *Jupiter*: Great Red Spot and turbulent atmospheric bands.
    - *Saturn*: 3D tilted concentric ring system with Cassini division and realistic shadow casting.
    - *Uranus*: Pale cyan-blue ice giant.
    - *Neptune*: Vivid azure with dark storm spots.
  - **Asteroid Belt**: Thousands of orbiting rocky asteroids between Mars and Jupiter with individual Keplerian velocities.
  - **Lambertian 3D Sphere Shading**: Day/night terminators on every planet facing the central Sun with cosine lighting falloff.
  - **Dual Scale Modes (`[X]`)**:
    - `CINEMATIC`: Exaggerated planet sizes for beautiful visual clarity and cinematic enjoyment.
    - `SCIENTIFIC`: True relative planetary diameter ratios for educational accuracy.
- **Hand-Controlled Space Exploration**:
  - **Open Palm Drag**: Smoothly rotates the 3D camera yaw and pitch around the solar system.
  - **Pinch Drag**: Zooms the camera in toward planets or out to deep space overview.
  - **Pointing Gesture**: Points at a planet to focus and automatically fly the camera into close orbit.
  - **Educational Planet Card HUD**: Displays genuine astronomical specifications (category, diameter, orbital period, surface temperature, and notable scientific facts).
  - **Simulation Controls**: `[SPACE]` pauses orbital physics, `[X]` toggles scale mode, `[C]` closes the portal.

---

### 🪄 2. Mystic Magic Engine (Doctor Strange-Inspired)
- **Procedural Mandalas**: Multi-layer procedural circular energy shields featuring:
  - Concentric outer rings with fine rotating rune tick marks.
  - Counter-rotating inner 8-pointed star (octagram / interlaced squares).
  - Segmented dashed rings and radial energy spokes.
  - Pulsing central white-hot energy core.
  - Framerate-independent rotation speeds driven by delta-time (`dt`).
- **Fiery Energy Palette**: High-intensity fiery amber (`(0, 140, 255)`), glowing gold (`(0, 220, 255)`), and white-hot cores, with hotkey toggle to **Eldritch Cyan** and **Amethyst Purple**.
- **High-Performance 2D Particle System (`src/particle_system.py`)**:
  - Emits spinning circumferential sparks along shield rims that fly off tangentially with drag and updraft physics.
  - Additive blending (`cv2.add`) and multi-pass Gaussian blur for brilliant cinematic bloom.
- **5 Gesture-Triggered Spells**:
  1. **Mystic Shield**: Open palm facing camera spawns a rotating glowing mandala anchored to the palm center.
  2. **Energy Blast**: Pinch followed by quick release triggers an expanding radial shockwave ring and an explosive burst of 75 spark particles.
  3. **Energy Beam**: Pointing index finger shoots a focused laser beam with an intense core, golden outer glow, and traveling beam particles.
  4. **Mystic Portal (Two-Hand Magic)**: Both hands held up create a large connected portal between hands; spreading or closing hands dynamically scales the radius, and rotating relative hand positions rotates the mandala!
  5. **Energy Trail**: Rapid finger motion creates a luminous trail with fading spark embers.

---

### ✍️ 3. Air Writing Studio
- **Fingertip Drawing**: Tracks index fingertip (landmark 8) with low-latency Exponential Moving Average (EMA) coordinate smoothing.
- **Natural Drawing Gesture**: Pinching thumb tip (landmark 4) and index tip (landmark 8) starts drawing; releasing the pinch cleanly ends the stroke without stray connector lines.
- **Gapless Stroke Interpolation**: Automatically interpolates sub-pixel coordinates between consecutive points during rapid hand sweeps.
- **Tools & Brushes**:
  - `PEN`: Solid vibrant strokes.
  - `HIGHLIGHTER`: Semi-transparent highlighter marker strokes.
  - `ERASER`: Proximity-based eraser that removes intersecting stroke segments.
- **Color Palette**: White, Cyan, Green, Yellow, Orange, Red, and Purple (cycle with `[T]` or mouse click).
- **Adjustable Size**: Brush thickness from `2px` to `36px` (adjust with `[+]` and `[-]`).
- **Magical-Energy Ink Mode (`[G]`)**: Adds intense multi-layer Gaussian bloom and neon glowing aura to drawn lines.
- **Undo / Redo / Clear**: Full history stacks with `[Z]` for undo, `[Y]` for redo, and `[C]` for clear.
- **PNG Export (`[S]`)**: Saves high-resolution drawings to `recordings/air_writing_YYYYMMDD_HHMMSS.png` with support for transparent background (RGBA) export.
- **Dual Canvas Modes**: Draw over the live webcam feed (`OVERLAY`) or over a distraction-free dark slate canvas (`BLACKBOARD`).

---

### 🌟 4. RealityFrame Ghost Portal (Invisibility Illusion)
- **Median Plate Acquisition**: Computes a noise-free static background plate by taking the temporal median (`np.median`) across 25 sampled frames.
- **On-Screen Countdown HUD**: Displays a cyberpunk countdown timer (`2.5s`) prompting the user to step clear of the camera view during plate capture.
- **Live Recapture (`[R]` Hotkey)**: Recapture the background anytime without restarting the application.
- **Fault-Tolerant Fallback**: If recapture is aborted or fails, the application automatically restores the previous valid background plate without crashing.
- **Dual Cybernetic Styling**: High-visibility neon gold (`(0, 215, 255)`) glowing borders, technical grid lines, corner brackets, and dimension telemetry.

---

### 🎨 5. Expanded Visual Filter Engine (6+ Filters)
- **Color Inversion (Negative)**: Bitwise inversion (`255 - pixel`).
- **Cyber Canny Edge Detection**: Gaussian noise filtering, gradient edge extraction, and neon edge styling against a dark backdrop.
- **Thermal Vision False-Color Maps**: Luminance-to-temperature false-color maps with support for **Inferno**, **Jet**, and **Ocean** colormaps.
- **Pencil Sketch**: Color dodge blend dividing raw grayscale luminance by inverted blurred shading (`min(255, (gray * 256) / (255 - blur + 1))`).
- **Grayscale Vision**: Balanced 3-channel luminance representation.
- **Pixelated Vision (Digital Mosaic)**: Block downsampling and nearest-neighbor upscaling for a digital mosaic effect.
- **Custom Pipeline Filter**: Extensible composite filter class chaining multiple sequential filter stages.
- **Bi-Directional Cycling**: Rapidly cycle forward (`[N]`, `[M]`, or `[SPACE]`) or backward (`[P]`) through all registered filters.

---

### ⌨️ 6. Virtual Finger Typing Air Keyboard
- **Full QWERTY Layout**: 5-row keyboard including numbers `0-9`, symbols, letters `A-Z`, spacebar, backspace, enter, caps lock, shift, clear, copy, and mode toggle.
- **Live Text Input Bar**: Displays typed text above the keyboard with a blinking cursor and instantaneous updates.
- **Pinch-to-Type Mode**: Rising-edge gesture detection between thumb tip (landmark 4) and index fingertip (landmark 8) prevents accidental double-strokes while held.
- **Dwell-to-Type Mode**: Hovering the index fingertip over a key accumulates dwell progress with a visual radial arc indicator; completing the dwell triggers the key.
- **Windows Clipboard Sync**: One-touch `[COPY]` key copies accumulated text directly to the Windows system clipboard via `clip.exe`.
- **Audio Feedback**: Subtle auditory click on every registered keystroke.

---

### 📊 7. Interactive Cyberpunk Dashboard & Telemetry
- **Mouse-Clickable Top Navigation**: 11 interactive top dashboard buttons supporting mouse clicks (`cv2.setMouseCallback`) as well as keyboard shortcuts.
- **Diagnostic Telemetry Card (`[F]` / `[H]`)**: Displays real-time rolling FPS, frame latency (ms), webcam resolution, active mode, active filter, tracked hand count, tracking confidence, background plate status, inference time (ms), and render time (ms).

---

## 3. Keyboard Shortcuts & Controls

| Shortcut | Action |
|:---|:---|
| `1` | Switch to **Mode 1: Standard Filter Box** |
| `2` | Switch to **Mode 2: RealityFrame ROI Portal** |
| `3` | Switch to **Mode 3: RealityFrame Ghost Portal** |
| `4` or `K` | Switch to / Toggle **Mode 4: Virtual Keyboard** |
| `5` or `W` | Switch to / Toggle **Mode 5: Air Writing Studio** |
| `6` or `G` | Switch to / Toggle **Mode 6: Mystic Magic Engine** |
| `7` or `U` | Switch to / Toggle **Mode 7: Cinematic Universe Portal** |
| `X` | Toggle **Cinematic / Scientific Planet Scale** in Universe Portal |
| `SPACE` | **Pause / Resume Orbit Simulation** (Universe Portal) / Next Filter |
| `Z` | **Undo** last drawn stroke in Air Writing |
| `Y` | **Redo** undone stroke in Air Writing |
| `C` | **Clear Canvas** / **Clear Text** / **Close Active Universe Portal** |
| `S` | **Save Air Writing Drawing** to PNG |
| `+` / `-` | **Increase / Decrease Brush Size** in Air Writing |
| `T` | **Cycle Color** (Air Writing) / **Toggle Pinch/Dwell** (Keyboard) |
| `G` | **Toggle Magic Glow** (Air Writing) / **Cycle Magic Theme** (Mystic Magic) |
| `R` | **Recapture** RealityFrame Background Plate |
| `N` / `M` | Cycle to **Next Filter** |
| `P` | Cycle to **Previous Filter** |
| `F` or `H` | Toggle **HUD Telemetry Dashboard** |
| `Q` or `ESC` | **Exit Application** safely |
| **Mouse Left-Click** | Click any top button on the on-screen dashboard |

---

## 4. Running the Application

In **Windows PowerShell**, use `.\`:

```powershell
# Launch default (Standard Filter Mode)
.\run.bat

# Launch directly into Cinematic Universe Portal
.\run.bat --universe

# Launch directly into Doctor Strange Mystic Magic Engine
.\run.bat --magic

# Launch directly into Air Writing Studio
.\run.bat --air-writing

# Launch directly into RealityFrame Ghost Mode (Invisibility Portal)
.\run.bat --mode REALITYFRAME_GHOST

# Launch directly into Virtual Keyboard Mode in Dwell-to-type mode
.\run.bat --keyboard --typing-mode DWELL
```

Or run via Python directly in your active virtual environment:

```powershell
python main.py --universe
python main.py --magic
python main.py --air-writing
```

---

## 5. Automated Test Suite

Run the full pytest automated test suite across all 14 modules:

```powershell
.\test.bat
# or
python -m pytest
```

### Test Coverage Summary:
- **`tests/test_universe_portal.py`**: State machine transitions (`IDLE` -> `DRAWING` -> `CHARGING` -> `DISTORTING` -> `OPENING` -> `ACTIVE` -> `CLOSING`), masked circular alpha compositing isolation, closing dissolve, quality presets, scale mode toggles.
- **`tests/test_universe_engine.py`**: 3D perspective projection, orbital physics, planet specifications, Lambertian day/night terminator shading, camera rotation/zoom, planet focus.
- **`tests/test_circle_detector.py`**: Kåsa algebraic circle fitting, covariance straight-line rejection, circularity error tolerance, angular coverage span (>= 300°), closure gap check.
- **`tests/test_air_writing.py`**: Stroke creation, pinch detection, interpolation, undo/redo stacks, eraser splitting, PNG export, transparent RGBA export, blackboard vs overlay rendering.
- **`tests/test_mystic_magic.py`**: Doctor Strange mandala generation, open-palm shield trigger, pointing energy beam, two-hand portal scaling, pinch-release blast, particle rendering.
- **`tests/test_particle_system.py`**: Particle physics, emission limits, circumferential ring sparks, beam particles, drag, buoyancy, additive blending.
- **`tests/test_integration_modes.py`**: Mode switching across all 7 modes, mouse click callbacks on all 11 dashboard buttons, gesture isolation.
- **`tests/test_background_manager.py`**: Median plate computation, countdown ingestion, live recapture, fault-tolerant fallback.
- **`tests/test_camera.py`**: Device initialization, resolution negotiation, frame mirroring, clean resource teardown.
- **`tests/test_filters.py`**: All 6+ visual filters, immutability, shape preservation, bi-directional cycling.
- **`tests/test_frame_processing.py`**: Deterministic processing with mocked hand landmarks, edge clamping, status indicators.
- **`tests/test_keyboard.py`**: QWERTY layout, hover detection, rising-edge pinch typing, dwell timer typing, clipboard copy, hand loss recovery.
- **`tests/test_reality_frame.py`**: Portal geometry calculation, 2-frame stability check, coordinate clamping, ghost mode replacement.
- **`tests/test_roi_controller.py`**: 2D bounding box sorting, crossing hands, minimum size enforcement, EMA smoothing, discontinuity resets.

**Test Results: 100 passed in 2.06s (100% passing).**
