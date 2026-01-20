# Vision AI Ganador

Servicio de liveness anti-spoofing con backend FastAPI y demo en Streamlit.
Evalua liveness usando 3 senales: PAD (anti-spoof), parpadeo (EAR) y movimiento de cabeza (yaw).

## Stack
- Python 3.x
- FastAPI + Uvicorn
- OpenCV + numpy
- UniFace (RetinaFace, MiniFASNet, Landmark106)
- ONNX Runtime
- Streamlit (frontend)

## Arquitectura (capas)
- `app/api`: endpoints HTTP y validacion de entrada.
- `app/application`: orquestacion del pipeline (`LivenessService`).
- `app/domain`: reglas de negocio (blink, head movement, decision).
- `app/infrastructure`: modelos y utilidades (uniface, video -> frames).
- `project-roo/app`: frontend Streamlit.
- `project-roo/scripts`: demos CLI con webcam.

## Flujo general del pipeline (backend)
1) **Input**:
   - `/liveness/check` recibe 1 imagen.
   - `/liveness/check-video` recibe un video (AVI).
   - `/liveness/check-frames` recibe N imagenes (frames).
2) **Preprocesamiento**:
   - Resize por `preproc_max_side_px`.
   - Video -> frames (muestreo uniforme hasta `max_frames`).
3) **PAD (anti-spoof)**:
   - `RetinaFace` detecta rostro.
   - `MiniFASNet` predice REAL/FAKE y `confidence`.
4) **Landmarks + EAR**:
   - `Landmark106` calcula 106 puntos faciales.
   - EAR promedio (ojo izq/der) por frame.
5) **Head pose**:
   - `solvePnP` + `rvec_to_euler_degrees` para yaw/pitch/roll.
6) **Decision**:
   - Se evalua PAD + blink + head movement.
   - Estrategia `balanced` o `strict`.

## Modelos usados
- **RetinaFace**: deteccion de rostro.
- **MiniFASNet**: anti-spoof (PAD). Usa el modelo default de la libreria.
- **Landmark106**: landmarks faciales (106 puntos).

## Endpoints (API)
Base URL: `http://127.0.0.1:8000`

### POST /liveness/check
Entrada: 1 imagen (`multipart/form-data`, campo `file`).

Uso: PAD en un solo frame (demo rapido).

### POST /liveness/check-video
Entrada: 1 video (`multipart/form-data`, campo `file`).

Uso: el backend extrae frames de forma uniforme y evalua liveness.

### POST /liveness/check-frames
Entrada: multiples imagenes (`multipart/form-data`, campo `files`).

Notas:
- Si `len(files) > max_frames` responde `413`.
- Este endpoint NO recorta frames; el cliente debe limitar.

## Respuesta (contract)
Ejemplo tipico:
```json
{
  "liveness": true,
  "confidence": 0.87,
  "checks": {
    "blink_detected": true,
    "head_movement": false,
    "blink_count": 2,
    "anti_spoof": {
      "label": "REAL",
      "confidence": 0.92,
      "passed": true
    }
  },
  "message": "frames_with_face=14 real_count=12 conf_med=0.87 blinks=2 yaw_range=18.4"
}
```

## Logica de negocio (detalle)
### PAD (anti-spoof)
- Se usa `RetinaFace` para detectar rostro por frame.
- Se usa `MiniFASNet` para REAL/FAKE y confidence.
- `anti_min_real_frames`: minimo de frames con `is_real=True`.
- `anti_min_confidence`: umbral de mediana de confidence.

### Parpadeo (EAR)
- Se calcula EAR promedio entre ojo izquierdo y derecho.
- Se toma baseline con `blink_baseline_frames`.
- Umbrales:
  - `close_th = baseline * blink_close_ratio`
  - `open_th = baseline * blink_open_ratio`
- Un parpadeo se cuenta cuando:
  - EAR baja de `close_th` por al menos `blink_min_closed_frames`,
  - luego sube por encima de `open_th`.

### Movimiento de cabeza (yaw)
- Se usa `rvec_to_euler_degrees` para obtener yaw por frame.
- Se toma baseline con `head_baseline_frames`.
- `head_movement = max(|yaw - baseline|) >= head_yaw_delta_deg`.

### Decision final
- `balanced`: PAD ok AND (blink_count >= min_blinks OR head_movement).
- `strict`: PAD ok AND (blink_count >= min_blinks AND head_movement).

## Configuracion (env vars)
Las variables usan prefijo `LIVENESS_`. Referencia: `app/.env.example`.

Principales:
- `LIVENESS_MAX_FRAMES` (default 90)
- `LIVENESS_PREPROC_MAX_SIDE_PX` (default 640)
- `LIVENESS_ANTI_MIN_REAL_FRAMES` (default 10)
- `LIVENESS_ANTI_MIN_CONFIDENCE` (default 0.70)
- `LIVENESS_BLINK_*` (baseline, ratios, min count)
- `LIVENESS_HEAD_*` (baseline, yaw delta)
- `LIVENESS_DECISION_MODE` (`balanced` o `strict`)

Nota: `check-frames` rechaza si `len(files) > max_frames`.

## Instalacion (backend)
```bash
python -m venv venv_vision_ai
venv_vision_ai\\Scripts\\activate
pip install -r requirements.txt
```

Run:
```bash
uvicorn app.main:app --reload
```

Logs: `liveness.log` (root del repo).

## Frontend (Streamlit)
El frontend esta en `project-roo/app/app_frontend.py`.

Instalacion:
```bash
pip install streamlit requests
```

Run:
```bash
streamlit run project-roo/app/app_frontend.py
```

Comportamiento:
- Intenta `/check-video` primero.
- Si falla, cae a `/check-frames`.
- Si el AVI queda vacio, no llama `/check-video`.
- En el fallback limita frames para evitar 413.

## Scripts de demo
Ubicacion: `project-roo/scripts`
- `webcam_demo.py`: envia batches a `/check-frames`.
- `webcam_demo_local.py`: demo local sin API (usa UniFace directo).
- `webcam_stream_client.py`: cliente para endpoint streaming (experimental).

## Tests
Ubicacion: `app/tests`
```bash
pytest
```

Incluyen:
- contract tests de endpoints
- tests de decision, blink y head movement

## Estructura del repo (resumen)
```
.
├─ app
│  ├─ api
│  ├─ application
│  ├─ domain
│  ├─ infrastructure
│  └─ tests
├─ project-roo
│  ├─ app
│  └─ scripts
├─ requirements.txt
└─ README.md
```

## Troubleshooting
### 413 "Demasiados frames"
El endpoint `/check-frames` no recorta. Si envias mas de `max_frames`, responde 413.
Solucion: muestrear frames en el cliente o reducir FPS/duracion.

### "Could not decode video or extracted 0 frames"
Puede pasar si el AVI esta vacio o OpenCV no abre el temporal.
Revisa los bytes del video y permisos de escritura temporal.

## Licencia
No especificada.
