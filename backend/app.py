import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT_DIR = Path(__file__).resolve().parents[1]
DEEP3D_DIR = ROOT_DIR / "Deep3D"
INFERENCE_SCRIPT = DEEP3D_DIR / "inference.py"
UPLOAD_DIR = ROOT_DIR / "backend" / "uploads"
OUTPUT_DIR = ROOT_DIR / "backend" / "outputs"

# Switch models by changing MODEL_PATH env var. Example:
# MODEL_PATH=Deep3D/export/deep3d_v1.0_640x360_cpu.pt
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEEP3D_DIR / "export" / "deep3d_v1.0_640x360_cpu.pt")))
# Default to repo-level ff.pt for fine-tuned weights; allow override via env.
_finetuned_ckpt_env = os.getenv("FINETUNED_CKPT_PATH", str(ROOT_DIR / "ff.pt")).strip()
FINETUNED_CKPT_PATH = Path(_finetuned_ckpt_env) if _finetuned_ckpt_env else None
INFERENCE_TIMEOUT_SECONDS = int(os.getenv("INFERENCE_TIMEOUT_SECONDS", "7200"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _image_file_to_mp4(src: Path, dest: Path, suffix: str) -> None:
    """Build a short H.264 MP4 from a still image or short GIF so inference can run."""
    vf_even = "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    if suffix == ".gif":
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-t",
            "10",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            vf_even,
            str(dest),
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-loop",
            "1",
            "-i",
            str(src),
            "-c:v",
            "libx264",
            "-t",
            "5",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            vf_even,
            str(dest),
        ]
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )


def _require_ffmpeg_bins() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "ffmpeg / ffprobe not found on PATH. Install FFmpeg (includes both). "
                "macOS: brew install ffmpeg"
            ),
        )


app = FastAPI(title="Deep3D API", version="0.1.0")

# Allow any localhost / 127.0.0.1 origin with any port (Vite, Live Server, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/api/downloads", StaticFiles(directory=str(OUTPUT_DIR)), name="downloads")


@app.get("/api/health")
def health_check() -> dict:
    return {"status": "ok"}


@app.get("/api/config")
def get_config() -> dict:
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "finetuned_ckpt_path": str(FINETUNED_CKPT_PATH) if FINETUNED_CKPT_PATH else None,
        "finetuned_ckpt_exists": FINETUNED_CKPT_PATH.exists() if FINETUNED_CKPT_PATH else False,
        "inference_timeout_seconds": INFERENCE_TIMEOUT_SECONDS,
        "ffmpeg_on_path": shutil.which("ffmpeg") is not None,
        "ffprobe_on_path": shutil.which("ffprobe") is not None,
    }


@app.post("/api/convert")
async def convert_video(
    file: UploadFile = File(...),
    inv: bool = Form(False),
) -> dict:
    if not INFERENCE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail="inference.py not found")

    if not MODEL_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Model file not found at {MODEL_PATH}. "
                "From the repo root run: python3 scripts/download_deep3d_model.py "
                "Then restart the API. Or set MODEL_PATH to an existing .pt file."
            ),
        )
    if FINETUNED_CKPT_PATH and not FINETUNED_CKPT_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Fine-tuned checkpoint not found at {FINETUNED_CKPT_PATH}. "
                "Set FINETUNED_CKPT_PATH to your trained checkpoint (e.g. checkpoints/best.pt)."
            ),
        )

    _require_ffmpeg_bins()

    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    video_suffixes = {".mp4", ".mov", ".avi", ".mkv"}
    image_suffixes = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}
    allowed = video_suffixes | image_suffixes
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use video (mp4, mov, …) or image (jpg, png, webp, …).",
        )

    job_id = uuid.uuid4().hex[:12]
    input_path = UPLOAD_DIR / f"{job_id}{suffix}"
    output_path = OUTPUT_DIR / f"{job_id}_3d.mp4"
    tmp_dir = ROOT_DIR / "backend" / "tmp" / job_id
    tmp_dir.mkdir(parents=True, exist_ok=True)

    with input_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if suffix in image_suffixes:
        video_for_inference = UPLOAD_DIR / f"{job_id}_fromimg.mp4"
        try:
            _image_file_to_mp4(input_path, video_for_inference, suffix)
        except subprocess.CalledProcessError as exc:
            err = (exc.stderr or exc.stdout or "ffmpeg failed")[-2000:]
            raise HTTPException(status_code=500, detail=f"Could not convert image to video: {err}") from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail="ffmpeg is required to process images. Install ffmpeg and ensure it is on PATH.",
            ) from exc
    else:
        video_for_inference = input_path

    cmd = [
        sys.executable,
        str(INFERENCE_SCRIPT),
        "--model",
        str(MODEL_PATH),
        "--video",
        str(video_for_inference),
        "--out",
        str(output_path),
        "--tmpdir",
        str(tmp_dir),
    ]
    if FINETUNED_CKPT_PATH:
        cmd.extend(["--finetuned-ckpt", str(FINETUNED_CKPT_PATH)])
    if inv:
        cmd.append("--inv")

    try:
        subprocess.run(
            cmd,
            cwd=str(DEEP3D_DIR),
            check=True,
            capture_output=True,
            text=True,
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Inference timed out") from exc
    except subprocess.CalledProcessError as exc:
        error_log = (exc.stderr or exc.stdout or "Inference failed")[-3000:]
        raise HTTPException(status_code=500, detail=error_log) from exc

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Output file was not generated")

    return {
        "job_id": job_id,
        "output_file": output_path.name,
        "download_url": f"/api/downloads/{output_path.name}",
    }


@app.get("/api/download/{file_name}")
def download_file(file_name: str):
    file_path = OUTPUT_DIR / file_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=file_name, media_type="video/mp4")
