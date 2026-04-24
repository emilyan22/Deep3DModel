import os
import shutil
import subprocess
import sys
import uuid
from functools import lru_cache
from pathlib import Path

import torch
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
# Match finetune-rebased default model path; override via MODEL_PATH env var.
MODEL_PATH = Path(os.getenv("MODEL_PATH", str(DEEP3D_DIR / "export" / "deep3d_v1.0_640x360_cpu.pt")))
CPU_FALLBACK_MODEL_PATH = DEEP3D_DIR / "export" / "deep3d_v1.0_640x360_cpu.pt"
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


@lru_cache(maxsize=8)
def _torchscript_compatibility(model_path_str: str) -> dict:
    model_path = Path(model_path_str)
    if not model_path.exists():
        return {"exists": False, "compatible": False, "reason": "model file not found"}
    try:
        scripted = torch.jit.load(str(model_path), map_location="cpu")
        code = str(scripted.code)
        if 'torch.device("cuda:0")' in code:
            return {
                "exists": True,
                "compatible": False,
                "reason": "CUDA-only TorchScript graph; this host will require CPU fallback",
            }
        return {"exists": True, "compatible": True, "reason": "model appears host-compatible"}
    except Exception as exc:
        return {"exists": True, "compatible": False, "reason": f"model load check failed: {type(exc).__name__}"}


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
    preferred_status = _torchscript_compatibility(str(MODEL_PATH))
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_host_compatible": preferred_status["compatible"],
        "model_host_compatibility_reason": preferred_status["reason"],
        "cpu_fallback_model_path": str(CPU_FALLBACK_MODEL_PATH),
        "cpu_fallback_model_exists": CPU_FALLBACK_MODEL_PATH.exists(),
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

    def _build_cmd(model_path: Path) -> list[str]:
        cmd = [
            sys.executable,
            str(INFERENCE_SCRIPT),
            "--model",
            str(model_path),
            "--video",
            str(video_for_inference),
            "--out",
            str(output_path),
            "--tmpdir",
            str(tmp_dir),
        ]
        if inv:
            cmd.append("--inv")
        return cmd

    def _looks_like_cuda_locked_model(error_text: str) -> bool:
        e = error_text.lower()
        return (
            "device=torch.device(\"cuda:0\")" in e
            or "arguments from the 'cuda' backend" in e
            or "could not run 'aten::empty_strided'" in e
        )

    try:
        used_model = MODEL_PATH
        subprocess.run(
            _build_cmd(used_model),
            cwd=str(DEEP3D_DIR),
            check=True,
            capture_output=True,
            text=True,
            timeout=INFERENCE_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(status_code=504, detail="Inference timed out") from exc
    except subprocess.CalledProcessError as exc:
        primary_error = exc.stderr or exc.stdout or "Inference failed"
        can_fallback = (
            MODEL_PATH != CPU_FALLBACK_MODEL_PATH
            and CPU_FALLBACK_MODEL_PATH.exists()
            and _looks_like_cuda_locked_model(primary_error)
        )
        if can_fallback:
            try:
                used_model = CPU_FALLBACK_MODEL_PATH
                subprocess.run(
                    _build_cmd(used_model),
                    cwd=str(DEEP3D_DIR),
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=INFERENCE_TIMEOUT_SECONDS,
                )
            except subprocess.CalledProcessError as fallback_exc:
                fallback_error = fallback_exc.stderr or fallback_exc.stdout or "Fallback inference failed"
                error_log = (
                    "Fine-tuned model appears CUDA-only on this machine and fallback model failed.\n\n"
                    f"Primary model: {MODEL_PATH}\n"
                    f"Fallback model: {CPU_FALLBACK_MODEL_PATH}\n\n"
                    f"Primary error:\n{primary_error[-1200:]}\n\n"
                    f"Fallback error:\n{fallback_error[-1200:]}"
                )
                raise HTTPException(status_code=500, detail=error_log) from fallback_exc
            except subprocess.TimeoutExpired as fallback_timeout:
                raise HTTPException(status_code=504, detail="Fallback inference timed out") from fallback_timeout
        else:
            error_log = (primary_error or "Inference failed")[-3000:]
            raise HTTPException(status_code=500, detail=error_log) from exc

    if not output_path.exists():
        raise HTTPException(status_code=500, detail="Output file was not generated")

    return {
        "job_id": job_id,
        "output_file": output_path.name,
        "download_url": f"/api/downloads/{output_path.name}",
        "model_used": str(used_model),
        "used_cpu_fallback": str(used_model) == str(CPU_FALLBACK_MODEL_PATH),
    }


@app.get("/api/download/{file_name}")
def download_file(file_name: str):
    file_path = OUTPUT_DIR / file_name
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=str(file_path), filename=file_name, media_type="video/mp4")
