import os
import time
import argparse
from pathlib import Path
from tqdm import tqdm

import numpy as np
import cv2
import torch

from data import transform,impro
from utils import util,ffmpeg

parser = argparse.ArgumentParser()
parser.add_argument("--gpu_id", default=0, type=int,help="choose your device")
parser.add_argument("--model", default='./export/deep3d_v1.0_640x360_cpu.pt', type=str,help="input model path")
parser.add_argument("--video", default='./samples/waterbottle.mp4', type=str,help="input video path")
parser.add_argument("--out", default='./results/waterbottle_3d.mp4', type=str,help="output video path")
parser.add_argument('--inv', action='store_true', help='some video need to reverse left and right views')
parser.add_argument("--tmpdir", default='./tmp', type=str,help="output video path")
parser.add_argument('--gpu', action='store_true', help='force GPU usage even for CPU-named model files')
parser.add_argument('--disp-scale', default=1.0, type=float,
    help='disparity amplifier: right = left + scale*(model_out - left). '
         'Use ~5.0 to compensate for narrow-IPD (e.g. 12mm tree shrew vs 60mm human) '
         'when the frozen model was pretrained on human-scale stereo.')
parser.add_argument(
    "--finetuned-ckpt",
    default="",
    type=str,
    help="Optional TorchScript .pt; when set, load weights from this file instead of --model "
         "(finetune-rebased backend passes base --model plus this path).",
)
opt = parser.parse_args()

weights_path = (opt.finetuned_ckpt or "").strip() or opt.model
cuda_hint = weights_path


def apply_tree_shrew_color_profile(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Apply a perceptual color profile closer to tree-shrew-like RG discrimination:
    - suppress rainbow/chromatic edge fringing,
    - compress red/green separation toward yellow-brown tones,
    - preserve blue/yellow structure,
    - slightly boost blue-violet local contrast.
    """
    x = frame_bgr.astype(np.float32)
    b = x[..., 0]
    g = x[..., 1]
    r = x[..., 2]

    # 1) Reduce chromatic edge fringing (rainbow/neon outlines) by pulling
    # high-chroma, high-edge pixels toward luminance.
    luma = 0.114 * b + 0.587 * g + 0.299 * r
    edge = np.abs(cv2.Laplacian(luma, cv2.CV_32F, ksize=3))
    chroma = np.sqrt((r - g) ** 2 + (b - 0.5 * (r + g)) ** 2)
    edge_w = np.clip((edge - 8.0) / 36.0, 0.0, 1.0)
    chroma_w = np.clip((chroma - 16.0) / 64.0, 0.0, 1.0)
    fringe_w = 0.7 * edge_w * chroma_w
    b = b * (1.0 - fringe_w) + luma * fringe_w
    g = g * (1.0 - fringe_w) + luma * fringe_w
    r = r * (1.0 - fringe_w) + luma * fringe_w

    # 2) Desaturate red/green opponent channel specifically.
    rg_mean = 0.5 * (r + g)
    rg_delta = (r - g) * 0.12  # strong RG compression
    r = rg_mean + rg_delta
    g = rg_mean - rg_delta

    # 3) Collapse foliage/bark separation toward yellow-brown by anchoring
    # both R and G toward a shared warm tone (without flattening whole image).
    warm_anchor = np.clip(0.88 * rg_mean + 14.0, 0.0, 255.0)
    r = 0.65 * r + 0.35 * warm_anchor
    g = 0.70 * g + 0.30 * (0.88 * warm_anchor)

    # 4) Keep blue/yellow channels comparatively informative and slightly
    # enhance blue-violet contrast with a subtle unsharp mask on blue.
    b_blur = cv2.GaussianBlur(b, (0, 0), 1.1)
    b = np.clip(b + 0.22 * (b - b_blur), 0.0, 255.0)

    out = np.stack([b, g, r], axis=-1)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


# Determine device before loading model so map_location moves all tensors (including TorchScript constants)
if opt.gpu or ('cuda' in cuda_hint and torch.cuda.is_available()):
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{opt.gpu_id}')
        print(f"Using CUDA GPU: {opt.gpu_id}")
    else:
        device = torch.device("cpu")
        opt.gpu_id = -1
        print("Using CPU (GPU requested but CUDA not available)")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    opt.gpu_id = 0
    print("Using Apple MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    opt.gpu_id = -1
    print("Using CPU")

try:
    net = torch.jit.load(weights_path, map_location=device)
except TypeError as exc:
    # Some TorchScript checkpoints include float64 tensors that MPS cannot load.
    # Fall back to CPU so inference can proceed on Apple Silicon machines.
    if device.type == "mps" and "Cannot convert a MPS Tensor to float64 dtype" in str(exc):
        print("MPS load failed due to float64 tensors; falling back to CPU.")
        device = torch.device("cpu")
        opt.gpu_id = -1
        net = torch.jit.load(weights_path, map_location=device)
    else:
        raise
if opt.gpu_id >= 0:
    net.half()
net.eval()
process = transform.PreProcess()
if opt.gpu_id >= 0:
    process.to(device).half()

fps,duration,height,width = ffmpeg.get_video_infos(opt.video)
video_length = int(fps*duration)

def infer_model_resolution(model_path, fallback_w, fallback_h):
    model_name = os.path.basename(model_path)
    # Expected pattern in original checkpoints: *_<width>x<height>_*.pt
    try:
        size_token = model_name.split('_')[2]
        w_str, h_str = size_token.split('x')
        return int(w_str), int(h_str)
    except Exception:
        print(
            f"Could not parse resolution from model name '{model_name}'. "
            f"Falling back to input video size {fallback_w}x{fallback_h}."
        )
        return int(fallback_w), int(fallback_h)

out_width, out_height = infer_model_resolution(weights_path, width, height)

util.clean_tempfiles(opt.tmpdir)
util.makedirs(os.path.split(opt.out)[0])
ffmpeg.video2voice(opt.video,os.path.join(opt.tmpdir, 'tmp.wav'))

# Optional tip overlay (repo may not ship medias/tips_30.mp4)
tips = []
tips_path = Path(__file__).resolve().parent / "medias" / "tips_30.mp4"
cap_tips = cv2.VideoCapture(str(tips_path))
if cap_tips.isOpened():
    while True:
        ret, tip = cap_tips.read()
        if ret:
            tips.append(
                torch.from_numpy(
                    cv2.resize(tip, (out_width, int(out_width * 200 / 3840)), interpolation=cv2.INTER_LANCZOS4)
                )
            )
        else:
            break
cap_tips.release()

tip_h = tip_w = 0
tip_background = None
if tips:
    tip_h = tips[0].shape[0]
    tip_w = tips[0].shape[1]
    tip_background = torch.ones((3, tip_h, tip_w))
    if opt.gpu_id >= 0:
        tip_background = tip_background.to(device).half()

alpha = 5
cap = cv2.VideoCapture(opt.video)
frames_pool = []
output = np.zeros((out_height*1,out_width*2,3),np.uint8)
for i in range(alpha*2+1):
    ret, cur_frame = cap.read()
    if height != out_height or width != out_width:
        cur_frame = cv2.resize(cur_frame,(out_width,out_height),interpolation=cv2.INTER_LANCZOS4)
    frames_pool.append(torch.from_numpy(cur_frame))


x0 = frames_pool[0]
if opt.gpu_id >= 0:
    x0 = x0.to(device).half()
x0 = process(x0)

print("start inference...")
for frame in tqdm(range(video_length)):
    if frame<alpha:
        beta = 0
    elif alpha<=frame<video_length-alpha:
        beta = -(frame-alpha)

    if alpha<frame<video_length-alpha:
        ret, cur_frame = cap.read()
        if height != out_height or width != out_width:
            cur_frame = cv2.resize(cur_frame,(out_width,out_height),interpolation=cv2.INTER_LANCZOS4)
        if not ret or cur_frame is None:
            break
        frames_pool.pop(0)
        frames_pool.append(torch.from_numpy(cur_frame))

    x1 = frames_pool[np.clip(frame-alpha+beta,0,alpha*2)]
    x2 = frames_pool[np.clip(frame-1+beta,0,alpha*2)]
    x3 = frames_pool[frame+beta]
    x4  = frames_pool[np.clip(frame+1+beta,0,alpha*2)]
    x5  = frames_pool[np.clip(frame+alpha+beta,0,alpha*2)]

    if opt.gpu_id >= 0:
        x1,x2,x3,x4,x5 = x1.to(device).half(),x2.to(device).half(),x3.to(device).half(),x4.to(device).half(),x5.to(device).half()
    x1,x2,x3,x4,x5 = process(x1),process(x2),process(x3),process(x4),process(x5)
   
    input_data = torch.cat((x1,x2,x0,x3,x4,x5),dim=0)
    input_data = input_data.reshape(1,*input_data.shape)
    
    with torch.no_grad():
        out = net(input_data)
        x0 = out.clone().detach()[0]
    
    left = x3
    right = out[0]

    if opt.disp_scale != 1.0:
        # Amplify the learned parallax without touching the frozen model weights.
        # The model outputs right ≈ left when IPD is narrow; scaling the delta
        # compensates for the human-to-shrew IPD ratio (~60mm / 12mm = 5×).
        right = x3 + opt.disp_scale * (right - x3)
        right = torch.clamp(right, 0.0, 1.0)

    if tips:
        tip = tips.pop(0)
        if opt.gpu_id >= 0:
            tip = tip.to(device).half()
        tip = process(tip)
        left[:,out_height-tip_h:out_height,:] = left[:,out_height-tip_h:out_height,:]*(1-tip) + tip_background*tip
        right[:,out_height-tip_h:out_height,:] = right[:,out_height-tip_h:out_height,:]*(1-tip) + tip_background*tip

    if opt.inv:
        pred = torch.cat((right,left),dim=2)
    else:
        pred = torch.cat((left,right),dim=2)
    pred = transform.tensor2im(pred)
    pred = apply_tree_shrew_color_profile(pred)
    impro.imwrite(os.path.join(opt.tmpdir, 'cvt','%06d'%(frame+1)+'.png'),pred,True)

print("start write to video...")
ffmpeg.image2video(fps,os.path.join(opt.tmpdir, 'cvt','%06d.png'),os.path.join(opt.tmpdir, 'tmp.wav'),opt.out)
cap.release()
util.clean_tempfiles(opt.tmpdir,tmp_init=False)