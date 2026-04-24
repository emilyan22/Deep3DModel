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
parser.add_argument("--finetuned-ckpt", default='', type=str, help="fine-tuned checkpoint path (.pt) to load with model.load_state_dict")
parser.add_argument("--video", default='./samples/waterbottle.mp4', type=str,help="input video path")
parser.add_argument("--out", default='./results/waterbottle_3d.mp4', type=str,help="output video path")
parser.add_argument('--inv', action='store_true', help='some video need to reverse left and right views')
parser.add_argument("--tmpdir", default='./tmp', type=str,help="output video path")
opt = parser.parse_args()


net = torch.jit.load(opt.model)
loaded_finetuned_torchscript = False
if opt.finetuned_ckpt:
    # Support both formats:
    # 1) TorchScript archive (.pt) -> replace net directly
    # 2) state-dict checkpoint (.pt/.pth) -> load_state_dict into net
    try:
        scripted_model = torch.jit.load(opt.finetuned_ckpt, map_location="cpu")
        net = scripted_model
        loaded_finetuned_torchscript = True
        print(f"Loaded fine-tuned TorchScript model: {opt.finetuned_ckpt}")
    except Exception:
        try:
            checkpoint = torch.load(opt.finetuned_ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            # Older torch versions do not support weights_only kwarg.
            checkpoint = torch.load(opt.finetuned_ckpt, map_location="cpu")

        state_dict = checkpoint["model_state"] if isinstance(checkpoint, dict) and "model_state" in checkpoint else checkpoint
        load_result = net.load_state_dict(state_dict, strict=False)
        print(
            f"Loaded fine-tuned checkpoint: {opt.finetuned_ckpt} "
            f"(missing={len(load_result.missing_keys)}, unexpected={len(load_result.unexpected_keys)})"
        )
net.eval()
process = transform.PreProcess()

# Device detection - prioritize CPU for CPU models to avoid compatibility issues
if 'cpu' in opt.model:
    # Force CPU for CPU models to avoid MPS/CPU tensor mismatch
    device = torch.device("cpu")
    opt.gpu_id = -1
    print("Using CPU (forced for CPU model)")
elif 'cuda' in opt.model and torch.cuda.is_available():
    device = torch.device(f'cuda:{opt.gpu_id}')
    net.to(device).half()
    process.to(device).half()
    print(f"Using CUDA GPU: {opt.gpu_id}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    net.to(device).half()
    process.to(device).half()
    opt.gpu_id = 0
    print("Using Apple MPS (Metal Performance Shaders)")
else:
    device = torch.device("cpu")
    opt.gpu_id = -1
    print("Using CPU (fallback)")

out_width  = int(os.path.basename(opt.model).split('_')[2].split('x')[0])
out_height = int(os.path.basename(opt.model).split('_')[2].split('x')[1])

fps,duration,height,width = ffmpeg.get_video_infos(opt.video)
video_length = int(fps*duration)

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
fallback_to_base_done = False

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
        try:
            out = net(input_data)
        except RuntimeError as exc:
            err = str(exc)
            # Some finetuned TorchScript exports were traced with a hardcoded cuda:0 op.
            # On CPU-only machines this fails at runtime; fall back to the base model.
            if (
                loaded_finetuned_torchscript
                and not torch.cuda.is_available()
                and not fallback_to_base_done
                and "cuda:0" in err
            ):
                print(
                    "[WARN] Finetuned TorchScript uses hardcoded CUDA ops and cannot run on CPU. "
                    f"Falling back to base model: {opt.model}"
                )
                net = torch.jit.load(opt.model)
                net.eval()
                fallback_to_base_done = True
                out = net(input_data)
            else:
                raise
        x0 = out.clone().detach()[0]
    
    left = x3
    right = out[0]
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
    impro.imwrite(os.path.join(opt.tmpdir, 'cvt','%06d'%(frame+1)+'.png'),pred,True)

print("start write to video...")
ffmpeg.image2video(fps,os.path.join(opt.tmpdir, 'cvt','%06d.png'),os.path.join(opt.tmpdir, 'tmp.wav'),opt.out)
cap.release()
util.clean_tempfiles(opt.tmpdir,tmp_init=False)