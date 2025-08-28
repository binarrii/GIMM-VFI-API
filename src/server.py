import json
import multiprocessing
import os
import shutil
import uuid
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass
from typing import Union

import cv2
import dotenv
import numpy as np
import requests
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from tqdm import tqdm

from models import create_model
from utils.flow_viz import flow_to_image
from utils.setup import single_setup
from utils.utils import InputPadder, set_seed
from video_Nx import images_to_video, load_image

dotenv.load_dotenv()
_HTTP_PREFIX = os.getenv("GVFI_HTTP_PREFIX", default="").rstrip("/")
_PATH_PREFIX = os.getenv("GVFI_PATH_PREFIX", default="").rstrip("/")

_WORK_DIR = "work"
_INPUT_DIR = f"{_WORK_DIR}/input"
_OUTPUT_DIR = f"{_WORK_DIR}/output"

os.makedirs(_INPUT_DIR, exist_ok=True)
os.makedirs(_OUTPUT_DIR, exist_ok=True)


@dataclass
class VFIArgs:
    model_config: str = "configs/gimmvfi/gimmvfi_r_arb.yaml"
    load_path: str = "pretrained_ckpt/gimmvfi_r_arb_lpips.pt"
    source_path: str = None
    output_path: str = None
    N: int = 8
    ds_factor: float = 1.0
    result_path: str = None
    postfix: str = None
    seed: int = 0
    eval: bool = True


class VFIRequest(BaseModel):
    uid: str = None
    images: Union[list[str], str] = None
    video: str = None
    n: int = 8
    output_type: str = "path"
    notify_url: str = None


_model = None


def _load_model():
    global _model

    args = VFIArgs()
    config = single_setup(args, extra_args=(), train=False)
    _model, _ = create_model(config.arch)
    _model = _model.to(torch.device("cuda"))

    if not args.load_path == "":
        if "ours" in args.load_path:
            ckpt = torch.load(args.load_path, map_location="cpu")

            def convert(param):
                return {
                    k.replace("module.feature_bone", "frame_encoder"): v
                    for k, v in param.items()
                    if "feature_bone" in k
                }

            ckpt = convert(ckpt)
            _model.load_state_dict(ckpt, strict=False)
        else:
            ckpt = torch.load(args.load_path, map_location="cpu")
            _model.load_state_dict(ckpt["state_dict"], strict=True)
    else:
        ckpt = None
        if args.eval or args.resume:
            raise ValueError("load_path must be specified in evaluation or resume mode")


def _download_files(urls: Union[list[str], str], download_path: str) -> None:
    """
    Downloads files from given URLs and saves them to the specified directory.

    Args:
        urls (Union[list[str], str]): A list of file URLs or a single file URL to download.
        download_path (str): The directory path where the downloaded files will be stored.
    """
    os.makedirs(download_path, exist_ok=True)

    if isinstance(urls, str):
        urls = [urls]

    for idx, url in enumerate(urls):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            file_path = os.path.join(download_path, url.split("/")[-1])
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        except requests.exceptions.RequestException as e:
            print(f"Failed to download {url}: {e}")


def _copy_files(src: str, dest: str) -> None:
    """
    Copies files from a directory or a single specific file to the destination directory.

    Args:
        src (str): Directory path of files or a single file path to copy.
        dest (str): Directory path where files will be copied.
    """
    src = os.path.expanduser(src)
    dest = os.path.expanduser(dest)
    os.makedirs(dest, exist_ok=True)

    try:
        if os.path.isfile(src):
            shutil.copy(src, dest)
        elif os.path.isdir(src):
            for item in os.listdir(src):
                item_path = os.path.join(src, item)
                if os.path.isfile(item_path):
                    shutil.copy(item_path, dest)
        else:
            raise ValueError(f"Source {src} is not a valid file or directory.")
    except Exception as e:
        print(f"Failed to copy {src}: {e}")


def _extract_frames_from_video(video_path: str, output_dir: str) -> None:
    """
    Extracts frames from a video file and saves them as .png files in the specified directory.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory where the extracted frames will be saved.
    """
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"{frame_count:06d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"{frame_count} frames extracted, frame rate: {frame_rate}")


def _get_video_frame_rate(video_path: str) -> float:
    """
    Reads the frame rate of a video file.

    Args:
        video_path (str): Path to the video file.

    Returns:
        float: The frame rate of the video.
    """
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate


def _run(req: VFIRequest):
    uid = req.uid
    print(f"Executing task {uid} on process {os.getpid()}")

    src_path, dst_path = f"{_INPUT_DIR}/{uid}", f"{_OUTPUT_DIR}/{uid}"
    os.makedirs(src_path, exist_ok=True)

    if req.images:
        if isinstance(req.images, str) and req.images.startswith(("/", "~")):
            _copy_files(src=req.images, dest=src_path)
        else:
            _download_files(urls=req.images, download_path=src_path)
    elif req.video:
        if req.video.startswith(("/", "~")):
            _copy_files(src=req.video, dest=src_path)
        else:
            _download_files(urls=req.video, download_path=src_path)

        vfilename = req.video.split("/")[-1]
        _extract_frames_from_video(f"{src_path}/{vfilename}", src_path)
    else:
        raise ValueError("Images or video must be provided")

    args = VFIArgs(source_path=src_path, output_path=dst_path, N=req.n)
    set_seed(args.seed)
    device = torch.device("cuda")

    os.makedirs(args.output_path, exist_ok=True)

    source_path = args.source_path
    output_path = os.path.join(args.output_path, f"o-{uid}.mp4")
    flow_output_path = os.path.join(args.output_path, f"f-{uid}.mp4")

    img_list = sorted(
        filter(lambda f: f.endswith((".png", ".jpg")), os.listdir(source_path))
    )
    images = []
    ori_image = []
    flows = []
    start = 0
    end = len(img_list) - 1

    for j in tqdm(range(start, end)):
        img_path0 = os.path.join(source_path, img_list[j])
        img_path2 = os.path.join(source_path, img_list[j + 1])
        # prepare data b,c,h,w
        I0 = load_image(img_path0)
        if j == start:
            images.append(
                (I0.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                    :, :, ::-1
                ].astype(np.uint8)
            )
            ori_image.append(
                (I0.squeeze().detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                    :, :, ::-1
                ].astype(np.uint8)
            )
            # images[-1] = cv2.hconcat([ori_image[-1], images[-1]])
        I2 = load_image(img_path2)
        padder = InputPadder(I0.shape, 32)
        I0, I2 = padder.pad(I0, I2)
        xs = torch.cat((I0.unsqueeze(2), I2.unsqueeze(2)), dim=2).to(
            device, non_blocking=True
        )
        _model.eval()
        batch_size = xs.shape[0]
        s_shape = xs.shape[-2:]

        _model.zero_grad()
        ds_factor = args.ds_factor
        with torch.no_grad():
            coord_inputs = [
                (
                    _model.sample_coord_input(
                        batch_size,
                        s_shape,
                        [1 / args.N * i],
                        device=xs.device,
                        upsample_ratio=ds_factor,
                    ),
                    None,
                )
                for i in range(1, args.N)
            ]
            timesteps = [
                i * 1 / args.N * torch.ones(xs.shape[0]).to(xs.device).to(torch.float)
                for i in range(1, args.N)
            ]
            all_outputs = _model(xs, coord_inputs, t=timesteps, ds_factor=ds_factor)
            out_frames = [padder.unpad(im) for im in all_outputs["imgt_pred"]]
            out_flowts = [padder.unpad(f) for f in all_outputs["flowt"]]
        flowt_imgs = [
            flow_to_image(
                flowt.squeeze().detach().cpu().permute(1, 2, 0).numpy(),
                convert_to_bgr=True,
            )
            for flowt in out_flowts
        ]
        I1_pred_img = [
            (I1_pred[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0)[
                :, :, ::-1
            ].astype(np.uint8)
            for I1_pred in out_frames
        ]

        for i in range(args.N - 1):
            images.append(I1_pred_img[i])
            flows.append(flowt_imgs[i])
            # images[-1] = cv2.hconcat([ori_image[-1], images[-1]])

        images.append(
            (
                (padder.unpad(I2)).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                * 255.0
            )[:, :, ::-1].astype(np.uint8)
        )
        ori_image.append(
            (
                (padder.unpad(I2)).squeeze().detach().cpu().numpy().transpose(1, 2, 0)
                * 255.0
            )[:, :, ::-1].astype(np.uint8)
        )
        # images[-1] = cv2.hconcat([ori_image[-1], images[-1]])

    _fps = args.N * 2
    if req.video:
        vfsp = _get_video_frame_rate(f"{src_path}/{vfilename}")
        _fps = min(vfsp * args.N, 60)

    images_to_video(images[:-1], output_path, fps=_fps)
    images_to_video(flows, flow_output_path, fps=_fps)
    print(len(images))

    return (output_path, flow_output_path)


multiprocessing.set_start_method("spawn", force=True)
executor = ProcessPoolExecutor(max_workers=2, initializer=_load_model)


def lifespan(_: FastAPI):
    yield
    executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(lifespan=lifespan)
app.mount("/videos", StaticFiles(directory=_OUTPUT_DIR), name="output")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/vfi")
async def vfi(req: VFIRequest):
    req.uid = str(uuid.uuid4()).replace("-", "")
    future = executor.submit(_run, req)

    def send_notify(f: Future[tuple[str, str]]):
        opath, fpath = f.result()
        if req.output_type == "url":
            ovideo = f"{_HTTP_PREFIX}/videos/{opath.lstrip(_OUTPUT_DIR)}"
            fvideo = f"{_HTTP_PREFIX}/videos/{fpath.lstrip(_OUTPUT_DIR)}"
        else:
            ovideo = (
                opath
                if opath.startswith("/")
                else f"{_PATH_PREFIX}{opath.lstrip(_WORK_DIR)}"
            )
            fvideo = (
                fpath
                if fpath.startswith("/")
                else f"{_PATH_PREFIX}{fpath.lstrip(_WORK_DIR)}"
            )

        data = {"task_id": req.uid, "ovideo": ovideo, "fvideo": fvideo}
        print(json.dumps(data))

        if req.notify_url:
            requests.post(req.notify_url, json=data)

    future.add_done_callback(send_notify)

    return {"task_id": req.uid}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8185)
