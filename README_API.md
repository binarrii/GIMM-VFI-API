## GIMM-VFI API

REST API for **GIMM-VFI**, with asynchronous queue, and asynchronous response

### Parameters

- Input

| Parameter     | Type                   | Default | Description                                                                 |
|---------------|------------------------|---------|-----------------------------------------------------------------------------|
| `images`      | Array[string] / string | null    | List of image URLs or a single image directory to process.                  |
| `video`       | string                 | null    | URL or path to the video file to process.                                   |
| `n`           | int                    | 8       | Number of frames to interpolate between each pair of input frames.          |
| `ds_factor`   | float                  | 1.0     | Downscale factor for processing images or video.                            |
| `output_type` | string                 | "path"  | Determines if the output is a file path or a URL.                           |
| `notify_url`  | string                 | null    | URL to send a notification to once processing is complete.                  |

**Note**:
*Choose either the `images` or `video` parameter, not both.*
*If you choose to use file paths as input or output, they must be absolute paths.*

- Output

| Parameter     | Type                   | Description                                                                           |
|---------------|------------------------|---------------------------------------------------------------------------------------|
| `task_id`     | string                 | Task ID represented as a string.                                                      |


### Examples

- List of image URLs as input

```bash

curl -s http://10.252.25.251:8185/vfi \
     --json '{
       "images": [
         "http://10.252.25.251:18080/vfi/frames/00020.png",
         "http://10.252.25.251:18080/vfi/frames/00028.png",
         "http://10.252.25.251:18080/vfi/frames/00036.png",
         "http://10.252.25.251:18080/vfi/frames/00044.png",
         "http://10.252.25.251:18080/vfi/frames/00052.png",
         "http://10.252.25.251:18080/vfi/frames/00060.png",
         "http://10.252.25.251:18080/vfi/frames/00068.png",
         "http://10.252.25.251:18080/vfi/frames/00076.png",
         "http://10.252.25.251:18080/vfi/frames/00084.png",
         "http://10.252.25.251:18080/vfi/frames/00092.png",
         "http://10.252.25.251:18080/vfi/frames/00100.png"
       ],
       "output_type": "url"
     }'

```

- Image directory as input, images must be sequentially named, requires shared storage, such as NAS

```bash

curl -s http://10.252.25.251:8185/vfi --json '{"images": "/app/demo/input_frames"}'

```

- Video file path as input, video URL as output, interpolation ratio `1:2`, output frame rate, e.g. `30 -> 60 fps`

```bash

curl -s http://10.252.25.251:8185/vfi --json '{"video": "/app/work/input/1029.mp4", "n": 2, "output_type": "url"}'

```

- Asynchronous notification, the `notify_url` request parameter must be provided

```json

{
  "task_id": "b86ed977735b48bd8b6142539f9e895f",
  "ovideo": "http://10.252.25.251:8185/videos/b86ed977735b48bd8b6142539f9e895f/o-b86ed977735b48bd8b6142539f9e895f.mp4",
  "fvideo": "http://10.252.25.251:8185/videos/b86ed977735b48bd8b6142539f9e895f/f-b86ed977735b48bd8b6142539f9e895f.mp4"
}

```
