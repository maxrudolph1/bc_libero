import os
import imageio
import numpy as np

from .visualization_utils import make_grid, save_numpy_as_video


def video_pad_time(videos):
    nframe = np.max([video.shape[0] for video in videos])
    padded = []
    for video in videos:
        npad = nframe - len(video)
        padded_frame = video[[-1], :, :, :].copy()
        video = np.vstack([video, np.tile(padded_frame, [npad, 1, 1, 1])])
        padded.append(video)
    return np.array(padded)


def make_grid_video_from_numpy(video_array, ncol, output_name='./output.mp4', speedup=1, padding=5, **kwargs):
    videos = []
    for video in video_array:
        if speedup != 1:
            video = video[::speedup]
        videos.append(video)
    videos = video_pad_time(videos)  # N x T x H x W x 3
    grid_frames = []
    for t in range(videos.shape[1]):
        grid_frame = make_grid(videos[:, t], ncol=ncol, padding=padding)
        grid_frames.append(grid_frame)
    save_numpy_as_video(np.array(grid_frames), output_name, **kwargs)


def rearrange_videos(videos, success, success_vid_first, fail_vid_first):
    success = np.array(success)
    rearrange_idx = np.arange(len(success))
    if success_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[success_idx], videos[fail_idx]], axis=0)
        rearrange_idx = np.concatenate([success_idx, fail_idx], axis=0)
    if fail_vid_first:
        success_idx = rearrange_idx[success]
        fail_idx = rearrange_idx[np.logical_not(success)]
        videos = np.concatenate([videos[fail_idx], videos[success_idx]], axis=0)
        rearrange_idx = np.concatenate([fail_idx, success_idx], axis=0)
    return videos, rearrange_idx


def render_done_to_boundary(frames, color=(0, 255, 0)):
    """
    If done, render a color boundary to the frame.
    Args:
        frame: (c, h, w)
        color: rgb value to illustrate success, default: (0, 255, 0)
    """
    """
    If done, render a color boundary to each frame in a batch.
    Args:
        frames: (t, c, h, w) - batch of frames
        color: rgb value to illustrate success, default: (0, 255, 0)
    Returns:
        Processed frames with boundaries.
    """
    t, c, h, w = frames.shape
    color = np.array(color, dtype=frames.dtype)[:, None, None]
    boundary = int(min(h, w) * 0.015)

    # Apply color boundary to all frames
    frames[:, :, :boundary, :] = color
    frames[:, :, -boundary:, :] = color
    frames[:, :, :, :boundary] = color
    frames[:, :, :, -boundary:] = color

    return frames


def save_numpy_to_video(video_array, output_path, num_terminated, fps=30):
    """
    Convert numpy array to video
    video_array: (B, T, C, H, W) or (B, T, H, W, C)
    """
    for i, video_data in enumerate(video_array):
        if video_data.shape[1] == 3: 
            video_data = video_data.transpose(0, 2, 3, 1)

        if num_terminated == []:
            final_path = os.path.join(output_path, f'{i}.mp4')
        else:
            final_path = os.path.join(output_path, f'{i}_{num_terminated[i]}.mp4')
        video_writer = imageio.get_writer(final_path, fps=fps)
        for im in video_data:
            video_writer.append_data(im)
        video_writer.close()
    
    print(f"Videos are saved successfully to {output_path}.")

    