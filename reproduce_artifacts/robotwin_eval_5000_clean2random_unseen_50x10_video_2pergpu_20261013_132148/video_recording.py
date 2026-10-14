"""Record a fresh head-camera frame per action without changing physics steps."""
from functools import wraps


def wrap_take_action(original):
    @wraps(original)
    def recorded(self, *args, **kwargs):
        path = self.eval_video_path
        if path is None:
            return original(self, *args, **kwargs)
        before = self.take_action_cnt
        if before == 0:
            self.eval_video_ffmpeg.stdin.write(self.get_obs()['observation']['head_camera']['rgb'].tobytes())
        # The original writes cached observations inside the physics loop.
        # Disable those writes, then capture the camera after the action.
        self.eval_video_path = None
        try:
            result = original(self, *args, **kwargs)
        finally:
            self.eval_video_path = path
        if self.take_action_cnt > before:
            self.eval_video_ffmpeg.stdin.write(self.get_obs()['observation']['head_camera']['rgb'].tobytes())
        return result
    return recorded
