import torch
import torch.utils.data

class FirstClipSampler(torch.utils.data.Sampler):
    """
    Samples at most `max_video_clips_per_video` clips for each video sequentially

    Arguments:
        video_clips (VideoClips): video clips to sample from
        max_clips_per_video (int): maximum number of clips to be sampled per video
    """
    def __init__(self, video_clips, max_clips_per_video):
        #if not isinstance(video_clips, visionmod.video_utils.VideoClips):
        #    raise TypeError("Expected video_clips to be an instance of VideoClips, "
        #                    "got {}".format(type(video_clips)))
        self.video_clips = video_clips
        self.max_clips_per_video = max_clips_per_video

    def __iter__(self):
        idxs = []
        s = 0
        # select at most max_clips_per_video for each video, sequentially
        for c in self.video_clips.clips:
            length = len(c)
            size = min(length, self.max_clips_per_video)
            sampled = torch.arange(length)[:size] + s # the only change from `RandomClipSampler`
            s += length
            idxs.append(sampled)
        idxs = torch.cat(idxs)
        # shuffle all clips randomly
        perm = torch.randperm(len(idxs))
        idxs = idxs[perm].tolist()
        return iter(idxs)

    def __len__(self):
        return sum(min(len(c), self.max_clips_per_video) for c in self.video_clips.clips)