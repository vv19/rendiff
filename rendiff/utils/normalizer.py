class Normalizer:
    def __init__(self, stats, device='cpu'):
        self.min = stats['min'].to(device)
        self.max = stats['max'].to(device)

        for key in stats.keys():
            stats[key] = stats[key].to(device)
        self.stats = stats

    def normalize_cam(self, x, camera_name):
        return (x - self.stats[f'{camera_name}_min']) / (
                self.stats[f'{camera_name}_max'] - self.stats[f'{camera_name}_min']) * 2 - 1

    def denormalize_cam(self, x, camera_name):
        return (x + 1) / 2 * (self.stats[f'{camera_name}_max'] - self.stats[f'{camera_name}_min']) + \
            self.stats[f'{camera_name}_min']

    def normalize(self, x):
        return (x - self.min[:x.shape[-1]]) / (self.max[:x.shape[-1]] - self.min[:x.shape[-1]]) * 2 - 1

    def denormalize(self, x):
        return (x + 1) / 2 * (self.max[:x.shape[-1]] - self.min[:x.shape[-1]]) + self.min[:x.shape[-1]]

    def normalize_action_pred(self, x):
        min_action = -(self.max[:x.shape[-1]] - self.min[:x.shape[-1]])
        max_action = self.max[:x.shape[-1]] - self.min[:x.shape[-1]]
        return (x - min_action) / (max_action - min_action) * 2 - 1

    def denormalize_action_pred(self, x):
        min_action = -(self.max[:x.shape[-1]] - self.min[:x.shape[-1]])
        max_action = self.max[:x.shape[-1]] - self.min[:x.shape[-1]]
        return (x + 1) / 2 * (max_action - min_action) + min_action


class IndividualNormalizer:
    def __init__(self, stats, device='cpu'):
        self.min = stats['min_ind'].unsqueeze(0).to(device)
        self.max = stats['max_ind'].unsqueeze(0).to(device)
        for key in stats.keys():
            stats[key] = stats[key].to(device)
        self.stats = stats

    def normalize_cam(self, x, camera_name):
        return (x - self.stats[f'{camera_name}_min_ind']) / (
                self.stats[f'{camera_name}_max_ind'] - self.stats[f'{camera_name}_min_ind']) * 2 - 1

    def denormalize_cam(self, x, camera_name):
        return (x + 1) / 2 * (self.stats[f'{camera_name}_max_ind'] - self.stats[f'{camera_name}_min_ind']) + \
            self.stats[f'{camera_name}_min_ind']

    def normalize(self, x):
        return (x - self.min[..., :x.shape[-1]]) / (self.max[..., :x.shape[-1]] - self.min[..., :x.shape[-1]]) * 2 - 1

    def denormalize(self, x):
        return (x + 1) / 2 * (self.max[..., :x.shape[-1]] - self.min[..., :x.shape[-1]]) + self.min[..., :x.shape[-1]]

    def normalize_action_pred(self, x):
        min_action = -(self.max[..., :x.shape[-1]] - self.min[..., :x.shape[-1]])
        max_action = self.max[..., :x.shape[-1]] - self.min[..., :x.shape[-1]]
        return (x - min_action) / (max_action - min_action) * 2 - 1

    def denormalize_action_pred(self, x):
        min_action = -(self.max[..., :x.shape[-1]] - self.min[..., :x.shape[-1]])
        max_action = self.max[..., :x.shape[-1]] - self.min[..., :x.shape[-1]]
        return (x + 1) / 2 * (max_action - min_action) + min_action
